import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging

from models.policy_network import PolicyNetwork, RnnHiddenState
from models.embedding_model import EmbeddingModel

from utils import device 

logger = logging.getLogger(__name__)

class RolloutBuffer:
    def __init__(self, num_steps: int, batch_size: int, embedding_dim: int, device: torch.device):
        self.num_steps = num_steps 
        self.batch_size = batch_size 
        self.device = device

        self.query_embeddings = torch.zeros((batch_size, embedding_dim), device=device)
        self.example_embeddings = torch.zeros((batch_size, num_steps, embedding_dim), device=device)
        self.actions = torch.zeros((batch_size, num_steps), dtype=torch.long, device=device)
        
        self.log_probs: Optional[torch.Tensor] = torch.zeros((batch_size, num_steps), device=device)
        self.values: Optional[torch.Tensor] = torch.zeros((batch_size, num_steps), device=device)

        self.rewards = torch.zeros((batch_size, num_steps), device=device)
        self.advantages = torch.zeros((batch_size, num_steps), device=device)
        self.returns = torch.zeros((batch_size, num_steps), device=device)
        self.final_llm_loss = torch.zeros((batch_size,), device=device)
        
        self.queries: List[Dict] = []
        self.selected_examples_text: List[List[Dict]] = [[] for _ in range(batch_size)]

    def add_step_data(self, 
                      step: int, 
                      actions: torch.Tensor, 
                      example_embeddings: torch.Tensor,
                      example_texts: List[Dict],
                      log_probs: Optional[torch.Tensor] = None, 
                      values: Optional[torch.Tensor] = None):

        self.actions[:, step] = actions
        self.example_embeddings[:, step, :] = example_embeddings
        
        if log_probs is not None:
            self.log_probs[:, step] = log_probs
        if values is not None:
            self.values[:, step] = values
        
        for i in range(self.batch_size):
            self.selected_examples_text[i].append(example_texts[i])


class EpisodeSampler:
    def __init__(self, 
                 policy_network: PolicyNetwork, 
                 embedding_model: EmbeddingModel,
                 num_examples: int):
        
        self.policy_network = policy_network
        self.embedding_model = embedding_model
        self.num_examples = num_examples 
        self.device = device
        logger.info(f"EpisodeSampler initialized for {self.num_examples} steps.")

    @torch.no_grad()
    def collect_episodes(self, 
                         query_batch: List[Dict], 
                         corpus: List[Dict],
                         corpus_embeddings: torch.Tensor) -> RolloutBuffer:
        
        batch_size = len(query_batch)
        
        buffer = RolloutBuffer(
            num_steps=1,  
            batch_size=batch_size, 
            embedding_dim=self.embedding_model.dim,
            device=self.device
        )
        
        query_texts = [item['query'] for item in query_batch]
        buffer.queries = query_batch

        query_embeddings = self.embedding_model.encode(query_texts)
        buffer.query_embeddings = query_embeddings
        

        actions, log_probs, values, _ = self.policy_network(query_embeddings)

        lambda_vals = (actions.float() * 0.05).unsqueeze(1).clamp(0.0, 1.0)
        
        buffer.actions[:, 0] = actions
        buffer.log_probs[:, 0] = log_probs
        buffer.values[:, 0] = values

        num_examples = self.num_examples
        batch_selected_indices = torch.zeros((batch_size, num_examples), dtype=torch.long, device=self.device)
        batch_selected_embs = torch.zeros((batch_size, num_examples, self.embedding_model.dim), device=self.device)
        
        sim_scores = torch.matmul(query_embeddings, corpus_embeddings.T)
        relevance_scores = sim_scores

        selected_mask = torch.zeros_like(sim_scores, dtype=torch.bool)

        for t in range(num_examples):
            if t == 0:
                step_scores = relevance_scores
            else:

                selected_embs_so_far = batch_selected_embs[:, :t, :]
                corpus_expanded = corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                sim_to_selected = torch.bmm(corpus_expanded, selected_embs_so_far.transpose(1, 2))
                diversity_penalty, _ = torch.max(sim_to_selected, dim=2)
                
                step_scores = (lambda_vals * relevance_scores) - \
                              ((1 - lambda_vals) * diversity_penalty)

            step_scores.masked_fill_(selected_mask, -torch.inf)

            current_action = torch.argmax(step_scores, dim=1) # (B,)

            current_embs = corpus_embeddings[current_action]
            batch_selected_indices[:, t] = current_action
            batch_selected_embs[:, t, :] = current_embs
            selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)

        buffer.mmr_selected_embeddings = batch_selected_embs # Shape (B, K, D)

        for i in range(batch_size):
            indices = batch_selected_indices[i].tolist()
            buffer.selected_examples_text[i] = [corpus[idx] for idx in indices]
            
        logger.debug(f"Collected batch with adaptive lambdas: {lambda_vals.flatten()[:5]}...")
        return buffer