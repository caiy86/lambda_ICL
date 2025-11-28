import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import pandas as pd 
import numpy as np

from config import train_config as config
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.sampler import EpisodeSampler, RolloutBuffer
from utils import device

logger = logging.getLogger(__name__)

@torch.no_grad()
def run_evaluation(llm_wrapper: LLMWrapper,
                   val_loader: torch.utils.data.DataLoader,
                   corpus_data: List[Dict[str, str]],
                   corpus_embeddings: torch.Tensor,
                   check_correct_fn: callable,
                   system_prompt: str,
                   prompt_strategy: str,
                   mode: str,
                   sampler: Optional[EpisodeSampler] = None) -> float:
    
    logger.info(f"Starting evaluation (Mode: {mode})...")
    if mode == 'policy':
        if sampler is None:
            raise ValueError("Sampler required for policy evaluation.")
        sampler.policy_network.eval()
    
    total_correct = 0
    total_nll = 0.0 
    total_samples = 0
    
    for query_batch_list in tqdm(val_loader, desc=f"Validating ({mode})"):
        batch_size = len(query_batch_list)
        
        buffer = sampler.collect_episodes(
            query_batch=query_batch_list,
            corpus=corpus_data,
            corpus_embeddings=corpus_embeddings
        )
        
        prompts = []
        targets = []
        for i in range(batch_size):
            query_data = buffer.queries[i]
            example_data = buffer.selected_examples_text[i]
            
            prompt_str = llm_wrapper.build_chat_prompt(
                system_prompt=system_prompt,
                examples=example_data,
                query=query_data['query'],
                strategy=prompt_strategy
            )
            prompts.append(prompt_str)
            targets.append(query_data['answer'])

        generated_texts, generated_nlls = llm_wrapper.generate_for_evaluation(
            prompts, max_new_tokens=config.MAX_GEN_TOKENS
        )
        
        nlls_list = generated_nlls.cpu().tolist() if isinstance(generated_nlls, torch.Tensor) else generated_nlls

        for i in range(batch_size):
            if check_correct_fn(target_answer=targets[i], pred_text=generated_texts[i]):
                total_correct += 1
            total_nll += nlls_list[i] 
        
        total_samples += batch_size

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    avg_nll = total_nll / total_samples if total_samples > 0 else float('inf')
    
    logger.info(f"Eval Finished. Acc: {accuracy:.2f}%, Avg NLL: {avg_nll:.4f}")
    return accuracy

@torch.no_grad()
def generate_best_lambda_dataset(
    query_loader: DataLoader,
    corpus_data: List[Dict[str, str]],
    corpus_embeddings: torch.Tensor,
    embedding_model: EmbeddingModel,
    llm_wrapper: LLMWrapper,
    num_candidates: int = 10  
) -> TensorDataset:

    logger.info(f"--- Generating Optimal Lambda Dataset (Iterative Search K={num_candidates}) ---")

    all_query_embs = []
    all_best_actions = []
    
    weights = torch.ones(21, device=utils.device)
    weights[11:] = 3.0 
    
    for query_batch_list in tqdm(query_loader, desc="Searching optimal lambdas"):
        real_batch_size = len(query_batch_list)
        
        query_texts = [item['query'] for item in query_batch_list]
        query_embs = embedding_model.encode(query_texts) # (B, D)

        batch_min_losses = torch.full((real_batch_size,), float('inf'), dtype=torch.bfloat16, device=utils.device)
        batch_best_actions = torch.zeros((real_batch_size,), dtype=torch.long, device=utils.device)

        query_indices = [item.get('corpus_index', -1) for item in query_batch_list]
        self_mask_indices = torch.tensor(query_indices, device=utils.device) # (B,)
        has_valid_indices = (self_mask_indices >= 0).any()
        
        for _ in range(num_candidates):

            curr_actions = torch.multinomial(weights, real_batch_size, replacement=True) # (B,)
            lambda_vals = (curr_actions.float() * 0.05).unsqueeze(1) # (B, 1)

            batch_selected_indices = torch.zeros((real_batch_size, config.NUM_EXAMPLES), dtype=torch.long, device=utils.device)
            batch_selected_embs = torch.zeros((real_batch_size, config.NUM_EXAMPLES, embedding_model.dim), device=utils.device)
            
            sim_scores = torch.matmul(query_embs, corpus_embeddings.T) # (B, Corpus)
            relevance_scores = sim_scores
            selected_mask = torch.zeros_like(sim_scores, dtype=torch.bool)

            if has_valid_indices:
                rows = torch.arange(real_batch_size, device=utils.device)
                valid = self_mask_indices >= 0
                if valid.any():
                    selected_mask[rows[valid], self_mask_indices[valid]] = True

            for t in range(config.NUM_EXAMPLES):
                if t == 0:
                    step_scores = relevance_scores
                else:
                    selected_embs_so_far = batch_selected_embs[:, :t, :]
                    sim_to_selected = torch.matmul(selected_embs_so_far, corpus_embeddings.T)
                    diversity_penalty, _ = torch.max(sim_to_selected, dim=1)
                    step_scores = (lambda_vals * relevance_scores) - ((1 - lambda_vals) * diversity_penalty)

                step_scores.masked_fill_(selected_mask, -torch.inf)
                current_action = torch.argmax(step_scores, dim=1)
                
                batch_selected_indices[:, t] = current_action
                batch_selected_embs[:, t, :] = corpus_embeddings[current_action]
                selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)

            prompts = []
            targets = []
            for i in range(real_batch_size):
                selected_indices = batch_selected_indices[i].cpu().tolist()
                example_data = [corpus_data[idx] for idx in selected_indices]
                prompt_str = llm_wrapper.build_chat_prompt(
                    system_prompt=config.SYSTEM_PROMPT,
                    examples=example_data,
                    query=query_batch_list[i]['query']
                )
                prompts.append(prompt_str)
                targets.append(query_batch_list[i]['answer'])
            
            curr_losses = llm_wrapper.get_batch_loss(prompts, targets).to(utils.device) # (B,)

            update_mask = curr_losses < batch_min_losses
            batch_min_losses[update_mask] = curr_losses[update_mask]
            batch_best_actions[update_mask] = curr_actions[update_mask]

        all_query_embs.append(query_embs.cpu())
        all_best_actions.append(batch_best_actions.cpu())
        
    all_query_embs = torch.cat(all_query_embs, dim=0)
    all_best_actions = torch.cat(all_best_actions, dim=0)
    
    logger.info(f"Generated Dataset: {len(all_query_embs)} samples.")
    logger.info(f"Action Dist: \n{pd.Series(all_best_actions.numpy()).value_counts().sort_index()}")
    
    return TensorDataset(all_query_embs, all_best_actions)

def pretrain_agent(agent: PolicyNetwork, optimizer: optim.Optimizer, pretrain_dataset: TensorDataset):
    logger.info(f"--- Starting Supervised Pre-training ---")

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
    crit_loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, config.PRETRAIN_MAX_EPOCHS + 1):
        agent.train() 
        total_loss = 0.0
        correct = 0
        total = 0
        
        for query_embs, target_actions in pretrain_loader:
            query_embs, target_actions = query_embs.to(device), target_actions.to(device)

            features = agent.feature_net(query_embs)
            logits = agent.actor_head(features)
            loss = crit_loss_fn(logits, target_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target_actions).sum().item()
            total += target_actions.size(0)
            
        avg_loss = total_loss / len(pretrain_loader)
        logger.info(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Accuracy = {correct/total:.4f} ({correct}/{total})")
        if avg_loss < config.PRETRAIN_LOSS_THRESHOLD:
            logger.info("Loss threshold reached.")
            break

def main():

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    utils.setup_logging(log_level="INFO", log_file=os.path.join(config.LOG_DIR, f"pretrain_{config.RUN_NAME}.log"))
    # utils.initialize_seeds(config.SEED)
    logger.info(f"Using device: {device}")

    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(model_name=config.LLM_MODEL_NAME)
    
    agent = PolicyNetwork(
        embedding_dim=embedding_model.dim,
        hidden_dim=config.AGENT_HIDDEN_DIM,
        dropout=config.AGENT_DROPOUT
    ).to(device)
    
    corpus_data, corpus_embeddings_cpu = dataloader.get_corpus() 
    corpus_embeddings = corpus_embeddings_cpu.to(device)

    dataset_path = f"{config.CACHE_DIR}/pretrain_dataset.pt"
    if os.path.exists(dataset_path):
        logger.info(f"Loading existing dataset from {dataset_path}...")
        best_lambda_dataset = torch.load(dataset_path)
    else:
        logger.info("Dataset not found. Generating new dataset...")

        pretrain_query_loader = dataloader.get_dataloader(
            split='train',batch_size=config.BATCH_SIZE, shuffle=True, nums=config.PRETRAIN_NUMS,seed=None 
        )
        
        best_lambda_dataset = generate_best_lambda_dataset(
            query_loader=pretrain_query_loader,
            corpus_data=corpus_data,
            corpus_embeddings=corpus_embeddings,
            embedding_model=embedding_model,
            llm_wrapper=llm_wrapper,
            num_candidates=10 
        )
        torch.save(best_lambda_dataset, dataset_path)
        logger.info(f"Dataset saved to {dataset_path}")

    optimizer = optim.AdamW(agent.parameters(), lr=config.PRETRAIN_LR) 
    pretrain_agent(agent, optimizer, best_lambda_dataset)

    logger.info("--- Evaluation ---")
    sampler = EpisodeSampler(agent, embedding_model, config.NUM_EXAMPLES)
    val_loader = dataloader.get_dataloader(split='dev', batch_size=config.BATCH_SIZE, shuffle=False,nums=128,seed=None)
    
    run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,      
        corpus_data=corpus_data,       
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        mode='policy', 
        sampler=sampler
    )

    torch.save(agent.state_dict(), f"{config.CACHE_DIR}/pre_mdl_{config.RUN_NAME}.pt")
    logger.info(f"Pre-trained model saved to: {config.CACHE_DIR}/pre_mdl_{config.RUN_NAME}.pt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Error", exc_info=True)
        raise e