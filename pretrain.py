import torch
from torch.nn import Dropout
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import pandas as pd 
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import train_config as config
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.sampler import EpisodeSampler, RolloutBuffer
from utils import device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_evaluation(llm_wrapper: LLMWrapper,
                   val_loader: DataLoader, 
                   corpus_data: List[Dict[str, str]],
                   corpus_embeddings: torch.Tensor,
                   check_correct_fn: callable,
                   system_prompt: str,
                   sampler: EpisodeSampler,
                   desc: str = "Eval") -> float:
    
    sampler.policy_network.eval()
    
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(val_loader, desc=desc, leave=False):
        curr_batch_size = len(batch)
        
        buffer = sampler.collect_episodes(
            query_batch=batch,
            corpus=corpus_data,
            corpus_embeddings=corpus_embeddings
        )
        
        prompts = []
        targets = []
        for i in range(curr_batch_size):
            query_data = buffer.queries[i]
            example_data = buffer.selected_examples_text[i]
            
            prompt_str = llm_wrapper.build_chat_prompt(
                system_prompt=system_prompt,
                examples=example_data,
                query=query_data['query'],
            )
            prompts.append(prompt_str)
            targets.append(query_data['answer'])

        generated_texts, _ = llm_wrapper.generate_for_evaluation(
            prompts, max_new_tokens=config.MAX_GEN_TOKENS
        )
        
        for i in range(curr_batch_size):
            if check_correct_fn(target_answer=targets[i], pred_text=generated_texts[i]):
                total_correct += 1
        
        total_samples += curr_batch_size

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    return accuracy

@torch.no_grad()
def generate_data(query_loader: DataLoader,
                  corpus_data: List[Dict[str, str]],
                  corpus_embeddings: torch.Tensor,
                  embedding_model: EmbeddingModel,
                  llm_wrapper: LLMWrapper,
                  check_correct_fn: callable) -> TensorDataset:

    logger.info(f"--- Generating Oracle Dataset (Full Scan 21 Actions & Soft Labels) ---")
    
    all_indices = list(range(21)) 
    lambda_values = [round(i * 0.05, 2) for i in all_indices]
    logger.info(f"Scanning all lambdas: {lambda_values}")

    all_query_embs = []
    all_target_dists = [] 
    all_value_targets = []  
    all_actor_masks = []  

    total_samples = 0
    avg_max_prob = 0
    TEMPERATURE = 0.5

    for query_batch_list in tqdm(query_loader, desc="Oracle Searching"):
        batch_size = len(query_batch_list)

        query_texts = [item['query'] for item in query_batch_list]
        query_embs = embedding_model.encode(query_texts) # (B, 384)
        
        sim_scores_base = torch.matmul(query_embs, corpus_embeddings.T)
        
        batch_loss_matrix = torch.full((batch_size, 21), float('inf'), device=utils.device)

        for action_idx, lam_val in enumerate(lambda_values):
            lambda_tensor = torch.full((batch_size, 1), lam_val, device=utils.device)
            
            batch_selected_indices = torch.zeros((batch_size, config.NUM_EXAMPLES), dtype=torch.long, device=utils.device)
            batch_selected_embs = torch.zeros((batch_size, config.NUM_EXAMPLES, embedding_model.dim), device=utils.device)
            
            relevance_scores = sim_scores_base
            selected_mask = torch.zeros_like(relevance_scores, dtype=torch.bool)
            
            query_indices = [item.get('corpus_index', -1) for item in query_batch_list]
            self_mask_indices = torch.tensor(query_indices, device=utils.device)
            if (self_mask_indices >= 0).any():
                rows = torch.arange(batch_size, device=utils.device)
                valid = self_mask_indices >= 0
                selected_mask[rows[valid], self_mask_indices[valid]] = True

            for t in range(config.NUM_EXAMPLES):
                if t == 0:
                    step_scores = relevance_scores
                else:
                    selected_embs_so_far = batch_selected_embs[:, :t, :]
                    # (B, 1, Corpus)
                    corpus_expanded = corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                    # (B, Corpus, t) = (B, Corpus, D) @ (B, D, t)
                    sim_to_selected = torch.bmm(corpus_expanded, selected_embs_so_far.transpose(1, 2))
                    diversity_penalty, _ = torch.max(sim_to_selected, dim=2)
                    
                    step_scores = (lambda_tensor * relevance_scores) - ((1 - lambda_tensor) * diversity_penalty)
                
                step_scores.masked_fill_(selected_mask, -torch.inf)
                current_action = torch.argmax(step_scores, dim=1)
                
                batch_selected_indices[:, t] = current_action
                batch_selected_embs[:, t, :] = corpus_embeddings[current_action]
                selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)
          
            prompts = []
            targets = []
            for b in range(batch_size):
                selected_indices = batch_selected_indices[b].cpu().tolist()
                example_data = [corpus_data[idx] for idx in selected_indices]
                prompt_str = llm_wrapper.build_chat_prompt(
                    system_prompt=config.SYSTEM_PROMPT,
                    examples=example_data,
                    query=query_batch_list[b]['query']
                )
                prompts.append(prompt_str)
                targets.append(query_batch_list[b]['answer'])
            
            losses = llm_wrapper.get_batch_loss(prompts, targets) # (B,)
            batch_loss_matrix[:, action_idx] = losses.to(utils.device)

        batch_loss_matrix = torch.nan_to_num(batch_loss_matrix, nan=1e9, posinf=1e9)

        neg_loss = -batch_loss_matrix
        target_dists = F.softmax(neg_loss / TEMPERATURE, dim=1) # (B, 21)

        curr_max_prob = target_dists.max(dim=1)[0].mean().item()
        avg_max_prob = (avg_max_prob * total_samples + curr_max_prob * batch_size) / (total_samples + batch_size)

        min_losses, _ = batch_loss_matrix.min(dim=1)
        probs = torch.exp(-min_losses)
        batch_value_targets = 2.0 * probs - 1.0

        batch_actor_masks = torch.ones(batch_size, device=utils.device)

        all_query_embs.append(query_embs.cpu())
        all_target_dists.append(target_dists.cpu())
        all_value_targets.append(batch_value_targets.cpu())
        all_actor_masks.append(batch_actor_masks.cpu())
        
        total_samples += batch_size

    logger.info(f"Oracle Generated. Avg Max Prob: {avg_max_prob:.4f} (Should be > 0.5)")
    if avg_max_prob < 0.2:
        logger.warning("Warning: Oracle labels are too flat! Consider lowering TEMPERATURE further.")
        
    dataset = TensorDataset(
        torch.cat(all_query_embs, dim=0),
        torch.cat(all_target_dists, dim=0),
        torch.cat(all_value_targets, dim=0),
        torch.cat(all_actor_masks, dim=0)
    )
    
    logger.info(f"Oracle Dataset Generated. Total Samples: {total_samples}")
    return dataset
    
def pretrain_agent_with_value(agent: PolicyNetwork, optimizer: optim.Optimizer, pretrain_dataset: TensorDataset):
    logger.info(f"--- Starting Supervised Pre-training (Soft Label + Value) ---")

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
    
    mse_loss_fn = torch.nn.MSELoss()
    
    for epoch in range(1, config.PRETRAIN_MAX_EPOCHS + 1):
        agent.train() 
        stats = {"loss": 0.0, "actor": 0.0, "value": 0.0, "ent_target": 0.0, "ent_pred": 0.0}
        
        for query_embs, target_dists, value_targets, actor_masks in pretrain_loader:
            query_embs = query_embs.to(device)
            target_dists = target_dists.to(device)   # (B, 21)
            value_targets = value_targets.to(device) # (B,)
            actor_masks = actor_masks.to(device)
            
            actor_logits, pred_values = agent.get_logits_and_values(query_embs)

            log_probs = F.log_softmax(actor_logits, dim=-1)
            per_sample_actor_loss = -(target_dists * log_probs).sum(dim=-1)
            
            actor_loss = (per_sample_actor_loss * actor_masks).sum() / (actor_masks.sum() + 1e-8)
            
            value_loss = mse_loss_fn(pred_values, value_targets)
            
            loss = actor_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            with torch.no_grad():
                ent_target = -(target_dists * torch.log(target_dists + 1e-9)).sum(dim=-1).mean()
                probs = F.softmax(actor_logits, dim=-1)
                ent_pred = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

            stats["loss"] += loss.item()
            stats["actor"] += actor_loss.item()
            stats["value"] += value_loss.item()
            stats["ent_target"] += ent_target.item()
            stats["ent_pred"] += ent_pred.item()
            
        for k in stats: stats[k] /= len(pretrain_loader)
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}: Total={stats['loss']:.4f} | Actor={stats['actor']:.4f} | Value={stats['value']:.4f}")
            # 如果 Ent_Pred 远大于 Ent_Target，说明模型欠拟合
            logger.info(f"    Entropy: Target={stats['ent_target']:.4f} vs Pred={stats['ent_pred']:.4f}")

def main():

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    utils.setup_logging(log_level="INFO", log_file=os.path.join(config.LOG_DIR, f"pretrain_{config.RUN_NAME}.log"))
    utils.initialize_seeds(config.SEED)
    logger.info(f"Using device: {device}")

    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(model_name=config.LLM_MODEL_NAME)
    
    corpus_data, corpus_embeddings = dataloader.get_corpus() 
    corpus_embeddings_cpu = corpus_embeddings.cpu()
    
    logger.info("--- Initializing RBF Centers with K-Means on Corpus ---")
    logger.info(f"Corpus shape for K-Means: {corpus_embeddings_cpu.shape}")

    RBF_NUM_CENTERS = 1024 
    
    kmeans = MiniBatchKMeans(
        n_clusters=RBF_NUM_CENTERS, 
        batch_size=1024, 
        random_state=config.SEED,
        n_init='auto'
    )
    kmeans.fit(corpus_embeddings_cpu)
    
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    logger.info(f"K-Means converged. Centers shape: {cluster_centers.shape}")

    from models.policy_network import RBFPolicyNetwork
    agent = RBFPolicyNetwork(
        embedding_dim=embedding_model.dim,
        num_centers=RBF_NUM_CENTERS,
        dropout = config.AGENT_DROPOUT
    ).to(device)
    agent.centers.data = cluster_centers.to(device)
    logger.info("RBFPolicyNetwork initialized with K-Means centers from Corpus.")
        
    dataset_path = f"{config.CACHE_DIR}/pretrain_dataset_{config.TEMPERATURE}.pt"
    if os.path.exists(dataset_path):
        logger.info(f"Loading existing dataset from {dataset_path}...")
        best_lambda_dataset = torch.load(dataset_path, weights_only=False)
    else:
        logger.info("Dataset not found. Generating new Oracle dataset (Full Scan)...")

        pretrain_query_loader = dataloader.get_dataloader(
            split='train',
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            nums=config.PRETRAIN_NUMS,
            seed=config.PRETRAIN_SEED 
        )
        
        best_lambda_dataset = generate_data(
            query_loader=pretrain_query_loader,
            corpus_data=corpus_data,
            corpus_embeddings=corpus_embeddings,
            embedding_model=embedding_model,
            llm_wrapper=llm_wrapper,
            check_correct_fn=dataloader.check_correct 
        )
        torch.save(best_lambda_dataset, dataset_path)
        logger.info(f"Dataset saved to {dataset_path}")

    optimizer = optim.AdamW(agent.parameters(), lr=config.PRETRAIN_LR)

    pretrain_agent_with_value(agent, optimizer, best_lambda_dataset)

    logger.info("--- Evaluation ---")
    sampler = EpisodeSampler(agent, embedding_model, config.NUM_EXAMPLES)
    val_loader = dataloader.get_dataloader(split='dev', batch_size=config.BATCH_SIZE, shuffle=False, seed=None)
    
    accuracy = run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,      
        corpus_data=corpus_data,       
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        sampler=sampler
    )

    logger.info(f"Validation accuracy: {accuracy:.2f}%")

    torch.save(agent.state_dict(), f"{config.CACHE_DIR}/pre_mdl_RBF_{config.RUN_NAME}.pt")
    logger.info(f"Policy saved to pre_train_RBF.pt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Error", exc_info=True)
        raise e