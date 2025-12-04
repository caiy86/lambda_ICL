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
def generate_oracle_dataset_with_value(
    query_loader: DataLoader,
    corpus_data: List[Dict[str, str]],
    corpus_embeddings: torch.Tensor,
    embedding_model: EmbeddingModel,
    llm_wrapper: LLMWrapper,
    check_correct_fn: callable
) -> TensorDataset:

    logger.info(f"--- Generating Oracle Dataset (Grid Search & Value Estimation) ---")
    
    lambda_candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    candidate_indices = [int(round(x / 0.05)) for x in lambda_candidates]
    logger.info(f"Searching lambdas: {lambda_candidates} -> Indices: {candidate_indices}")

    all_query_embs = []
    all_target_dists = []
    all_value_targets = []
    all_actor_masks = []

    total_solvable = 0
    total_samples = 0

    for query_batch_list in tqdm(query_loader, desc="Oracle Searching"):
        batch_size = len(query_batch_list)
        
        query_texts = [item['query'] for item in query_batch_list]
        query_embs = embedding_model.encode(query_texts)
        
        query_indices = [item.get('corpus_index', -1) for item in query_batch_list]
        self_mask_indices = torch.tensor(query_indices, device=utils.device)
        has_valid_indices = (self_mask_indices >= 0).any()

        correctness_matrix = torch.zeros((batch_size, len(candidate_indices)), dtype=torch.bool, device=utils.device)

        for i, (lam_val, action_idx) in enumerate(zip(lambda_candidates, candidate_indices)):
            lambda_tensor = torch.full((batch_size, 1), lam_val, device=utils.device)
            batch_selected_indices = torch.zeros((batch_size, config.NUM_EXAMPLES), dtype=torch.long, device=utils.device)
            batch_selected_embs = torch.zeros((batch_size, config.NUM_EXAMPLES, embedding_model.dim), device=utils.device)
            sim_scores = torch.matmul(query_embs, corpus_embeddings.T)
            relevance_scores = sim_scores
            selected_mask = torch.zeros_like(sim_scores, dtype=torch.bool)
            
            if has_valid_indices:
                rows = torch.arange(batch_size, device=utils.device)
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
            
            generated_texts, _ = llm_wrapper.generate_for_evaluation(
                prompts, max_new_tokens=config.MAX_GEN_TOKENS
            )
            
            for b in range(batch_size):
                if check_correct_fn(target_answer=targets[b], pred_text=generated_texts[b]):
                    correctness_matrix[b, i] = True

        batch_target_dists = torch.zeros((batch_size, 21), device=utils.device)
        batch_value_targets = torch.full((batch_size,), -1.0, device=utils.device)
        batch_actor_masks = torch.zeros((batch_size,), device=utils.device)

        for b in range(batch_size):
            valid_indices_mask = correctness_matrix[b]
            if valid_indices_mask.any():
                total_solvable += 1
                batch_value_targets[b] = 1.0
                batch_actor_masks[b] = 1.0
                valid_indices = torch.tensor(candidate_indices, device=utils.device)[valid_indices_mask]
                prob = 1.0 / len(valid_indices)
                batch_target_dists[b].index_fill_(0, valid_indices, prob)
            else:
                pass

        all_query_embs.append(query_embs.cpu())
        all_target_dists.append(batch_target_dists.cpu())
        all_value_targets.append(batch_value_targets.cpu())
        all_actor_masks.append(batch_actor_masks.cpu())
        
        total_samples += batch_size

    all_query_embs = torch.cat(all_query_embs, dim=0)
    all_target_dists = torch.cat(all_target_dists, dim=0)
    all_value_targets = torch.cat(all_value_targets, dim=0)
    all_actor_masks = torch.cat(all_actor_masks, dim=0)
    
    solvable_rate = (total_solvable / total_samples) * 100 if total_samples > 0 else 0
    logger.info(f"Oracle Dataset Generated. Samples: {total_samples}, Solvable: {total_solvable} ({solvable_rate:.2f}%)")
    
    return TensorDataset(all_query_embs, all_target_dists, all_value_targets, all_actor_masks)
    
def pretrain_agent_with_value(agent: PolicyNetwork, optimizer: optim.Optimizer, pretrain_dataset: TensorDataset):
    logger.info(f"--- Starting Supervised Pre-training (Actor + Value) ---")

    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
    
    mse_loss_fn = torch.nn.MSELoss()
    
    for epoch in range(1, config.PRETRAIN_MAX_EPOCHS + 1):
        agent.train() 
        total_loss = 0.0
        total_actor_loss = 0.0
        total_value_loss = 0.0
        
        for query_embs, target_dists, value_targets, actor_masks in pretrain_loader:
            query_embs = query_embs.to(device)
            target_dists = target_dists.to(device)
            value_targets = value_targets.to(device)
            actor_masks = actor_masks.to(device)

            features = agent.feature_net(query_embs)
            logits = agent.actor_head(features)
            values = agent.value_head(features).squeeze(-1)

            bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            per_sample_bce_loss = bce_loss_fn(logits, target_dists)
            per_sample_actor_loss = per_sample_bce_loss.mean(dim=-1)
            actor_loss = (per_sample_actor_loss * actor_masks).sum() / (actor_masks.sum() + 1e-8)
            value_loss = mse_loss_fn(values, value_targets)
            loss = actor_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_value_loss += value_loss.item()
            
        avg_loss = total_loss / len(pretrain_loader)
        avg_actor = total_actor_loss / len(pretrain_loader)
        avg_value = total_value_loss / len(pretrain_loader)
        
        logger.info(f"Epoch {epoch}: Total Loss={avg_loss:.4f} (Actor={avg_actor:.4f}, Value={avg_value:.4f})")

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

    dataset_path = f"{config.CACHE_DIR}/pretrain_datasetv3.pt"
    if os.path.exists(dataset_path):
        logger.info(f"Loading existing dataset from {dataset_path}...")
        best_lambda_dataset = torch.load(dataset_path,weights_only=False)
    else:
        logger.info("Dataset not found. Generating new dataset...")

        pretrain_query_loader = dataloader.get_dataloader(
            split='train',batch_size=config.BATCH_SIZE, shuffle=True, nums=config.PRETRAIN_NUMS,seed=config.PRETRAIN_SEED 
        )
        
        best_lambda_dataset = generate_oracle_dataset_with_value(
            query_loader=pretrain_query_loader,
            corpus_data=corpus_data,
            corpus_embeddings=corpus_embeddings,
            embedding_model=embedding_model,
            llm_wrapper=llm_wrapper,
            check_correct_fn=dataloader.check_correct 
        )
        torch.save(best_lambda_dataset, dataset_path)
        logger.info(f"Dataset saved to {dataset_path}")

    optimizer = optim.AdamW(agent.parameters(), lr=config.PRETRAIN_LR, weight_decay=1e-4)
    pretrain_agent_with_value(agent, optimizer, best_lambda_dataset)

    logger.info("--- Evaluation ---")
    sampler = EpisodeSampler(agent, embedding_model, config.NUM_EXAMPLES)
    val_loader = dataloader.get_dataloader(split='dev', batch_size=config.BATCH_SIZE, shuffle=False, seed=None)
    
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