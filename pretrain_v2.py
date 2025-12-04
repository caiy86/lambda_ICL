import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from tqdm import tqdm
from typing import List, Dict

from config import train_config as config
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.sampler import EpisodeSampler
from utils import device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

@torch.no_grad()
def run_evaluation(llm_wrapper: LLMWrapper,
                   query_list: List[Dict], 
                   corpus_data: List[Dict[str, str]],
                   corpus_embeddings: torch.Tensor,
                   check_correct_fn: callable,
                   system_prompt: str,
                   sampler: EpisodeSampler,
                   desc: str = "Eval") -> float:
    
    sampler.policy_network.eval()
    
    total_correct = 0
    total_samples = 0

    batch_size = config.BATCH_SIZE

    batches = [query_list[i:i + batch_size] for i in range(0, len(query_list), batch_size)]
    
    for batch in tqdm(batches, desc=desc, leave=False):
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
def generate_oracle_tensor(
    query_list: List[Dict],
    corpus_data: List[Dict],
    corpus_embeddings: torch.Tensor,
    embedding_model: EmbeddingModel,
    llm_wrapper: LLMWrapper,
    check_correct_fn: callable
) -> TensorDataset:
    
    logger.info(f"Generating Oracle Labels for {len(query_list)} samples...")
    
    lambda_candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    candidate_indices = [int(round(x / 0.05)) for x in lambda_candidates]
    
    all_query_embs = []
    all_target_dists = []
    all_value_targets = []
    all_actor_masks = []
    
    batch_size = config.BATCH_SIZE
    batches = [query_list[i:i + batch_size] for i in range(0, len(query_list), batch_size)]
    
    total_solvable = 0
    
    for batch in tqdm(batches, desc="Oracle Labeling"):
        curr_bs = len(batch)
        query_texts = [item['query'] for item in batch]
        query_embs = embedding_model.encode(query_texts)
        
        # Self-mask logic
        query_indices = [item.get('corpus_index', -1) for item in batch]
        self_mask_indices = torch.tensor(query_indices, device=utils.device)
        has_valid_indices = (self_mask_indices >= 0).any()
        
        correctness_matrix = torch.zeros((curr_bs, len(candidate_indices)), dtype=torch.bool, device=utils.device)
        
        for i, (lam_val, _) in enumerate(zip(lambda_candidates, candidate_indices)):
            lambda_tensor = torch.full((curr_bs, 1), lam_val, device=utils.device)

            batch_selected_indices = torch.zeros((curr_bs, config.NUM_EXAMPLES), dtype=torch.long, device=utils.device)
            batch_selected_embs = torch.zeros((curr_bs, config.NUM_EXAMPLES, embedding_model.dim), device=utils.device)

            sim_scores = torch.matmul(query_embs, corpus_embeddings.T)
            relevance_scores = sim_scores.clone()

            selected_mask = torch.zeros_like(sim_scores, dtype=torch.bool)
            if has_valid_indices:
                rows = torch.arange(curr_bs, device=utils.device)
                valid = self_mask_indices >= 0
                selected_mask[rows[valid], self_mask_indices[valid]] = True

            for t in range(config.NUM_EXAMPLES):
                if t == 0:
                    step_scores = relevance_scores.clone() # Clone to avoid modifying original
                else:
                    selected_embs_so_far = batch_selected_embs[:, :t, :]
                    
                    sim_to_selected = torch.matmul(selected_embs_so_far, corpus_embeddings.T)
                    
                    diversity_penalty, _ = torch.max(sim_to_selected, dim=1)
                    
                    step_scores = (lambda_tensor * relevance_scores) - ((1 - lambda_tensor) * diversity_penalty)
                
                step_scores.masked_fill_(selected_mask, -torch.inf)
              
                current_action = torch.argmax(step_scores, dim=1) # (Batch,)
               
                batch_selected_indices[:, t] = current_action
                batch_selected_embs[:, t, :] = corpus_embeddings[current_action]
                
                selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)

            prompts = []
            targets = []
            for b in range(curr_bs):
                indices = batch_selected_indices[b].cpu().tolist()
                example_data = [corpus_data[idx] for idx in indices]
                prompt = llm_wrapper.build_chat_prompt(config.SYSTEM_PROMPT, example_data, batch[b]['query'])
                prompts.append(prompt)
                targets.append(batch[b]['answer'])
            
            gen_texts, _ = llm_wrapper.generate_for_evaluation(prompts, max_new_tokens=config.MAX_GEN_TOKENS)
            
            for b in range(curr_bs):
                if check_correct_fn(targets[b], gen_texts[b]):
                    correctness_matrix[b, i] = True
        
        batch_target_dists = torch.zeros((curr_bs, 21), device=utils.device)
        batch_value_targets = torch.full((curr_bs,), -1.0, device=utils.device)
        batch_actor_masks = torch.zeros((curr_bs,), device=utils.device)
        
        for b in range(curr_bs):
            valid_mask = correctness_matrix[b]
            if valid_mask.any():
                total_solvable += 1
                batch_value_targets[b] = 1.0
                batch_actor_masks[b] = 1.0
                valid_indices = torch.tensor(candidate_indices, device=utils.device)[valid_mask]
                batch_target_dists[b].index_fill_(0, valid_indices, 1.0)
        
        all_query_embs.append(query_embs.cpu())
        all_target_dists.append(batch_target_dists.cpu())
        all_value_targets.append(batch_value_targets.cpu())
        all_actor_masks.append(batch_actor_masks.cpu())

    logger.info(f"Oracle Solvability: {total_solvable}/{len(query_list)} ({(total_solvable/len(query_list))*100:.2f}%)")
    
    return TensorDataset(
        torch.cat(all_query_embs),
        torch.cat(all_target_dists),
        torch.cat(all_value_targets),
        torch.cat(all_actor_masks)
    )

def get_split_and_oracle_data(corpus_data, corpus_embeddings, embedding_model, llm_wrapper):

    cache_path = f"{config.CACHE_DIR}/pretrain_split_cache.pt"
    
    if os.path.exists(cache_path):
        logger.info(f"Loading cached split data from {cache_path}...")
        data_dict = torch.load(cache_path, weights_only=False)
        return data_dict['train_raw'], data_dict['val_raw'], data_dict['train_tensor']
    
    logger.info("Cache not found. Generating new split and oracle data...")
    
    total_nums = config.PRETRAIN_NUMS + 128

    raw_loader = dataloader.get_dataloader(
        split='train', batch_size=total_nums, shuffle=True, nums=total_nums, seed=config.PRETRAIN_SEED
    )

    all_samples = [item for batch in raw_loader for item in batch] 
    all_samples = all_samples[:total_nums]
    
    train_raw = all_samples[:config.PRETRAIN_NUMS]
    val_raw = all_samples[config.PRETRAIN_NUMS:]
    logger.info(f"Split sizes -> Train: {len(train_raw)}, Val: {len(val_raw)}")
    
    train_tensor = generate_oracle_tensor(
        train_raw, corpus_data, corpus_embeddings, embedding_model, llm_wrapper, dataloader.check_correct
    )
    
    data_dict = {
        'train_raw': train_raw,
        'val_raw': val_raw,
        'train_tensor': train_tensor
    }
    torch.save(data_dict, cache_path)
    logger.info(f"Data cached to {cache_path}")
    
    return train_raw, val_raw, train_tensor

def main():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    utils.setup_logging(log_level="INFO", log_file=os.path.join(config.LOG_DIR, f"pretrain_{config.RUN_NAME}.log"))

    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(model_name=config.LLM_MODEL_NAME)
    
    agent = PolicyNetwork(
        embedding_dim=embedding_model.dim,
        hidden_dim=config.AGENT_HIDDEN_DIM,
        dropout=config.AGENT_DROPOUT
    ).to(device)
    
    optimizer = optim.AdamW(agent.parameters(), lr=config.PRETRAIN_LR, weight_decay=1e-3)
    
    corpus_data, corpus_embeddings_cpu = dataloader.get_corpus() 
    corpus_embeddings = corpus_embeddings_cpu.to(device)

    train_raw, val_raw, train_tensor = get_split_and_oracle_data(
        corpus_data, corpus_embeddings, embedding_model, llm_wrapper
    )
    
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True, drop_last=True)
    
    fixed_train_batch = train_raw[:64]
    
    sampler = EpisodeSampler(agent, embedding_model, config.NUM_EXAMPLES)
    
    logger.info("--- Starting Pretraining ---")
    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    best_val_acc = 0.0
    
    for epoch in range(1, config.PRETRAIN_MAX_EPOCHS + 1):
        agent.train()
        total_loss = 0.0
        
        for query_embs, target_dists, value_targets, actor_masks in train_loader:
            query_embs = query_embs.to(device)
            target_dists = target_dists.to(device) # shape: (B, 21), 0 or 1
            value_targets = value_targets.to(device)
            actor_masks = actor_masks.to(device)

            features = agent.feature_net(query_embs)
            logits = agent.actor_head(features)
            values = agent.value_head(features).squeeze(-1)

            per_sample_bce = bce_loss_fn(logits, target_dists) # (B, 21)
            per_sample_actor_loss = per_sample_bce.mean(dim=-1) # Mean over classes
            actor_loss = (per_sample_actor_loss * actor_masks).sum() / (actor_masks.sum() + 1e-8)
            
            # Value Loss
            value_loss = mse_loss_fn(values, value_targets)
            
            loss = actor_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.4f}")
        
        if epoch % 50 == 0:
            train_acc = run_evaluation(
                llm_wrapper, fixed_train_batch, corpus_data, corpus_embeddings,
                dataloader.check_correct, config.SYSTEM_PROMPT, sampler, desc="Eval Train"
            )

            val_acc = run_evaluation(
                llm_wrapper, val_raw, corpus_data, corpus_embeddings,
                dataloader.check_correct, config.SYSTEM_PROMPT, sampler, desc="Eval Val"
            )
            
            logger.info(f"Epoch {epoch} Result | Loss: {avg_loss:.4f} | Train Acc (Seen): {train_acc:.2f}% | Val Acc (Unseen): {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(agent.state_dict(), f"{config.CACHE_DIR}/pre_mdl_{config.RUN_NAME}.pt")
                logger.info(f"New Best Model Saved! (Val Acc: {best_val_acc:.2f}%)")
            
    logger.info("Pretraining Finished.")

if __name__ == "__main__":

    embedding_model = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(config.LLM_MODEL_NAME)
    corpus_data, corpus_embeddings = dataloader.get_corpus()

    agent = PolicyNetwork(
        embedding_dim=corpus_embeddings.shape[1],
        hidden_dim=config.AGENT_HIDDEN_DIM,
        dropout=config.AGENT_DROPOUT
    ).to(device)
    mdl_path = "cache/lambda_icl_qwen_0.6b/pre_mdl_1204_1720.pt"
    agent.load_state_dict(torch.load(mdl_path, map_location=device))
    agent.eval()
    
    val_loader = dataloader.get_dataloader(split='dev', batch_size=config.BATCH_SIZE, shuffle=False)

    sampler = EpisodeSampler(policy_network=agent, embedding_model=embedding_model, num_examples=config.NUM_EXAMPLES)
    from pretrain import run_evaluation
    val_acc = run_evaluation(
        llm_wrapper, 
        val_loader,
        corpus_data,
        corpus_embeddings,
        dataloader.check_correct,
        config.SYSTEM_PROMPT,
        mode = "policy",
        sampler=sampler
    )
    print(f"Val accuracy: {val_acc:.2f}%")