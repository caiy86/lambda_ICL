import torch
import torch.optim as optim
import torch.nn.functional as F
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

os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)
logger = logging.getLogger(__name__)

@torch.no_grad()
def run_metric_evaluation(llm_wrapper: LLMWrapper,query_data: DataLoader,  corpus_data: List[Dict],corpus_embeddings: torch.Tensor,check_correct_fn: callable,sampler: EpisodeSampler,desc: str) -> float:

    sampler.policy_network.eval()
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(query_data, desc=desc, leave=False):
        bsz = len(batch)
        total_samples += bsz

        buffer = sampler.collect_episodes(
            query_batch=[item for item in batch],
            corpus=corpus_data,
            corpus_embeddings=corpus_embeddings
        )
        
        prompts = []
        targets = []
        for i in range(bsz):
            prompt = llm_wrapper.build_chat_prompt(
                system_prompt=config.SYSTEM_PROMPT,
                examples=buffer.selected_examples_text[i],
                query=batch[i]['query']
            )
            prompts.append(prompt)
            targets.append(batch[i]['answer'])

        generated_texts, _ = llm_wrapper.generate_for_evaluation(
            prompts, max_new_tokens=config.MAX_GEN_TOKENS
        )
        
        for tgt, pred in zip(targets, generated_texts):
            if check_correct_fn(tgt, pred):
                total_correct += 1
                
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    return accuracy

def generate_and_save_cache(
    fpath, 
    raw_list, 
    corpus_data, 
    corpus_embeddings, 
    embedding_model, 
    llm_wrapper, 
    check_fn
):
    
    logger.info(f"Starting Oracle Generation for {len(raw_list)} samples...")
    lambda_candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    candidate_indices = [int(round(x / 0.05)) for x in lambda_candidates]
    
    all_query_embs = []
    all_target_dists = []
    all_value_targets = []
    all_actor_masks = []
    
    batch_size = config.BATCH_SIZE
    batches = [raw_list[i:i + batch_size] for i in range(0, len(raw_list), batch_size)]
    
    solvable_count = 0
    
    for batch in tqdm(batches, desc="Oracle Gen"):
        curr_bs = len(batch)
        query_texts = [x['query'] for x in batch]
        query_embs = embedding_model.encode(query_texts)

        query_indices = [x.get('corpus_index', -1) for x in batch]
        self_mask_indices = torch.tensor(query_indices, device=utils.device)

        correct_mat = torch.zeros((curr_bs, len(lambda_candidates)), dtype=torch.bool, device=utils.device)
        sim_scores_base = torch.matmul(query_embs, corpus_embeddings.T)
        
        for i_lam, lam_val in enumerate(lambda_candidates):
            lam_tensor = torch.full((curr_bs, 1), lam_val, device=utils.device)
         
            sel_mask = torch.zeros_like(sim_scores_base, dtype=torch.bool)
            if (self_mask_indices >= 0).any():
                rows = torch.arange(curr_bs, device=utils.device)
                valid = self_mask_indices >= 0
                sel_mask[rows[valid], self_mask_indices[valid]] = True
                
            sel_embs = torch.zeros((curr_bs, config.NUM_EXAMPLES, embedding_model.dim), device=utils.device)
            sel_idxs = torch.zeros((curr_bs, config.NUM_EXAMPLES), dtype=torch.long, device=utils.device)
            
            for t in range(config.NUM_EXAMPLES):
                if t == 0:
                    step_scr = sim_scores_base.clone()
                else:
                    div_pen, _ = torch.max(torch.matmul(sel_embs[:, :t, :], corpus_embeddings.T), dim=1)
                    step_scr = (lam_tensor * sim_scores_base) - ((1 - lam_tensor) * div_pen)
                
                step_scr.masked_fill_(sel_mask, -1e9)
                act = torch.argmax(step_scr, dim=1)
                sel_idxs[:, t] = act
                sel_embs[:, t, :] = corpus_embeddings[act]
                sel_mask.scatter_(1, act.unsqueeze(1), True)
            
            prompts = []
            targets = []
            sel_idxs_list = sel_idxs.cpu().tolist()
            for b_idx in range(curr_bs):
                exs = [corpus_data[idx] for idx in sel_idxs_list[b_idx]]
                p = llm_wrapper.build_chat_prompt(config.SYSTEM_PROMPT, exs, batch[b_idx]['query'])
                prompts.append(p)
                targets.append(batch[b_idx]['answer'])
            
            # LLM Gen
            preds, _ = llm_wrapper.generate_for_evaluation(prompts, max_new_tokens=config.MAX_GEN_TOKENS)
            
            for b_idx in range(curr_bs):
                if check_fn(targets[b_idx], preds[b_idx]):
                    correct_mat[b_idx, i_lam] = True

        batch_dists = torch.zeros((curr_bs, 21), device=utils.device) 
        batch_vals = torch.full((curr_bs,), -1.0, device=utils.device)
        batch_masks = torch.zeros((curr_bs,), device=utils.device)
        
        for b_idx in range(curr_bs):
            valid = correct_mat[b_idx]
            if valid.any():
                solvable_count += 1
                batch_vals[b_idx] = 1.0
                batch_masks[b_idx] = 1.0
                valid_act_indices = torch.tensor(candidate_indices, device=utils.device)[valid]
                batch_dists[b_idx].index_fill_(0, valid_act_indices, 1.0)
        
        all_query_embs.append(query_embs.cpu())
        all_target_dists.append(batch_dists.cpu())
        all_value_targets.append(batch_vals.cpu())
        all_actor_masks.append(batch_masks.cpu())

    logger.info(f"Oracle Solvability: {solvable_count}/{len(raw_list)} ({solvable_count/len(raw_list):.2%})")
    
    ds = TensorDataset(
        torch.cat(all_query_embs),
        torch.cat(all_target_dists),
        torch.cat(all_value_targets),
        torch.cat(all_actor_masks)
    )
    torch.save(ds, fpath)
    return ds

def main():
    utils.setup_logging(log_level="INFO", log_file=os.path.join(config.LOG_DIR, f"pretrain_{config.RUN_NAME}.log"))
    logger.info(f"Run Name: {config.RUN_NAME}")
    
    embedding_model = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(config.LLM_MODEL_NAME)
    agent = PolicyNetwork(embedding_model.dim, config.AGENT_HIDDEN_DIM, config.AGENT_DROPOUT).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=config.PRETRAIN_LR)
    
    corpus_data, corpus_embeddings_cpu = dataloader.get_corpus()
    corpus_embeddings = corpus_embeddings_cpu.to(device)
    
    val_size = 64 
    train_size = config.PRETRAIN_NUMS 
    
    train_raw, val_raw = dataloader.get_train_val_split_data(
        split='train',
        train_nums=train_size,
        val_nums=val_size,
        seed=config.PRETRAIN_SEED
    )
    
    cache_file = f"{config.CACHE_DIR}/train_oracle_{train_size}_{config.PRETRAIN_SEED}.pt"
    if os.path.exists(cache_file):
        logger.info(f"Loading Oracle Cache: {cache_file}")
        train_ds = torch.load(cache_file, weights_only=False)
    else:
        logger.info("Cache miss. Generating Oracle...")
        train_ds = generate_and_save_cache(
            cache_file, train_raw, corpus_data, corpus_embeddings, 
            embedding_model, llm_wrapper, dataloader.check_correct
        )
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_seen_ds = DataLoader(
        dataloader.MtopQueryDataset(train_raw[:config.BATCH_SIZE * 4]),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=dataloader.list_dict_collate_fn
    )
    val_ds = DataLoader(
        dataloader.MtopQueryDataset(val_raw),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=dataloader.list_dict_collate_fn
    )
    test_ds = dataloader.get_dataloader(split='dev', batch_size=config.BATCH_SIZE, shuffle=False)


    sampler = EpisodeSampler(agent, embedding_model, config.NUM_EXAMPLES)

    logger.info("--- Starting Training ---")
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    # mse_loss = torch.nn.MSELoss()


    for epoch in range(1, config.PRETRAIN_MAX_EPOCHS + 1):
        agent.train()
        total_loss = 0.0
        
        for q_emb, t_dist, v_tgt, mask in train_loader:
            q_emb, t_dist, v_tgt, mask = q_emb.to(device), t_dist.to(device), v_tgt.to(device), mask.to(device)
            
            features = agent.feature_net(q_emb)
            logits = agent.actor_head(features)
            # values = agent.value_head(features).squeeze(-1)
            
            act_loss = (bce_loss(logits, t_dist).mean(dim=1) * mask).sum() / (mask.sum() + 1e-8)
            
            # val_loss = mse_loss(values, v_tgt)
            
            loss = act_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
        if avg_loss < config.PRETRAIN_LOSS_THRESHOLD:
            logger.info(f"Training stopped at epoch {epoch} as loss {avg_loss:.4f} < threshold {config.PRETRAIN_LOSS_THRESHOLD:.4f}")
            break
        
        if epoch % 20 == 0:
            logger.info(f"--- Eval @ Epoch {epoch} ---")
            
            train_acc = run_metric_evaluation(
                llm_wrapper, val_seen_ds, corpus_data, corpus_embeddings,
                dataloader.check_correct, sampler, "Eval Train Sub"
            )

            val_acc = run_metric_evaluation(
                llm_wrapper, val_ds, corpus_data, corpus_embeddings,
                dataloader.check_correct, sampler, "Eval Val"
            )
            
            logger.info(f"Epoch {epoch} Result | Train(sub) Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            # 保存val acc最好的模型
            if not hasattr(main, 'best_val_acc'):
                main.best_val_acc = 0.0
                main.best_epoch = 0
            if val_acc > main.best_val_acc:
                main.best_val_acc = val_acc
                main.best_epoch = epoch
                logger.info(f"New best Val Acc: {val_acc:.2f}% at epoch {epoch}, saving checkpoint.")
                torch.save(agent.state_dict(), f"{config.CACHE_DIR}/pre_mdl_{config.RUN_NAME}_val_best.pt")

    logger.info("--- Final Test on Dev Set ---")
    test_acc = run_metric_evaluation(
        llm_wrapper, test_ds, corpus_data, corpus_embeddings,
        dataloader.check_correct, sampler, "Eval Dev"
    )
    logger.info(f"Final Dev Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()