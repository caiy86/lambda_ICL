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

from config import pretrain_config as config
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
                   sampler: Optional[EpisodeSampler] = None, 
                   embedding_model: Optional[EmbeddingModel] = None, 
                   num_examples: int = 4) -> float:
    
    logger.info(f"Starting evaluation (Mode: {mode})...")
    
    if mode == 'policy' and sampler is None:
        raise ValueError("Sampler must be provided for 'policy' mode evaluation.")
    if mode == 'policy':
        sampler.policy_network.eval()
    
    # 如果是 MMR Baseline 模式 (非 policy)，我们需要一个临时的 sampler 或者手动实现
    # 为了简化，这里仅支持 'policy' 模式的评估 (用于验证预训练效果)
    # 如果需要对比 baseline，建议使用 main.py
    if mode != 'policy':
        logger.warning("Pretrain evaluation currently only supports 'policy' mode fully. Treating as Policy evaluation if sampler provided.")

    total_correct = 0
    total_nll = 0.0 
    total_samples = 0
    
    for query_batch_list in tqdm(val_loader, desc=f"Validating pretrain ({mode})"):
        batch_size = len(query_batch_list)
        
        # 使用 Sampler 收集数据 (自适应 Lambda)
        buffer = sampler.collect_episodes(
            query_batch=query_batch_list,
            corpus=corpus_data,
            corpus_embeddings=corpus_embeddings
        )
        
        prompts = []
        targets = []
        for i in range(batch_size):
            query_data = buffer.queries[i]
            # 获取 Sampler 选出的样本
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
            prompts, 
            max_new_tokens=config.MAX_GEN_TOKENS
        )
        
        if isinstance(generated_nlls, torch.Tensor):
            generated_nlls_list = generated_nlls.cpu().tolist()
        else:
            generated_nlls_list = generated_nlls

        for i in range(batch_size):
            pred_text = generated_texts[i]
            target_text = targets[i]
            if check_correct_fn(target_answer=target_text, pred_text=pred_text):
                total_correct += 1
            total_nll += generated_nlls_list[i] 
        
        total_samples += batch_size

    accuracy = 0.0
    avg_nll = float('inf')
    if total_samples > 0:
        accuracy = (total_correct / total_samples) * 100
        avg_nll = total_nll / total_samples
    
    logger.info(
        f"Evaluation Pretrain (Mode: {mode}) Finished. "
        f"Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples}), "
        f"Avg NLL: {avg_nll:.4f}"
    )
    return accuracy

@torch.no_grad()
def generate_best_lambda_dataset(
    query_loader: DataLoader,
    corpus_data: List[Dict[str, str]],
    corpus_embeddings: torch.Tensor,
    embedding_model: EmbeddingModel,
    llm_wrapper: LLMWrapper,
    num_candidates: int = 10  # 每个 Query 尝试多少个随机 Lambda
) -> TensorDataset:
    """
    核心逻辑：对于每个 Query，随机尝试 num_candidates 个 Lambda，
    计算它们对应的 MMR 检索结果在 LLM 上的 Loss，
    选取 Loss 最小的 Lambda 作为该 Query 的监督学习标签。
    """
    logger.info(f"--- Generating Optimal Lambda Dataset (Best-of-{num_candidates} Search) ---")

    all_query_embs = []
    all_best_actions = [] # 存储最佳 Lambda 的索引 (0-20)
    
    for query_batch_list in tqdm(query_loader, desc="Searching optimal lambdas"):
        real_batch_size = len(query_batch_list)
        
        query_texts = [item['query'] for item in query_batch_list]
        query_embs = embedding_model.encode(query_texts) # (B, D)
        
        # 1. 扩展 Query: (B, D) -> (B * K, D)
        # 这样我们可以一次性处理所有候选 Lambda
        query_embs_expanded = query_embs.repeat_interleave(num_candidates, dim=0)
        
        # 2. 随机生成 Lambda 索引 (0-20)
        # (B * K,)
        rand_actions = torch.randint(0, 21, (real_batch_size * num_candidates,), device=utils.device)
        
        # 转换为实际 Lambda 值 (0.00 - 1.00)
        lambda_vals = (rand_actions.float() * 0.05).unsqueeze(1) # (B*K, 1)
        
        # 3. 执行并行 MMR
        eff_batch_size = real_batch_size * num_candidates
        
        # 临时存储检索结果
        batch_selected_indices = torch.zeros((eff_batch_size, config.NUM_EXAMPLES), dtype=torch.long, device=utils.device)
        # 为了节省显存，这里我们不存储所有 step 的 embedding，只存 indices 和 当前 step 的 embedding
        batch_selected_embs_current_step = torch.zeros((eff_batch_size, config.NUM_EXAMPLES, embedding_model.dim), device=utils.device)
        
        sim_scores = torch.matmul(query_embs_expanded, corpus_embeddings.T) # (B*K, CorpusSize)
        relevance_scores = sim_scores
        selected_mask = torch.zeros_like(sim_scores, dtype=torch.bool)
        
        # 处理 Self-selection mask (如果是训练集，需要屏蔽自己)
        # 这里为了简化，假设 corpus 是全集，如果 query 在 corpus 里，最好屏蔽掉。
        # 考虑到效率，这里暂时略过严格的 self-masking，或者假设 pretrain query 不直接等于 corpus index
        
        for t in range(config.NUM_EXAMPLES):
            if t == 0:
                step_scores = relevance_scores
            else:
                # 计算多样性惩罚
                # 优化内存：使用 matmul 而不是 expand
                # selected_embs_so_far: (B*K, t, D)
                selected_embs_so_far = batch_selected_embs_current_step[:, :t, :]
                
                # (B*K, t, D) @ (D, C) -> (B*K, t, C)
                sim_to_selected = torch.matmul(selected_embs_so_far, corpus_embeddings.T)
                diversity_penalty, _ = torch.max(sim_to_selected, dim=1) # (B*K, C)
                
                step_scores = (lambda_vals * relevance_scores) - \
                              ((1 - lambda_vals) * diversity_penalty)

            step_scores.masked_fill_(selected_mask, -torch.inf)
            current_action = torch.argmax(step_scores, dim=1) # (B*K,)
            
            batch_selected_indices[:, t] = current_action
            batch_selected_embs_current_step[:, t, :] = corpus_embeddings[current_action]
            selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)
            
        # 4. 构建 Prompt 并计算 Loss
        prompts = []
        targets = []
        
        for i in range(eff_batch_size):
            # 找到对应的原始 query
            orig_idx = i // num_candidates
            query_data = query_batch_list[orig_idx]
            
            selected_indices = batch_selected_indices[i].cpu().tolist()
            example_data = [corpus_data[idx] for idx in selected_indices]
            
            prompt_str = llm_wrapper.build_chat_prompt(
                system_prompt=config.SYSTEM_PROMPT,
                examples=example_data,
                query=query_data['query'],
                strategy=config.PROMPT_STRATEGY
            )
            prompts.append(prompt_str)
            targets.append(query_data['answer'])
            
        # 获取 Loss (B*K,)
        # 注意：这里我们使用 no_grad，因为我们只是为了产生数据，不训练 LLM
        per_sample_loss = llm_wrapper.get_batch_loss(prompts, targets)
        
        # 5. 寻找每个 Query 的最佳 Lambda
        # Reshape: (B, K)
        per_sample_loss = per_sample_loss.view(real_batch_size, num_candidates)
        rand_actions = rand_actions.view(real_batch_size, num_candidates)
        
        # 找到最小 Loss 的索引 (0..K-1)
        min_loss_values, min_loss_indices = torch.min(per_sample_loss, dim=1) # (B,)
        
        # 提取对应的最佳 Action (0..20)
        # gather dim=1
        best_actions_batch = rand_actions.gather(1, min_loss_indices.unsqueeze(1)).squeeze(1) # (B,)
        
        # 收集数据
        all_query_embs.append(query_embs.cpu())
        all_best_actions.append(best_actions_batch.cpu())
        
    all_query_embs = torch.cat(all_query_embs, dim=0)
    all_best_actions = torch.cat(all_best_actions, dim=0)
    
    logger.info(f"Generated Dataset: {len(all_query_embs)} samples.")
    logger.info(f"Best Action Distribution: \n{pd.Series(all_best_actions.numpy()).value_counts().sort_index()}")
    
    return TensorDataset(all_query_embs, all_best_actions)

def pretrain_agent(agent: PolicyNetwork,
                   optimizer: optim.Optimizer,
                   pretrain_dataset: TensorDataset,
                   device: torch.device):
    
    logger.info(f"--- Starting Supervised Pre-training (Classification) ---")
    logger.info(f"Target Loss: {config.PRETRAIN_LOSS_THRESHOLD}")
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    # 既然 Action 是离散的类别 (0-20)，我们使用 CrossEntropyLoss
    crit_loss_fn = torch.nn.CrossEntropyLoss()
    
    epoch = 0
    
    while epoch < config.PRETRAIN_MAX_EPOCHS:
        epoch += 1
        agent.train() 
        
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        pbar = tqdm(pretrain_loader, desc=f"Pre-train Epoch {epoch}/{config.PRETRAIN_MAX_EPOCHS}")
        
        for batch in pbar:
            query_embs, target_actions = batch

            query_embs = query_embs.to(device)
            target_actions = target_actions.to(device) # (B,)

            # 前向传播，不需要传入 actions，我们想要 logits
            # forward 返回: actions, log_probs, values, entropy
            # 我们需要修改 forward 或者使用内部的 actor_head
            # 但 standard forward 计算了 dist，我们可以直接获取 logits 吗？
            # 让我们查看 PolicyNetwork 代码，forward 中 logits 是局部变量。
            # 为了方便，我们可以直接调用 agent.feature_net 和 agent.actor_head
            
            features = agent.feature_net(query_embs)
            logits = agent.actor_head(features) # (B, 21)
            
            loss = crit_loss_fn(logits, target_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率仅供参考
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == target_actions).sum().item()
            total_preds += target_actions.size(0)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct_preds/total_preds:.2%}"
            })
            
        avg_loss = total_loss / len(pretrain_loader)
        logger.info(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, Acc: {correct_preds/total_preds:.2%}")
        
        if avg_loss < config.PRETRAIN_LOSS_THRESHOLD:
            logger.info("Loss threshold reached. Stopping pre-training.")
            break

corpus_embeddings = None

def main():
    global corpus_embeddings 
    
    os.makedirs(config.LOG_DIR, exist_ok=True)
    utils.setup_logging(log_level=config.LOG_LEVEL, log_file=config.LOG_FILE)
    utils.initialize_seeds(config.SEED)
    device = utils.device
    logger.info(f"Using device: {device}")

    logger.info("--- Loading Models & Data for Pre-training ---")
    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(
        model_name=config.LLM_MODEL_NAME, 
    )
    
    # 使用新的 MLP 结构初始化 Policy
    agent = PolicyNetwork(
        embedding_dim=embedding_model.dim,
        hidden_dim=config.AGENT_HIDDEN_DIM,
        dropout=config.AGENT_DROPOUT
    ).to(device)
    
    corpus_data, corpus_embeddings_cpu = dataloader.get_corpus() 
    corpus_embeddings = corpus_embeddings_cpu.to(device)

    # 1. 准备数据加载器
    # 我们可以只用一部分数据来做这个 Search，或者全部
    pretrain_query_loader = dataloader.get_dataloader(
        split='train', 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, # Shuffle to get random mix
        nums=config.TRAIN_NUMS # 可以限制数量加快速度
    )
    
    # 2. 生成最佳 Lambda 数据集 (The "Loss-driven" Dataset)
    # 每个 Query 尝试 5-10 个随机 Lambda，取最好的
    best_lambda_dataset = generate_best_lambda_dataset(
        query_loader=pretrain_query_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        llm_wrapper=llm_wrapper,
        num_candidates=10  # 尝试 10 个随机值
    )

    # 3. 开始监督预训练
    optimizer = optim.AdamW(agent.parameters(), lr=config.PRETRAIN_LR)

    pretrain_agent(
        agent=agent,
        optimizer=optimizer,
        pretrain_dataset=best_lambda_dataset,
        device=device 
    )

    # 4. 评估预训练后的模型
    logger.info("--- Initializing Sampler for Evaluation ---")
    sampler = EpisodeSampler(
        policy_network=agent,
        embedding_model=embedding_model,
        num_examples=config.NUM_EXAMPLES
    )
    
    val_loader = dataloader.get_dataloader(
        split='dev',
        batch_size=config.BATCH_SIZE_VAL,
        shuffle=False 
    )

    logger.info("--- Running Evaluation on Pre-trained Policy ---")
    run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,      
        corpus_data=corpus_data,       
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        prompt_strategy=config.PROMPT_STRATEGY,
        mode='policy', 
        sampler=sampler, 
        embedding_model=embedding_model 
    )

    os.makedirs("checkpoints", exist_ok=True)
    save_path = config.PRETRAINED_MODEL_PATH
    torch.save(agent.state_dict(), save_path)
    logger.info(f"--- Pre-training Finished ---")
    logger.info(f"Pre-trained model saved to: {save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("An unhandled exception occurred!", exc_info=True)
        raise e