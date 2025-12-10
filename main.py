import torch
import torch.optim as optim
import os
import logging
from tqdm import tqdm
from typing import List, Dict,Optional
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.optim.lr_scheduler import LinearLR
import wandb

from config import train_config as config 
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.sampler import EpisodeSampler, RolloutBuffer 
from engine.reward_computer import RewardComputer
from trainer.ppo_trainer import PPOTrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

@torch.no_grad()
def run_evaluation(
    llm_wrapper: LLMWrapper,
    val_loader: torch.utils.data.DataLoader,
    corpus_data: List[Dict[str, str]],
    corpus_embeddings: torch.Tensor,
    check_correct_fn: callable,
    system_prompt: str,
    epoch: int, 
    mode: str,
    sampler: Optional[EpisodeSampler] = None, 
    embedding_model: Optional[EmbeddingModel] = None, 
    num_examples: int = 4,
    mmr_baseline_lambda: float = 0.7
) -> float:
    
    logger.info(f"Starting evaluation (Mode: {mode.upper()}) for Epoch {epoch}...")
    if mode =="policy":
        sampler.policy_network.eval() 
    
    total_correct = 0
    total_nll = 0.0 
    total_samples = 0
    
    all_prompts: List[str] = []
    all_generated_texts: List[str] = []
    all_generated_nlls: List[float] = []
    all_targets: List[str] = []

    for query_batch_list in tqdm(val_loader, desc=f"Validating Epoch {epoch} ({mode.upper()})"):
        batch_size = len(query_batch_list)
        
        if mode == 'policy':

            buffer = sampler.collect_episodes(
                query_batch=query_batch_list,
                corpus=corpus_data,
                corpus_embeddings=corpus_embeddings
            )
        else:

            buffer = RolloutBuffer(batch_size=batch_size, device=utils.device)
            buffer.queries = query_batch_list
            
            query_texts = [item['query'] for item in query_batch_list]
            query_embs = embedding_model.encode(query_texts)

            batch_selected_embs = torch.zeros((batch_size, num_examples, embedding_model.dim), device=utils.device)
            batch_selected_indices = torch.zeros((batch_size, num_examples), dtype=torch.long, device=utils.device)

            sim_scores = torch.matmul(query_embs, corpus_embeddings.T) # (B, Corpus)
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

                    step_scores = (mmr_baseline_lambda * relevance_scores) - \
                                  ((1 - mmr_baseline_lambda) * diversity_penalty)

                step_scores.masked_fill_(selected_mask, -torch.inf)

                current_action = torch.argmax(step_scores, dim=1)

                current_embs = corpus_embeddings[current_action]
                batch_selected_indices[:, t] = current_action
                batch_selected_embs[:, t, :] = current_embs
                selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)

            for i in range(batch_size):
                indices = batch_selected_indices[i].cpu().tolist()
                buffer.selected_examples_text[i] = [corpus_data[idx] for idx in indices]

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
        
        all_prompts.extend(prompts)
        all_generated_texts.extend(generated_texts)
        all_generated_nlls.extend(generated_nlls_list)
        all_targets.extend(targets)

    accuracy = 0.0
    avg_nll = float('inf')
    
    if total_samples > 0:
        accuracy = (total_correct / total_samples) * 100
        avg_nll = total_nll / total_samples
    
    logger.info(
        f"Evaluation Epoch {epoch} (Mode: {mode.upper()}) Finished. "
        f"Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples}), "
        f"Avg NLL: {avg_nll:.4f}"
    )

    logger.info("Saving evaluation results to file...")
    
    all_correct = [
        check_correct_fn(target, pred) 
        for target, pred in zip(all_targets, all_generated_texts)
    ]
    
    df = pd.DataFrame({
        "prompt": all_prompts,
        "target": all_targets,
        "prediction": all_generated_texts,
        "nll": all_generated_nlls,
        "is_correct": all_correct
    })

    os.makedirs(f"results/lambda_icl_qwen3_0.6b/{config.RUN_NAME}", exist_ok=True) 
    output_filename = f"results/lambda_icl_qwen3_0.6b/{config.RUN_NAME}/val_{epoch}.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8')
    logger.info(f"Evaluation results saved to: {output_filename}")

    for i in range(min(3, len(all_targets))):
        if not all_correct[i]:
            logger.debug(f"--- Bad Case Example {i+1} ---")
            logger.debug(f"PROMPT:\n{all_prompts[i]}")
            logger.debug(f"TARGET: {all_targets[i]}")
            logger.debug(f"PRED:   {all_generated_texts[i]}")
            logger.debug(f"NLL:    {all_generated_nlls[i]:.4f}")
            logger.debug("--------------------------")
            
    return accuracy

def merge_buffers(buffers: List[RolloutBuffer]) -> RolloutBuffer:
    total_batch_size = sum(b.batch_size for b in buffers)
    device = buffers[0].device

    merged = RolloutBuffer(batch_size=total_batch_size, device=device)
    
    merged.query_embeddings = torch.cat([b.query_embeddings for b in buffers], dim=0)
    merged.actions = torch.cat([b.actions for b in buffers], dim=0)
    merged.log_probs = torch.cat([b.log_probs for b in buffers], dim=0)
    merged.values = torch.cat([b.values for b in buffers], dim=0)
    merged.rewards = torch.cat([b.rewards for b in buffers], dim=0)
    merged.advantages = torch.cat([b.advantages for b in buffers], dim=0)
    merged.returns = torch.cat([b.returns for b in buffers], dim=0)
    merged.final_llm_loss = torch.cat([b.final_llm_loss for b in buffers], dim=0)
    
    merged.queries = []
    merged.selected_examples_text = []
    for b in buffers:
        merged.queries.extend(b.queries)
        merged.selected_examples_text.extend(b.selected_examples_text)
        
    return merged

def train():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    utils.setup_logging(log_level="INFO", log_file=os.path.join(config.LOG_DIR, f"{config.RUN_NAME}.log"))
    device = utils.device
    logger.info(f"Using device: {device}")
    logger.info(f"Starting run: {config.RUN_NAME}")

    # --- WandB Init ---
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=config.RUN_NAME,
            config={k: getattr(config, k) for k in dir(config) if not k.startswith("__") and not callable(getattr(config, k))}
        )

    # --- Model Init ---
    logger.info("--- Initializing Models ---")
    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(model_name=config.LLM_MODEL_NAME)
    from models.policy_network import RBFPolicyNetwork
    agent = RBFPolicyNetwork(
        embedding_dim=embedding_model.dim,
        num_centers=1024,
        dropout=config.AGENT_DROPOUT
    ).to(device)

    optimizer = optim.AdamW(agent.parameters(), lr=config.LR)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config.TOTAL_TRAIN_EPOCHS)

    if config.PRETRAINED_PATH and os.path.exists(config.PRETRAINED_PATH):
        logger.info(f"Loading pretrained model from: {config.PRETRAINED_PATH}")
        state_dict = torch.load(config.PRETRAINED_PATH, map_location=device)
        agent.load_state_dict(state_dict)
    else:
        logger.warning(f"No pretrained model found at {config.PRETRAINED_PATH}, starting from scratch!")

    logger.info("--- Loading Data ---")
    corpus_data, corpus_embeddings = dataloader.get_corpus()
    
    train_loader = dataloader.get_dataloader(split='train', batch_size=config.BATCH_SIZE, shuffle=True, nums=config.TRAIN_NUMS)
    val_loader = dataloader.get_dataloader(split='dev', batch_size=config.BATCH_SIZE, shuffle=False)
    
    sampler = EpisodeSampler(policy_network=agent, embedding_model=embedding_model, num_examples=config.NUM_EXAMPLES)
    reward_computer = RewardComputer(gamma=config.REWARD_GAMMA, lambda_=config.REWARD_LAMBDA, system_prompt=config.SYSTEM_PROMPT)
    ppo_trainer = PPOTrainer(
        agent=agent, 
        optimizer=optimizer, 
        ppo_epochs=config.PPO_EPOCHS,
        ppo_clip_eps=config.PPO_CLIP_EPS, 
        value_loss_coef=config.V_LOSS_COEF,
        entropy_bonus_coef=config.E_BONUS_COEF, 
        grad_clip_norm=config.GRAD_CLIP_NORM,
        mini_batch_size=config.PPO_MINIBATCH_SIZE
    )

    # --- Pretraining Evaluation with MMR Baseline ---
    logger.info("--- Running Pre-Training Evaluation (MMR Baseline) ---")
    mmr_val_accuracy = run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        epoch=0,
        mode='mmr',
        embedding_model=embedding_model,
        num_examples=config.NUM_EXAMPLES,
        mmr_baseline_lambda=getattr(config, "MMR_LAMBDA", 0.7),
    )   
    if config.USE_WANDB:
        wandb.log({"val/mrr_baseline_accuracy": mmr_val_accuracy, "epoch": 0})
    logger.info(f"MMR Baseline Validation Accuracy: {mmr_val_accuracy:.2f}%")

    logger.info("--- Starting Training Loop ---")
    
    total_batches = 0
    best_val_accuracy = -1.0 
    
    current_buffers = []    
    current_steps_count = 0   

    for epoch in range(config.TOTAL_TRAIN_EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{config.TOTAL_TRAIN_EPOCHS}")
        agent.train()

        epoch_stats = {
            "actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "reward": 0.0, "llm_loss": 0.0, "count": 0,
            "correct_samples": 0, "total_samples": 0 
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]")
        
        for query_batch_list in pbar:
            if not isinstance(query_batch_list, list):
                 batch_size = len(query_batch_list['query'])
                 query_batch_list = [{"query": query_batch_list['query'][i], "answer": query_batch_list['answer'][i]} for i in range(batch_size)]

            buffer = sampler.collect_episodes(
                query_batch=query_batch_list,
                corpus=corpus_data,
                corpus_embeddings=corpus_embeddings
            )

            buffer = reward_computer.compute_rewards_and_advantages(
                buffer=buffer,
                llm_wrapper=llm_wrapper,
                check_correct_fn=dataloader.check_correct,
            )
            if hasattr(buffer, 'info'):
                epoch_stats["correct_samples"] += buffer.info["correct_count"]
                epoch_stats["total_samples"] += buffer.info["total_count"]   

            current_buffers.append(buffer)
            current_steps_count += len(buffer.queries) 
            
            if current_steps_count >= config.UPDATE_TIMESTEPS:
                merged_buffer = merge_buffers(current_buffers)

                log_dict = ppo_trainer.train_step(buffer=merged_buffer)
                
                avg_llm_loss = merged_buffer.final_llm_loss.mean().item()
                avg_reward = merged_buffer.rewards.mean().item()
                
                epoch_stats["actor_loss"] += log_dict.get("actor_loss", 0)
                epoch_stats["value_loss"] += log_dict.get("value_loss", 0)
                epoch_stats["entropy"] += log_dict.get("entropy_loss", 0)
                epoch_stats["approx_kl"] += log_dict.get("approx_kl", 0)
                epoch_stats["reward"] += avg_reward
                epoch_stats["llm_loss"] += avg_llm_loss
                epoch_stats["count"] += 1
                
                total_batches += 1

                if config.USE_WANDB:
                    wandb.log({
                        "train/reward": avg_reward,
                        "train/actor_loss": log_dict["actor_loss"],
                        "train/value_loss": log_dict["value_loss"],
                        "train/entropy": log_dict["entropy_loss"], # 关注 Entropy 是否骤降
                        "train/kl": log_dict["approx_kl"],         # 关注 KL 是否过大 (>0.05 说明更新太激进)
                        "train/clip_frac": log_dict["clip_frac"],  # 关注 Clip 比例 (理想 0.0~0.2)
                        "train/explained_var": log_dict["explained_var"], # 关注 Critic 质量
                        "global_step": total_batches
                    })

                current_buffers = []
                current_steps_count = 0
                
                run_avg_actor = epoch_stats["actor_loss"] / epoch_stats["count"]
                run_avg_val = epoch_stats["value_loss"] / epoch_stats["count"]
                run_avg_rew = epoch_stats["reward"] / epoch_stats["count"]
                run_avg_kl = epoch_stats["approx_kl"] / epoch_stats["count"]

                run_avg_acc = 0.0
                if epoch_stats["total_samples"] > 0:
                    run_avg_acc = (epoch_stats["correct_samples"] / epoch_stats["total_samples"]) * 100    

                pbar.set_postfix({
                    "avg_act": f"{run_avg_actor:.3f}", # 平均 Actor Loss
                    "avg_crt": f"{run_avg_val:.3f}",   # 平均 Critic Loss
                    "avg_rew": f"{run_avg_rew:.3f}",   # 平均 Reward
                    "avg_kl": f"{run_avg_kl:.4f}",      # 平均 KL
                    "avg_acc": f"{run_avg_acc:.2f}%"
                })

        if epoch_stats["count"] > 0:
            avg_stats = {k: v / epoch_stats["count"] for k, v in epoch_stats.items() if k != "count"}
            train_acc = 0.0
            if epoch_stats["total_samples"] > 0:
                train_acc = (epoch_stats["correct_samples"] / epoch_stats["total_samples"]) * 100
            logger.info(
                f"Epoch {epoch+1} Summary | "
                f"Train Acc: {train_acc:.2f}% | " 
                f"Avg Rew: {avg_stats['reward']:.4f} | "
                f"Avg KL: {avg_stats['approx_kl']:.4f} | "
                f"Avg ActLoss: {avg_stats['actor_loss']:.4f}"
            )
            if config.USE_WANDB:
                wandb.log({
                    "epoch/train_accuracy": train_acc,
                    "epoch/avg_reward": avg_stats['reward'],
                    "epoch/avg_kl": avg_stats['approx_kl'],
                    "epoch/avg_actor_loss": avg_stats['actor_loss'],
                    "epoch": epoch + 1
                })

        scheduler.step()
        
        val_accuracy = run_evaluation(
            llm_wrapper=llm_wrapper,
            val_loader=val_loader,
            corpus_data=corpus_data,
            corpus_embeddings=corpus_embeddings,
            check_correct_fn=dataloader.check_correct,
            system_prompt=config.SYSTEM_PROMPT,
            epoch=epoch + 1, 
            mode='policy',     
            sampler=sampler        
        )
        if config.USE_WANDB:
            wandb.log({"val/accuracy": val_accuracy, "epoch": epoch + 1})

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logger.info(f"New best validation accuracy: {best_val_accuracy:.2f}%! Saving model...")
            torch.save(agent.state_dict(), f"cache/{config.PROJECT_NAME}/{config.RUN_NAME}_best.pt")

    if config.USE_WANDB:
        wandb.finish()

import gc 

if __name__ == "__main__":
    # try:
    #     train()
    # except Exception as e:
    #     logger.error("An unhandled exception occurred!", exc_info=True)
    #     raise e

    experiments = [
        {"lr": 1e-4, "E_BONUS_COEF": 0,     "loss_weight": 0.1},
        {"lr": 2e-4, "E_BONUS_COEF": 0.001, "loss_weight": 0.1},
        {"lr": 2e-4, "E_BONUS_COEF": 0,     "loss_weight": 0.2},
        {"lr": 3e-4, "E_BONUS_COEF": 0,     "loss_weight": 0.05},
    ]

    for i, exp in enumerate(experiments):
        try:
            config.LR = exp["lr"]
            config.E_BONUS_COEF = exp["E_BONUS_COEF"]
            config.LOSS_WEIGHT = exp["loss_weight"]
            
            config.RUN_NAME = utils.get_run_name(config.PROJECT_NAME)
            
            print(f"\n{'='*40}")
            print(f"Experiment {i+1}/{len(experiments)}: {config.RUN_NAME}")
            print(f"Params: LR={config.LR}, E_BONUS_COEF={config.E_BONUS_COEF}, Loss_W={config.LOSS_WEIGHT}")
            print(f"{'='*40}\n")

            train()

        except Exception as e:
            print(f"Experiment {config.RUN_NAME} FAILED!")
            import traceback
            traceback.print_exc()
        
        finally:
            print(f"Experiment {config.RUN_NAME} finished. Cleaning up...")

            gc.collect()
            torch.cuda.empty_cache()
            
            print("Cleanup done. Moving to next experiment...\n")
