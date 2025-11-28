import torch
import torch.optim as optim
import os
import logging
from tqdm import tqdm
from typing import List, Dict,Optional
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import train_config as config 
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.sampler import EpisodeSampler, RolloutBuffer 
from engine.reward_computer import RewardComputer
from trainer.ppo_trainer import PPOTrainer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def train():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    utils.setup_logging(log_level="INFO", log_file=os.path.join(config.LOG_DIR, f"{config.RUN_NAME}.log"))
    # utils.initialize_seeds(config.SEED)
    device = utils.device
    logger.info(f"Using device: {device}")
    logger.info(f"Starting run: {config.RUN_NAME}")

    logger.info("--- Initializing Models ---")
    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(
        model_name=config.LLM_MODEL_NAME, 
    )
    agent = PolicyNetwork(
        embedding_dim=embedding_model.dim,
        hidden_dim=config.AGENT_HIDDEN_DIM,
        dropout=config.AGENT_DROPOUT 
    ).to(device)

    logger.info("--- Loading Data ---")
    corpus_data, corpus_embeddings = dataloader.get_corpus()
    logger.info(f"Corpus embeddings computed. Shape: {corpus_embeddings.shape}")
    
    train_loader = dataloader.get_dataloader(
        split='train', 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        nums = config.TRAIN_NUMS
    )
    val_loader = dataloader.get_dataloader(
        split='dev',
        batch_size=config.BATCH_SIZE_VAL,
        shuffle=False 
    )
    
    logger.info("--- Initializing PPO Components ---")
    sampler = EpisodeSampler(
        policy_network=agent,
        embedding_model=embedding_model,
        num_examples=config.NUM_EXAMPLES
    )
    reward_computer = RewardComputer(
        gamma=config.REWARD_GAMMA,
        lambda_=config.REWARD_LAMBDA,
        system_prompt=config.SYSTEM_PROMPT,
    )
    optimizer = optim.AdamW(agent.parameters(), lr=config.LR)
    ppo_trainer = PPOTrainer(
        agent=agent,
        optimizer=optimizer,
        ppo_epochs=config.PPO_EPOCHS,
        ppo_clip_eps=config.PPO_CLIP_EPS,
        value_loss_coef=config.V_LOSS_COEF,
        entropy_bonus_coef=config.E_BONUS_COEF,
        grad_clip_norm=config.GRAD_CLIP_NORM
    )

    logger.info("--- Starting Training Loop ---")

    logger.info("--- Running Initial kNN Baseline Evaluation ---")

    run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        epoch=0, 
        mode='mmr',           
        sampler=None,
        embedding_model=embedding_model,
        num_examples=config.NUM_EXAMPLES,
        mmr_baseline_lambda=0.7
    )
    
    logger.info("--- kNN Baseline Evaluation Finished ---")
    
    total_batches = 0
    best_val_accuracy = -1.0 
    
    for epoch in range(config.TOTAL_TRAIN_EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{config.TOTAL_TRAIN_EPOCHS}")

        agent.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]")
        
        for query_batch_list in pbar:
            if not isinstance(query_batch_list, list):
                 batch_size = len(query_batch_list['query'])
                 query_batch_list = [
                     {"query": query_batch_list['query'][i], "answer": query_batch_list['answer'][i]}
                     for i in range(batch_size)
                 ]

            buffer = sampler.collect_episodes(
                query_batch=query_batch_list,
                corpus=corpus_data,
                corpus_embeddings=corpus_embeddings
            )

            buffer = reward_computer.compute_rewards_and_advantages(
                buffer=buffer,
                llm_wrapper=llm_wrapper
            )

            log_dict = ppo_trainer.train_step(
                buffer=buffer
            )
            
            total_batches += 1
            avg_llm_loss = buffer.final_llm_loss.mean().item()
            
            pbar.set_postfix({
                "actor_loss": f"{log_dict['actor_loss']:.4f}",
                "value_loss": f"{log_dict['value_loss']:.4f}",
                "avg_reward": f"{buffer.rewards.mean().item():.3f}",
                "llm_loss": f"{avg_llm_loss:.3f}"
            })

        if (epoch + 1) % 10 == 0 or epoch == 0:
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

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                logger.info(f"New best validation accuracy: {best_val_accuracy:.2f}%! Saving model...")

    logger.info("--- Final evaluation after all epochs ---")
    val_accuracy = run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        epoch = config.TOTAL_TRAIN_EPOCHS ,
        mode='policy',
        sampler=sampler 
    )
    logger.info(f"Final post-training evaluation accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        logger.info(f"New best validation accuracy: {best_val_accuracy:.2f}%! Saving model...")
        torch.save(agent.state_dict(), f"checkpoints/{config.RUN_NAME}_best.pt")

    logger.info(f"--- Training Finished ---")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error("An unhandled exception occurred!", exc_info=True)
        raise e