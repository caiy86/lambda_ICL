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

logger = logging.getLogger(__name__)

@torch.no_grad()
def run_evaluation(
    llm_wrapper: LLMWrapper,
    val_loader: torch.utils.data.DataLoader,
    corpus_data: List[Dict[str, str]],
    corpus_embeddings: torch.Tensor,
    check_correct_fn: callable,
    system_prompt: str,
    prompt_strategy: str,
    epoch: str, 
    mode: str,
    sampler: Optional[EpisodeSampler] = None, 
    embedding_model: Optional[EmbeddingModel] = None, 
    num_examples_knn: int = 4
) -> float:
    
    logger.info(f"Starting evaluation (Mode: {mode.upper()}) for Epoch {epoch}...")
    
    if mode == 'policy' and sampler is None:
        raise ValueError("Sampler must be provided for 'policy' mode evaluation.")
    if mode == 'knn' and (embedding_model is None or num_examples_knn is None):
        raise ValueError("EmbeddingModel and num_examples_knn must be provided for 'knn' mode evaluation.")

    if mode == 'policy':
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

            buffer = RolloutBuffer(num_examples_knn, batch_size, embedding_model.dim, utils.device)
            buffer.log_probs = None 
            buffer.values = None
            
            query_texts = [item['query'] for item in query_batch_list]
            buffer.queries = query_batch_list
            query_embs = embedding_model.encode(query_texts)

            sim_scores = torch.matmul(query_embs, corpus_embeddings.T)
            
            _, top_k_indices = torch.topk(sim_scores, k=num_examples_knn, dim=1)
            
            for t in range(num_examples_knn):
                actions = top_k_indices[:, t] # (B,)
                selected_example_embeddings = corpus_embeddings[actions]
                selected_example_texts = [corpus_data[idx.item()] for idx in actions]
                
                buffer.add_step_data(
                    step=t,
                    actions=actions,
                    example_embeddings=selected_example_embeddings,
                    example_texts=selected_example_texts

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

    os.makedirs("outputs", exist_ok=True) 
    output_filename = f"outputs/val_results_{config.RUN_NAME}_epoch_{epoch}_{mode}.csv"
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

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    utils.setup_logging(log_level="INFO", log_file=config.LOG_FILE)
    utils.initialize_seeds(config.SEED)
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
        rnn_type=config.AGENT_RNN_TYPE,
        rnn_layers=config.AGENT_RNN_LAYERS,
        dropout=config.AGENT_DROPOUT
    ).to(device)

    if os.path.exists(config.PRETRAINED_MODEL_PATH):
        agent.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=device))
        logger.info(f"Successfully loaded pre-trained MMR weights from: {config.PRETRAINED_MODEL_PATH}")
    else:
        logger.warning(f"Pre-trained model not found at: {config.PRETRAINED_MODEL_PATH}")
        logger.warning("Starting PPO training from random initialization.")

    logger.info("--- Loading Data ---")
    corpus_data, corpus_embeddings = dataloader.get_corpus()
    logger.info(f"Corpus embeddings computed. Shape: {corpus_embeddings.shape}")
    
    train_loader = dataloader.get_dataloader(
        split='train', 
        batch_size=config.BATCH_SIZE, 
        shuffle=True
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
        query_sim_weight=config.QUERY_SIM_WEIGHT,
        sample_sim_weight=config.SAMPLE_SIM_WEIGHT,
        final_loss_weight=config.FINAL_LOSS_WEIGHT,
        system_prompt=config.SYSTEM_PROMPT,
        prompt_strategy=config.PROMPT_STRATEGY
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
        prompt_strategy=config.PROMPT_STRATEGY,
        epoch="0_knn_baseline", 
        mode='knn',           
        embedding_model=embedding_model,
        num_examples_knn=config.NUM_EXAMPLES 
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
                llm_wrapper=llm_wrapper,
                embedding_model=embedding_model,
                corpus_embeddings=corpus_embeddings
            )

            log_dict = ppo_trainer.train_step(
                buffer=buffer,
                corpus_embeddings=corpus_embeddings
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
                prompt_strategy=config.PROMPT_STRATEGY,
                epoch=f"{epoch + 1}", 
                mode='policy',     
                sampler=sampler        
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                logger.info(f"New best validation accuracy: {best_val_accuracy:.2f}%! Saving model...")
                torch.save(agent.state_dict(), f"checkpoints/{config.RUN_NAME}_best.pt")
            torch.save(agent.state_dict(), f"checkpoints/train/{config.RUN_NAME}_epoch_{epoch+1}.pt")

    logger.info("--- Final evaluation after all epochs ---")
    val_accuracy = run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        prompt_strategy=config.PROMPT_STRATEGY,
        epoch = "final",
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