import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import pandas as pd 

from config import pretrain_config as config
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.reward_computer import RewardComputer
from engine.sampler import RolloutBuffer, EpisodeSampler
from utils import device

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

logger = logging.getLogger(__name__)


@torch.no_grad()

def run_evaluation(llm_wrapper: LLMWrapper,val_loader: torch.utils.data.DataLoader,corpus_data: List[Dict[str, str]],corpus_embeddings: torch.Tensor,check_correct_fn: callable,system_prompt: str,prompt_strategy: str,mode: str,sampler: Optional[EpisodeSampler] = None, embedding_model: Optional[EmbeddingModel] = None, num_examples: int = 4) -> float:
    
    logger.info(f"Starting evaluation (Mode: {mode})...")
    
    if mode == 'policy' and sampler is None:
        raise ValueError("Sampler must be provided for 'policy' mode evaluation.")
    if mode == 'mmr' and (embedding_model is None):
        raise ValueError(f"EmbeddingModel must be provided for '{mode}' mode evaluation ")

    if mode == 'policy':
        sampler.policy_network.eval() 
    elif mode == "mmr":
        logger.info(f"with MMR lambda = {config.MMR_LAMBDA}.")
    
    total_correct = 0
    total_nll = 0.0 
    total_samples = 0
    
    all_prompts: List[str] = []
    all_generated_texts: List[str] = []
    all_generated_nlls: List[float] = []
    all_targets: List[str] = []

    for query_batch_list in tqdm(val_loader, desc=f"Validating pretrain ({mode})"):
        batch_size = len(query_batch_list)
        
        if mode == 'policy':
            buffer = sampler.collect_episodes(
                query_batch=query_batch_list,
                corpus=corpus_data,
                corpus_embeddings=corpus_embeddings
            )
        else:
            buffer = RolloutBuffer(num_examples, batch_size, embedding_model.dim, utils.device)
            buffer.log_probs = None 
            buffer.values = None
            
            query_texts = [item['query'] for item in query_batch_list]
            buffer.queries = query_batch_list
            query_embs = embedding_model.encode(query_texts)

            sim_scores = torch.matmul(query_embs, corpus_embeddings.T)
            
            if mode == 'mmr':
                selected_mask = torch.zeros_like(sim_scores, dtype=torch.bool)
                relevance_scores = sim_scores
                batch_selected_embs_mmr = torch.zeros((batch_size, num_examples, embedding_model.dim), device=utils.device)

                for t in range(num_examples):
                    if t == 0:
                        step_scores = relevance_scores
                    else:
                        selected_embs_so_far = batch_selected_embs_mmr[:, :t, :]
                        corpus_expanded = corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                        sim_to_selected = torch.bmm(
                            corpus_expanded, 
                            selected_embs_so_far.transpose(1, 2)
                        )
                        diversity_penalty, _ = torch.max(sim_to_selected, dim=2)
                        step_scores = (config.MMR_LAMBDA * relevance_scores) - \
                                      ((1 - config.MMR_LAMBDA) * diversity_penalty)

                    step_scores.masked_fill_(selected_mask, -torch.inf)
                    current_action = torch.argmax(step_scores, dim=1)

                    current_embs = corpus_embeddings[current_action]
                    selected_example_texts = [corpus_data[idx.item()] for idx in current_action]
                    batch_selected_embs_mmr[:, t, :] = current_embs

                    buffer.add_step_data(
                        step=t,
                        actions=current_action,
                        example_embeddings=current_embs,
                        example_texts=selected_example_texts
                    )
                    
                    selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)
        
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
        f"Evaluation Pretrain (Mode: {mode}) Finished. "
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

    output_dir = os.path.join("results",config.PROJECT_NAME,config.RUN_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"val_results_pretrain_{mode}.csv")
    df.to_csv(output_filename, index=False, encoding='utf-8')
    logger.info(f"Evaluation results saved to: {output_filename}")
            
    return accuracy

@torch.no_grad()
def generate_mmr_trajectories(query_loader: DataLoader,corpus_data: List[Dict[str, str]],corpus_embeddings: torch.Tensor,embedding_model: EmbeddingModel) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:

    logger.info("--- Generating MMR Expert Trajectories ---")

    original_to_new_corpus_map = {
        item['corpus_index']: new_idx 
        for new_idx, item in enumerate(corpus_data)
        if 'corpus_index' in item
    }

    all_query_embs = []
    all_expert_actions = []
    all_selected_embs = []
    all_query_data = []
    
    for query_batch_list in tqdm(query_loader, desc="Generating MMR data"):
        batch_size = len(query_batch_list)

        query_texts = [item['query'] for item in query_batch_list]
        query_embs = embedding_model.encode(query_texts) 
        all_query_embs.append(query_embs)
        all_query_data.extend(query_batch_list)

        batch_actions = torch.zeros((batch_size, config.NUM_EXAMPLES), dtype=torch.long, device=utils.device)
        batch_selected_embs = torch.zeros((batch_size, config.NUM_EXAMPLES, embedding_model.dim), device=utils.device)

        relevance_scores = torch.matmul(query_embs, corpus_embeddings.T)
        selected_mask = torch.zeros_like(relevance_scores, dtype=torch.bool)

        query_indices_to_mask = []
        for i, item in enumerate(query_batch_list):
            if 'corpus_index' in item:
                original_idx = item['corpus_index']
         
                if original_idx in original_to_new_corpus_map:
                    new_corpus_idx = original_to_new_corpus_map[original_idx]
                    query_indices_to_mask.append((i, new_corpus_idx))

        if query_indices_to_mask:
            batch_indices = torch.tensor([idx[0] for idx in query_indices_to_mask], device=utils.device)
            corpus_indices = torch.tensor([idx[1] for idx in query_indices_to_mask], device=utils.device)
            selected_mask[batch_indices, corpus_indices] = True  

        for t in range(config.NUM_EXAMPLES):
            if t == 0:
                step_scores = relevance_scores
            else:
                selected_embs_so_far = batch_selected_embs[:, :t, :]
                corpus_expanded = corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                sim_to_selected = torch.bmm(
                    corpus_expanded, 
                    selected_embs_so_far.transpose(1, 2)
                )

                diversity_penalty, _ = torch.max(sim_to_selected, dim=2)

                step_scores = (config.MMR_LAMBDA * relevance_scores) - \
                              ((1 - config.MMR_LAMBDA) * diversity_penalty)

            step_scores.masked_fill_(selected_mask, -torch.inf)

            current_action = torch.argmax(step_scores, dim=1)

            batch_actions[:, t] = current_action
            current_embs = corpus_embeddings[current_action] # (B, E)
            batch_selected_embs[:, t, :] = current_embs

            selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)
            
        all_expert_actions.append(batch_actions)
        all_selected_embs.append(batch_selected_embs)
        
    all_query_embs = torch.cat(all_query_embs, dim=0)
    all_expert_actions = torch.cat(all_expert_actions, dim=0)
    all_selected_embs = torch.cat(all_selected_embs, dim=0)
    
    logger.info(f"Generated {all_query_embs.shape[0]} MMR trajectories.")
    return all_query_embs, all_expert_actions, all_selected_embs, all_query_data

@torch.no_grad()
def compute_expert_returns(query_data: List[Dict[str,str]],
                           expert_actions: torch.Tensor,
                           llm_wrapper: LLMWrapper,
                           reward_computer: RewardComputer,
                           corpus_data: List[Dict[str, str]],
                           batch_size: int,
                           embedding_dim: int ) -> torch.Tensor:

    logger.info("--- Computing Expert Returns for MMR Trajectories ---")
    
    all_returns = []
    
    num_queries = len(query_data)
    num_actions = expert_actions.shape[0]
    num_total = min(num_queries, num_actions)

    for i in tqdm(range(0, num_total, batch_size), desc="Computing expert returns"):

        b_queries = query_data[i : i + batch_size]
        b_actions = expert_actions[i : i + batch_size]
        b_size_current = b_actions.shape[0]

        buffer = RolloutBuffer(
            num_steps=config.NUM_EXAMPLES,
            batch_size=b_size_current,
            embedding_dim=embedding_dim, 
            device=utils.device
        )
        buffer.log_probs = None
        buffer.values = None 
        buffer.queries = b_queries
        buffer.actions = b_actions
        
        for j in range(b_size_current):
            actions_list = b_actions[j].tolist()
            buffer.selected_examples_text[j] = [corpus_data[idx] for idx in actions_list]
        
        buffer = reward_computer.compute_rewards_and_advantages(
            buffer=buffer,
            llm_wrapper=llm_wrapper,
            embedding_model=None, 
            corpus_embeddings=None
        )
        
        all_returns.append(buffer.returns)
        
    return torch.cat(all_returns, dim=0)

def pretrain_agent(agent: PolicyNetwork,
                   optimizer: optim.Optimizer,
                   pretrain_dataset: TensorDataset,
                   device: torch.device):
    
    logger.info(f"--- Starting Supervised Pre-training ---")
    logger.info(f"Target Actor Loss: {config.PRETRAIN_LOSS_THRESHOLD}")
    logger.info(f"Max Epochs: {config.PRETRAIN_MAX_EPOCHS}")
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    crit_loss_fn = torch.nn.MSELoss()
    
    epoch = 0
    current_actor_loss = float('inf')
    
    while epoch < config.PRETRAIN_MAX_EPOCHS:
        epoch += 1
        agent.train() 
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        pbar = tqdm(pretrain_loader, desc=f"Pre-train Epoch {epoch}/{config.PRETRAIN_MAX_EPOCHS}")
        
        for batch in pbar:
            query_embs, expert_actions, expert_selected_embs, expert_returns = batch

            query_embs = query_embs.to(device)
            expert_actions = expert_actions.to(device)
            expert_selected_embs = expert_selected_embs.to(device)
            expert_returns = expert_returns.to(device)

            _, new_values, _, all_logits = agent.forward(
                query_embeddings=query_embs,
                selected_example_embeddings=expert_selected_embs,
                selected_actions=expert_actions,
                corpus_embeddings=corpus_embeddings 
            )

            logits_flat = all_logits.view(-1, corpus_embeddings.shape[0])
            actions_flat = expert_actions.view(-1)
            
            actor_loss = F.cross_entropy(logits_flat, actions_flat)
            critic_loss = crit_loss_fn(new_values, expert_returns)
            
            total_loss = actor_loss + (critic_loss * config.V_LOSS_COEF)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
            pbar.set_postfix({
                "actor_loss(CE)": f"{actor_loss.item():.4f}",
                "critic_loss(MSE)": f"{critic_loss.item():.4f}"
            })
            
        avg_actor_loss = total_actor_loss / len(pretrain_loader)
        avg_critic_loss = total_critic_loss / len(pretrain_loader)
        logger.info(f"Pre-train Epoch {epoch} finished. "
                    f"Avg Actor Loss: {avg_actor_loss:.4f}, "
                    f"Avg Critic Loss: {avg_critic_loss:.4f}")
        
        if avg_actor_loss < config.PRETRAIN_LOSS_THRESHOLD:
            logger.info(f"Actor loss {avg_actor_loss:.4f} is below threshold {config.PRETRAIN_LOSS_THRESHOLD}.")
            logger.info("Stopping pre-training.")
            break
    
    if epoch == config.PRETRAIN_MAX_EPOCHS and avg_actor_loss >= config.PRETRAIN_LOSS_THRESHOLD:
        logger.warning(f"Reached max pre-training epochs ({config.PRETRAIN_MAX_EPOCHS}) without reaching loss threshold.")
        logger.warning(f"Final actor loss: {avg_actor_loss:.4f} (Threshold: {config.PRETRAIN_LOSS_THRESHOLD})")


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
    agent = PolicyNetwork(
        embedding_dim=embedding_model.dim,
        hidden_dim=config.AGENT_HIDDEN_DIM,
        rnn_type=config.AGENT_RNN_TYPE
    ).to(device)
    
    corpus_data, corpus_embeddings_cpu = dataloader.get_corpus() 
    corpus_embeddings = corpus_embeddings_cpu.to(device)
    logger.info(f"Corpus loaded. Size: {len(corpus_data)}, Embeddings shape: {corpus_embeddings.shape}")

    logger.info("Loading 'dev' split for MMR evaluation...")
    val_loader = dataloader.get_dataloader(
        split='dev',
        batch_size=config.BATCH_SIZE_VAL,
        shuffle=False 
    )

    logger.info("--- Running MMR Expert Baseline Evaluation ---")
    run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        prompt_strategy=config.PROMPT_STRATEGY, 
        mode='mmr',
        embedding_model=embedding_model
    )
    logger.info("--- MMR Expert Baseline Evaluation Finished ---")

    logger.info("Clearing CUDA cache after MMR evaluation to free VRAM...")
    torch.cuda.empty_cache()

    pretrain_query_loader = dataloader.get_dataloader(
        split='train', 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    query_embs, expert_actions, expert_selected_embs, query_data_list = generate_mmr_trajectories(
        query_loader=pretrain_query_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model
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
    
    expert_returns = compute_expert_returns(
        query_data=query_data_list,
        expert_actions=expert_actions,
        llm_wrapper=llm_wrapper,
        reward_computer=reward_computer,
        corpus_data=corpus_data,
        batch_size=config.BATCH_SIZE,
        embedding_dim=embedding_model.dim
    )

    pretrain_dataset = TensorDataset(
        query_embs.cpu(), 
        expert_actions.cpu(), 
        expert_selected_embs.cpu(), 
        expert_returns.cpu()
    )
    
    optimizer = optim.AdamW(agent.parameters(), lr=config.PRETRAIN_LR)

    pretrain_agent(
        agent=agent,
        optimizer=optimizer,
        pretrain_dataset=pretrain_dataset,
        device=device 
    )

    logger.info("--- Initializing Sampler for Policy Network Evaluation ---")
    sampler = EpisodeSampler(
        policy_network=agent,
        embedding_model=embedding_model,
        num_examples=config.NUM_EXAMPLES
    )

    logger.info("--- Running Evaluation on Pre-trained Policy Network ---")
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
    logger.info("--- Policy Network Evaluation Finished ---")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.state_dict(), config.PRETRAINED_MODEL_PATH)
    logger.info(f"--- MMR Pre-training Finished ---")
    logger.info(f"Pre-trained model saved to: {config.PRETRAINED_MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("An unhandled exception occurred!", exc_info=True)
        raise e