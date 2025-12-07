import torch
import torch.optim as optim
import os
from tqdm import tqdm
from typing import List, Dict, Optional
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb

from config import train_config as config 
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.sampler import EpisodeSampler, RolloutBuffer 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    
    print(f"Starting evaluation (Mode: {mode.upper()}) for Epoch {epoch}...")
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
    
    print(
        f"Evaluation Epoch {epoch} (Mode: {mode.upper()}) Finished. "
        f"Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples}), "
        f"Avg NLL: {avg_nll:.4f}"
    )

    print("Saving evaluation results to file...")
    
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
    print(f"Evaluation results saved to: {output_filename}")

    for i in range(min(3, len(all_targets))):
        if not all_correct[i]:
            print(f"--- Bad Case Example {i+1} ---")
            print(f"PROMPT:\n{all_prompts[i]}")
            print(f"TARGET: {all_targets[i]}")
            print(f"PRED:   {all_generated_texts[i]}")
            print(f"NLL:    {all_generated_nlls[i]:.4f}")
            print("--------------------------")
            
    return accuracy

def train():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    # utils.setup_logging(log_level="INFO", log_file=os.path.join(config.LOG_DIR, f"{config.RUN_NAME}.log"))
    device = utils.device
    print(f"Using device: {device}")
    print(f"Starting run: {config.RUN_NAME}")

    print("--- Initializing Models ---")
    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(model_name=config.LLM_MODEL_NAME)
    agent = PolicyNetwork(
        embedding_dim=embedding_model.dim,
        hidden_dim=config.AGENT_HIDDEN_DIM,
        dropout=config.AGENT_DROPOUT 
    ).to(device)

    print(f"Loading pretrained model from: {config.PRETRAINED_PATH}")
    state_dict = torch.load("cache/lambda_icl_qwen_0.6b/pre_mdl_1205_1520_val_best.pt", map_location=device)
    agent.load_state_dict(state_dict)

    print("--- Loading Data ---")
    corpus_data, corpus_embeddings = dataloader.get_corpus()

    val_loader = dataloader.get_dataloader(split='dev', batch_size=config.BATCH_SIZE, shuffle=False)
    
    sampler = EpisodeSampler(policy_network=agent, embedding_model=embedding_model, num_examples=config.NUM_EXAMPLES)

    print("--- Running Pre-Training Evaluation (MMR Baseline) ---")
    val_accuracy = run_evaluation(llm_wrapper=llm_wrapper,
        val_loader=val_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=config.SYSTEM_PROMPT,
        epoch=0,
        mode='policy',
        sampler= sampler,
        embedding_model=embedding_model,
        num_examples=config.NUM_EXAMPLES
    )   
    print(f"Validation Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print("An unhandled exception occurred!")
        raise e