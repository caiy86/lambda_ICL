import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import logging
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Dict

# 导入你的项目模块
from config import train_config as config
import utils
import data_utils.mtop_loader as dataloader
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from models.policy_network import PolicyNetwork
from engine.sampler import EpisodeSampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_loss_correctly(llm_wrapper: LLMWrapper, prompts: List[str], targets: List[str]):

    tokenizer = llm_wrapper.tokenizer
    model = llm_wrapper.model
    device = llm_wrapper.device
    
    tokenizer.padding_side = 'right'
    batch_input_ids = []
    batch_labels = []
    
    for p, t in zip(prompts, targets):

        p_ids = tokenizer(p, return_tensors='pt', add_special_tokens=False).input_ids[0]
        t_ids = tokenizer(t + tokenizer.eos_token, return_tensors='pt', add_special_tokens=False).input_ids[0]
        
        input_ids = torch.cat([p_ids, t_ids])
        
        label_ids = torch.cat([
            torch.full_like(p_ids, -100), 
            t_ids
        ])
        
        batch_input_ids.append(input_ids)
        batch_labels.append(label_ids)
    
    from torch.nn.utils.rnn import pad_sequence
    padded_inputs = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    padded_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100).to(device)
    attention_mask = (padded_inputs != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model(input_ids=padded_inputs, attention_mask=attention_mask)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = padded_labels[..., 1:].contiguous()
        
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape)
        
        loss_mask = (shift_labels != -100)
        per_sample_loss = loss.sum(dim=1) / (loss_mask.sum(dim=1) + 1e-9)
        
    return per_sample_loss.cpu()

def select_examples_fixed_lambda(
    query_texts: List[str],
    corpus_data: List[Dict],
    corpus_embeddings: torch.Tensor,
    embedding_model: EmbeddingModel,
    fixed_lambda: float,
    num_examples: int = 8,
    query_indices: List[int] = None 
):
    device = utils.device
    batch_size = len(query_texts)
    
    query_embeddings = embedding_model.encode(query_texts) # (B, D)
    
    batch_selected_indices = torch.zeros((batch_size, num_examples), dtype=torch.long, device=device)
    batch_selected_embs = torch.zeros((batch_size, num_examples, embedding_model.dim), device=device)
    
    sim_scores = torch.matmul(query_embeddings, corpus_embeddings.T)
    relevance_scores = sim_scores
    selected_mask = torch.zeros_like(sim_scores, dtype=torch.bool)

    if query_indices is not None:
        self_indices = torch.tensor(query_indices, device=device, dtype=torch.long)
        valid_mask = self_indices >= 0
        if valid_mask.any():
            batch_rows = torch.arange(batch_size, device=device)
            selected_mask[batch_rows[valid_mask], self_indices[valid_mask]] = True
    
    for t in range(num_examples):
        if t == 0:
            step_scores = relevance_scores
        else:
            selected_embs_so_far = batch_selected_embs[:, :t, :]
            corpus_expanded = corpus_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            sim_to_selected = torch.bmm(corpus_expanded, selected_embs_so_far.transpose(1, 2))
            diversity_penalty, _ = torch.max(sim_to_selected, dim=2)
            
            step_scores = (fixed_lambda * relevance_scores) - ((1 - fixed_lambda) * diversity_penalty)
            
        step_scores.masked_fill_(selected_mask, -torch.inf)
        current_action = torch.argmax(step_scores, dim=1)
        
        batch_selected_indices[:, t] = current_action
        batch_selected_embs[:, t, :] = corpus_embeddings[current_action]
        selected_mask.scatter_(dim=1, index=current_action.unsqueeze(1), value=True)
        
    selected_examples_list = []
    for i in range(batch_size):
        indices = batch_selected_indices[i].cpu().tolist()
        selected_examples_list.append([corpus_data[idx] for idx in indices])
        
    return selected_examples_list

def main():
    device = utils.device
    logger.info(f"Running Debug Analysis on device: {device}")
    
    logger.info("Loading Models...")
    embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)
    llm_wrapper = LLMWrapper(model_name=config.LLM_MODEL_NAME)
    
    logger.info("Loading Data...")
    corpus_data, corpus_embeddings = dataloader.get_corpus()
    
    train_loader = dataloader.get_dataloader(
        # split='train', 
        split = 'dev',
        batch_size=32,  
        shuffle=True, 
        nums=256 
    )
    
    agent = None
    # pretrained_path = "cache/lambda_icl_qwen_0.6b/pre_mdl_1128_1409.pt"
    pretrained_path = "cache/lambda_icl_qwen_0.6b/1130_1324_best.pt"

    if os.path.exists(pretrained_path):
        logger.info(f"Loading Pretrained Policy from {pretrained_path}")
        agent = PolicyNetwork(embedding_model.dim, config.AGENT_HIDDEN_DIM).to(device)
        agent.load_state_dict(torch.load(pretrained_path, map_location=device))
        agent.eval()
        sampler = EpisodeSampler(agent, embedding_model, config.NUM_EXAMPLES)
    else:
        logger.warning(f"Pretrained model not found at {pretrained_path}. Skipping Policy Test.")
        sampler = None

    modes = ["Fixed-Lambda-0.7"]
    if agent is not None:
        modes.append("Pretrained-Policy")
    
    results = []

    for mode in modes:
        logger.info(f"================ Testing Mode: {mode} ================")
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            query_texts = [b['query'] for b in batch]
            target_answers = [b['answer'] for b in batch]

            batch_indices = [b.get('corpus_index', -1) for b in batch]

            if mode == "Fixed-Lambda-0.7":
                selected_examples = select_examples_fixed_lambda(
                    query_texts, corpus_data, corpus_embeddings, embedding_model, fixed_lambda=0.7,query_indices=batch_indices
                )
            elif mode == "Pretrained-Policy":
                buffer = sampler.collect_episodes(batch, corpus_data, corpus_embeddings)
                selected_examples = buffer.selected_examples_text

            prompts = []
            for i, q_text in enumerate(query_texts):
                p = llm_wrapper.build_chat_prompt(
                    system_prompt=config.SYSTEM_PROMPT,
                    examples=selected_examples[i],
                    query=q_text
                )
                prompts.append(p)
            
            losses = compute_loss_correctly(llm_wrapper, prompts, target_answers)
            avg_batch_loss = losses.mean().item()
            total_loss += avg_batch_loss * len(batch)
            
            preds, _ = llm_wrapper.generate_for_evaluation(prompts, max_new_tokens=100)
            
            correct_count = 0
            for i, (tgt, pred) in enumerate(zip(target_answers, preds)):
                is_correct = dataloader.check_correct(tgt, pred)
                if is_correct:
                    correct_count += 1

                if batch_idx == 0 and i == 0:
                    logger.info(f"--- Sample Debug ({mode}) ---")
                    logger.info(f"Loss: {losses[i].item():.4f}")
                    logger.info(f"Target: {tgt}")
                    logger.info(f"Pred:   {pred}")
                    logger.info(f"Correct: {is_correct}")
                    logger.info("----------------------------")

            total_correct += correct_count
            total_samples += len(batch)
            
            logger.info(f"Batch {batch_idx+1}: Loss={avg_batch_loss:.4f}, Acc={correct_count}/{len(batch)}")
        
        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100
        logger.info(f"Mode {mode} Result: Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
        results.append({"Mode": mode, "Loss": avg_loss, "Accuracy": accuracy})

    logger.info("================ Final Comparison ================")
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()