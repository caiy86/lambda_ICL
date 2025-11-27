import torch
import torch.optim as optim
import os
import logging
from tqdm import tqdm
from typing import List, Dict,Optional
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import test_config as config 
import utils
import data_utils.mtop_loader as dataloader 
from models.embedding_model import EmbeddingModel
from models.llm_wrapper import LLMWrapper
from engine.sampler import EpisodeSampler, RolloutBuffer 

logger = logging.getLogger(__name__)

def run_evaluation(llm_wrapper: LLMWrapper,val_loader: torch.utils.data.DataLoader,corpus_data: List[Dict[str, str]],corpus_embeddings: torch.Tensor,check_correct_fn: callable,system_prompt: str,prompt_strategy: str,sampler: Optional[EpisodeSampler] = None, embedding_model: Optional[EmbeddingModel] = None, num_examples: int = 4) -> float:
   
    total_correct = 0
    total_nll = 0.0 
    total_samples = 0
    
    all_prompts: List[str] = []
    all_generated_texts: List[str] = []
    all_generated_nlls: List[float] = []
    all_targets: List[str] = []

    for query_batch_list in tqdm(val_loader, desc=f"Validating pretrain"):
        batch_size = len(query_batch_list)
        
        buffer = RolloutBuffer(num_examples, batch_size, embedding_model.dim, utils.device)
        buffer.log_probs = None 
        buffer.values = None
        
        query_texts = [item['query'] for item in query_batch_list]
        buffer.queries = query_batch_list
        query_embs = embedding_model.encode(query_texts)

        sim_scores = torch.matmul(query_embs, corpus_embeddings.T)

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
            intents, slots = dataloader.extract_schema(example_data)
            _system_prompt = system_prompt.replace('<intents list>', intents).replace('<slots list>', slots)
            prompt_str = llm_wrapper.build_chat_prompt(
                system_prompt=_system_prompt,
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
        f"Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples}), "
        f"Avg NLL: {avg_nll:.4f}"
    )

    logger.info("Saving evaluation results to file...")
    all_correct = [
        check_correct_fn(target, pred) 
        for target, pred in zip(all_targets, all_generated_texts)
    ]
    df = pd.DataFrame({
        "target": all_targets,
        "prediction": all_generated_texts,
        "is_correct": all_correct,
        "nll": all_generated_nlls,
        "prompt": all_prompts,
    })

    output_dir = os.path.join("results",config.PROJECT_NAME,config.RUN_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"val_results_test.csv")
    df.to_csv(output_filename, index=False, encoding='utf-8')
    logger.info(f"Evaluation results saved to: {output_filename}")
            
    return accuracy

def test():

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
        shuffle=False,
        nums = None
        # nums = config.VAL_NUMS
    )
    system_prompts = [
        'You are an expert assistant for semantic parsing. Given a user utterance, you must convert it into its logical form representation.',
'''You are a highly specialized semantic parsing assistant. Your sole task is to convert a User Utterance into its corresponding Logical Form (LF) following a strict, defined format.

1. Logical Form (LF) Structure Rules:

* **Basic Unit:** The Logical Form is constructed using nested square brackets `[ ... ]`.
* **Intent:**
  * The entire Logical Form MUST be wrapped by one **Intent** tag.
  * The Intent format is `[IN:INTENT_NAME ... ]`, where `INTENT_NAME` is the name of the intent.
* **Slot:**
  * Intents contain zero or more **Slots**.
  * The Slot format is `[SL:SLOT_NAME slot_value ]`, where `SLOT_NAME` is the name of the slot, and `slot_value` is the text extracted from the user utterance.
* **Strict Formatting:** The final output must be a single, well-formed string starting with `[IN:` and ending with `]`.

2. Task-Specific Domain Definition:

 you MUST **only** use the Intents and Slots defined in the lists below.

* **Available Intents List:**
  <intents list>

* **Available Slots List:**
  <slots list>
''',
'''You are a highly specialized semantic parsing assistant. Your sole task is to convert a User Utterance into its corresponding Logical Form (LF) following a strict, defined format.

1. Logical Form (LF) Structure Rules:

* **Basic Unit:** The Logical Form is constructed using nested square brackets `[ ... ]`.

* **Intent (IN):**
  * The Intent format is `[IN:INTENT_NAME ... ]`, where `INTENT_NAME` is the name of the intent.
  * The entire Logical Form MUST start with a single **Top-Level Intent**.
  * **Nesting:** Additional `[IN:...]` structures (Nested Intents) are allowed, but they can **only** appear as the value of a Slot.
  * *(Example: [SL:TODO [IN:GET_TODO ... ] ])*

* **Slot (SL):**
  * The Slot format is `[SL:SLOT_NAME slot_value ]`, where `SLOT_NAME` is the slot's name.
  * The `slot_value` can be one of two things:
    1.  **Text:** A span of text.
    2.  **A Nested Intent:** Another `[IN:...]` structure.

* **Verbatim Extraction (CRITICAL):**
  * When a `slot_value` is text (and not a nested Intent), that text MUST be extracted *exactly* and *verbatim* from the original User Utterance.
  * You must not change, rephrase, or add any text that was not present in the utterance.
  * *(Example: For `[SL:TODO a playdate ]`, the text "a playdate" must exist in the user utterance).*

* **Strict Formatting:** The final output must be a single, well-formed string starting with `[IN:` and ending with `]`.

2. Task-Specific Domain Definition:

For this task, you MUST **only** use the Intents and Slots defined in the lists below.

* **Available Intents List:**
  <intents list>

* **Available Slots List:**
  <slots list>
'''
    ]

    run_evaluation(
        llm_wrapper=llm_wrapper,
        val_loader=val_loader,
        corpus_data=corpus_data,
        corpus_embeddings=corpus_embeddings,
        check_correct_fn=dataloader.check_correct,
        system_prompt=system_prompts[2],
        prompt_strategy=config.PROMPT_STRATEGY,       
        embedding_model=embedding_model,
        num_examples=config.NUM_EXAMPLES 
    )

if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        logger.error("An unhandled exception occurred!", exc_info=True)
        raise e