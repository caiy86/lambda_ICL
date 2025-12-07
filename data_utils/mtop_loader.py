import logging
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Set
import random
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from models.embedding_model import EmbeddingModel 
from config import train_config as config 
import utils
import re

logger = logging.getLogger(__name__)

def _get_full_train_data_and_embeddings() -> Tuple[List[Dict], torch.Tensor]:

    logger.info("Loading full 'train' split (no global cache used)...")
    full_train_data = _load_hf_data(split='train')

    logger.info(f"Loading EmbeddingModel ({config.EMBEDDING_MODEL_NAME}) for K-Means/Sampling...")
    emb_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME)

    logger.info("Pre-computing all training data embeddings...")
    train_queries = [item['query'] for item in full_train_data]
    full_train_embeddings = emb_model.encode(
        train_queries, 
        batch_size=config.BATCH_SIZE, 
        show_progress_bar=True
    )

    logger.info(f"Full train data and embeddings loaded ({len(full_train_data)} samples).")
    del emb_model
    torch.cuda.empty_cache()

    return full_train_data, full_train_embeddings

def list_dict_collate_fn(batch: List[Dict]) -> List[Dict]:

    return batch

def _load_hf_data(split: str) -> List[Dict]:

    if split == "dev":
        hf_split = "validation"
    elif split == "test":
        hf_split = "test"
    else:
        hf_split = "train"
    
    try:
        dataset = load_dataset("KaiLv/UDR_MTOP", split=hf_split)
    except Exception as e:
        logger.error(f"Failed to load dataset 'KaiLv/UDR_MTOP' split '{hf_split}'. Error: {e}")
        raise e

    data = []
    for i, sample in enumerate(dataset):
        item = {
            "query": sample['question'],
            "answer": sample['logical_form']
        }
        if split == 'train':
            item['corpus_index'] = i
        data.append(item)
    
    return data

class MtopQueryDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

def get_corpus() -> Tuple[List[Dict[str, str]], torch.Tensor]:

    logger.info("--- Creating Corpus (Action Space) ---")
    
    full_train_data, full_train_embeddings = _get_full_train_data_and_embeddings()
    
    corpus_data = full_train_data
    corpus_embeddings = full_train_embeddings

    
    logger.info(f"Corpus created. Final size: {len(corpus_data)}")
    return corpus_data, corpus_embeddings.to(utils.device)

def get_dataloader(
    split: str, 
    batch_size: int, 
    shuffle: bool = False,
    nums: int = None,
    seed: int = None
) -> DataLoader:

    logger.info(f"Loading query dataloader (split={split}, batch_size={batch_size}, shuffle={shuffle}, nums={nums}, seed={seed})...")

    data = _load_hf_data(split=split)

    if nums is not None and len(data) > nums:
        logger.info(f"Randomly sampling {nums} examples for '{split}' split...")
        if seed is None:
            # Truly random
            query_data = random.sample(data, nums)
        else:
            rng = random.Random(seed)
            query_data = rng.sample(data, nums)
    else:
        if nums is not None:
            logger.info(f"Requested nums ({nums}) is larger than dataset size ({len(data)}). Using full dataset.")
        query_data = data

    dataset = MtopQueryDataset(query_data)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=False,
        collate_fn=list_dict_collate_fn
    )

def get_train_val_split_data(
    split: str,
    train_nums: int,
    val_nums: int,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:

    logger.info(f"Splitting '{split}' data: Train={train_nums}, Val={val_nums}, Seed={seed}")
    
    all_data = _load_hf_data(split=split)
    total_needed = train_nums + val_nums
    
    if len(all_data) < total_needed:
        logger.warning(f"Requested {total_needed} samples but only {len(all_data)} available. Using full dataset.")

    rng = random.Random(seed)
    rng.shuffle(all_data)
    
    train_data = all_data[:train_nums]
    val_data = all_data[train_nums : train_nums + val_nums]
    
    logger.info(f"Split result -> Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_indices = set(item.get('corpus_index') for item in train_data)
    val_indices = set(item.get('corpus_index') for item in val_data)
    intersection = train_indices.intersection(val_indices)
    assert len(intersection) == 0, f"Error: Train and Val sets overlap! ({len(intersection)} samples)"
    
    return train_data, val_data

def _split_children(content: str) -> List[str]:

    children = []
    bracket_level = 0
    current_start = -1

    for i, char in enumerate(content):
        if char == '[':
            if bracket_level == 0:
                current_start = i
            bracket_level += 1
        elif char == ']':
            bracket_level -= 1
            if bracket_level == 0 and current_start != -1:
                children.append(content[current_start : i + 1])
                current_start = -1
    return children

def canonicalize_lf(lf: str) -> str:

    lf = lf.strip().lower()
    
    if not lf.startswith('['):
        return lf

    try:
        tag_end = lf.index(':')
        
        name_end = -1
        for i in range(tag_end + 1, len(lf)):
            if lf[i] == ' ':
                name_end = i
                break
            if lf[i] == ']': 
                name_end = i
                break
        
        if name_end == -1: return lf 

        tag = lf[1:tag_end]        
        name = lf[tag_end+1:name_end] 
        
        content = lf[name_end+1:-1].strip()

    except ValueError:
        return lf 
    
    if tag == 'sl':
        canonical_value = canonicalize_lf(content)
        return f"[sl:{name} {canonical_value}]"
        
    elif tag == 'in':
        children = _split_children(content)

        canonical_children = [canonicalize_lf(child) for child in children]

        canonical_children.sort()

        joined_children = " ".join(canonical_children)
        return f"[in:{name} {joined_children}]"
        
    else:
        return lf 

def check_correct(target_answer: str, pred_text: str) -> bool:

    canonical_target = canonicalize_lf(target_answer)
    canonical_pred = canonicalize_lf(pred_text)

    return canonical_target == canonical_pred

def extract_schema(examples: List[Dict[str, str]], ) -> Tuple[str, str]:

    intent_pattern = re.compile(r"\[IN:([^ ]+)")
    slot_pattern = re.compile(r"\[SL:([^ ]+)")
    all_intents: Set[str] = set()
    all_slots: Set[str] = set()

    for ex in examples:
        lf = ex['answer']
        
        intents_found = intent_pattern.findall(lf)
        all_intents.update(intents_found)
        slots_found = slot_pattern.findall(lf)
        all_slots.update(slots_found)

    sorted_intents = sorted(list(all_intents))
    sorted_slots = sorted(list(all_slots))

    intents_string = f"[{', '.join(sorted_intents)}]"
    slots_string = f"[{', '.join(sorted_slots)}]"

    return intents_string, slots_string