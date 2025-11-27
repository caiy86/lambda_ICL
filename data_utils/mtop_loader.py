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
        batch_size=config.BATCH_SIZE_VAL, 
        show_progress_bar=True
    )

    logger.info(f"Full train data and embeddings loaded ({len(full_train_data)} samples).")
    del emb_model
    torch.cuda.empty_cache()

    return full_train_data, full_train_embeddings

def _list_dict_collate_fn(batch: List[Dict]) -> List[Dict]:

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

class _MtopQueryDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

def get_corpus() -> Tuple[List[Dict[str, str]], torch.Tensor]:

    logger.info("--- Creating Corpus (Action Space) ---")
    
    full_train_data, full_train_embeddings = _get_full_train_data_and_embeddings()
    
    if not config.USE_CLUSTERED_CORPUS:
        logger.warning(
            f"USE_CLUSTERED_CORPUS=False. Using the *entire* training dataset "
            f"({len(full_train_data)} samples) as corpus."
        )
        corpus_data = full_train_data
        corpus_embeddings = full_train_embeddings
        
    else:
        corpus_size_config = config.CORPUS_NUM_CLUSTERS * config.CORPUS_SIZE_PER_CLUSTER
        logger.info(
            f"USE_CLUSTERED_CORPUS=True. Running K-Means (K={config.CORPUS_NUM_CLUSTERS}) "
            f"to sample {corpus_size_config} examples for corpus..."
        )
        
        kmeans = MiniBatchKMeans(
            n_clusters=config.CORPUS_NUM_CLUSTERS,
            random_state=config.SEED,
            batch_size=256,
            n_init="auto"
        )
        cluster_labels = kmeans.fit_predict(full_train_embeddings.cpu().numpy())
        
        corpus_indices = []
        rng = np.random.default_rng(config.SEED)
        
        for cluster_id in range(config.CORPUS_NUM_CLUSTERS):
            indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
            
            if len(indices_in_cluster) == 0:
                logger.warning(f"K-Means Cluster {cluster_id} is empty! Skipping.")
                continue
            
            num_to_sample = min(config.CORPUS_SIZE_PER_CLUSTER, len(indices_in_cluster))
            sampled_indices = rng.choice(indices_in_cluster, num_to_sample, replace=False)
            
            corpus_indices.extend(sampled_indices)

        corpus_data = [full_train_data[i] for i in corpus_indices]
        corpus_embeddings = full_train_embeddings[corpus_indices]
    
    logger.info(f"Corpus created. Final size: {len(corpus_data)}")
    return corpus_data, corpus_embeddings.to(utils.device)

def get_dataloader(split: str, 
                   batch_size: int, 
                   shuffle: bool = False,
                   nums: int = None) -> DataLoader:
    """
    Get dataloader for specified split, limited by nums if provided.
    """
    logger.info(f"Loading query dataloader (split={split}, batch_size={batch_size}, shuffle={shuffle}, nums={nums})...")

    data = _load_hf_data(split=split)

    if nums is not None and len(data) > nums:
        logger.info(f"Randomly sampling {nums} examples for '{split}' split...")
        rng = random.Random(config.SEED)
        query_data = rng.sample(data, nums)
    else:
        if nums is not None:
            logger.info(f"Requested nums ({nums}) is larger than dataset size ({len(data)}). Using full dataset.")
        query_data = data

    dataset = _MtopQueryDataset(query_data)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=False,
        collate_fn=_list_dict_collate_fn
    )

# def check_correct(target_answer: str, pred_text: str) -> bool:

#     return target_answer.strip().lower() == pred_text.strip().lower()

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