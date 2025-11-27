import torch
from sentence_transformers import SentenceTransformer
from typing import List
import logging

from utils import device

logger = logging.getLogger(__name__)

class EmbeddingModel:

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):

        logging.info(f"[EmbeddingModel] Loading embedding model: {model_name}...")
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logging.info(f"[EmbeddingModel] Model loaded. Embedding dimension: {self.embedding_dim}")

    @property
    def dim(self) -> int:
        return self.embedding_dim

    @torch.no_grad()
    def encode(self, 
                 texts: List[str], 
                 batch_size: int = 32, 
                 show_progress_bar: bool = False,
                 normalize_embeddings: bool = True) -> torch.Tensor:

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_tensor=True
        )
        
        return embeddings