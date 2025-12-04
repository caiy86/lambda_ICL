import torch
import torch.nn as nn
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        logger.info(f"Initializing PolicyNetwork (Adaptive Lambda MLP)")

        self.feature_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        
        self.actor_head = nn.Linear(hidden_dim, 21)
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data, gain=1.0)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, query_embeddings: torch.Tensor, actions: torch.Tensor = None):

        # (B, D) -> (B, H)
        features = self.feature_net(query_embeddings)
        
        # (B, H) -> (B, 21)
        logits = self.actor_head(features)
        # (B, H) -> (B)
        values = self.value_head(features).squeeze(-1)

        dist = Categorical(logits=logits)

        if actions is None:
            if self.training:
                actions = dist.sample()
            else:
                actions = torch.argmax(logits, dim=-1)
            
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return actions, log_probs, values, entropy