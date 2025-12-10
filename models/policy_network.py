import torch
import torch.nn as nn
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_features(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        子类必须实现此方法，负责将 query_embeddings 映射为隐藏层特征
        """
        raise NotImplementedError

    def get_logits_and_values(self, query_embeddings: torch.Tensor):
        """
        统一接口：供预训练使用，直接获取 logits 和 values
        """
        features = self.extract_features(query_embeddings)
        features = self.dropout(features)
        logits = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def forward(self, query_embeddings: torch.Tensor, actions: torch.Tensor = None):
        """
        统一的 PPO 前向传播逻辑
        """
        logits, values = self.get_logits_and_values(query_embeddings)
        dist = Categorical(logits=logits)

        if actions is None:
            if self.training:
                actions = dist.sample()
            else:
                actions = torch.argmax(logits, dim=-1)
            
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return actions, log_probs, values, entropy

class LinearNetwork(PolicyNetwork):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        logger.info(f"Initializing PolicyNetwork (Adaptive Lambda MLP)")

        self.feature_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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

    def extract_features(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.feature_net(query_embeddings)

class RBFPolicyNetwork(PolicyNetwork):
    def __init__(self, embedding_dim: int, num_centers: int = 128, output_dim: int = 21, dropout: float = 0.0):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_centers = num_centers
        
        logger.info(f"Initializing RBFPolicyNetwork: Input={embedding_dim}, Centers={num_centers} (RBF Layer)")

        self.centers = nn.Parameter(torch.empty(num_centers, embedding_dim))
        self.log_sigmas = nn.Parameter(torch.zeros(num_centers))
        
        # 注意：这里 RBF 网络的 head 输入维度是 num_centers
        self.dropout = nn.Dropout(dropout)
        self.actor_head = nn.Linear(num_centers, output_dim)
        self.value_head = nn.Linear(num_centers, 1)
        
        self._init_weights()

    def _init_weights(self):
        # nn.init.normal_(self.centers, mean=0.0, std=1.0)
        nn.init.normal_(self.centers, mean=0.0, std=0.05)
        nn.init.constant_(self.log_sigmas, 0.0)
        nn.init.xavier_uniform_(self.actor_head.weight)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.constant_(self.value_head.bias, 0.0)

    def compute_rbf_features(self, x: torch.Tensor) -> torch.Tensor:
        x_norm_sq = x.pow(2).sum(dim=1, keepdim=True)       # [B, 1]
        c_norm_sq = self.centers.pow(2).sum(dim=1).unsqueeze(0)  # [1, K]
        dist_sq = x_norm_sq + c_norm_sq - 2 * torch.matmul(x, self.centers.t())
        dist_sq = torch.clamp(dist_sq, min=1e-6)
        clamped_log_sigmas = torch.clamp(self.log_sigmas, min=-1.0)
        sigmas = torch.exp(clamped_log_sigmas).unsqueeze(0) # [1, K]
        rbf_activations = torch.exp(-dist_sq / (2 * sigmas.pow(2) + 1e-8))
        return rbf_activations

    def extract_features(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.compute_rbf_features(query_embeddings)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))

class ResnetPolicyNetwork(PolicyNetwork):
    def __init__(
        self, 
        embedding_dim: int,     
        hidden_dim: int = 64,  
        dropout: float = 0.1,
        num_res_blocks: int = 2 
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        logger.info(f"Initializing Lightweight ResNet Policy: Input={embedding_dim}, Hidden={hidden_dim}, Layers={num_res_blocks}, Params=Small")

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout, expansion_factor=4) 
              for _ in range(num_res_blocks)]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, 21)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data, gain=1.0)
                else:
                    nn.init.normal_(param.data, mean=0.0, std=0.01)
            elif 'bias' in name:
                param.data.fill_(0)

    def extract_features(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(query_embeddings)
        x = self.res_blocks(x)
        return self.final_norm(x)