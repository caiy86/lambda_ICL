import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict
import logging

from utils import device
from models.policy_network import PolicyNetwork
from engine.sampler import RolloutBuffer

logger = logging.getLogger(__name__)

class PPOTrainer:

    def __init__(self,
                 agent: PolicyNetwork,
                 optimizer: Optimizer,
                 ppo_epochs: int = 4,
                 ppo_clip_eps: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_bonus_coef: float = 0.01,
                 grad_clip_norm: float = 2.0):

        self.agent = agent
        self.optimizer = optimizer
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_eps = ppo_clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_bonus_coef = entropy_bonus_coef
        self.grad_clip_norm = grad_clip_norm

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        
        query_embeddings = buffer.query_embeddings 
        old_actions = buffer.actions[:, 0]     # Lambda Index
        old_log_probs = buffer.log_probs[:, 0]
        advantages = buffer.advantages[:, 0]
        returns = buffer.returns[:, 0]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        all_actor_losses = []
        all_value_losses = []
        all_entropy_losses = []

        for epoch in range(self.ppo_epochs):
            
            _, new_log_probs, new_values, entropy = self.agent.forward(
                query_embeddings=query_embeddings,
                actions=old_actions 
            )
            

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + (value_loss * self.value_loss_coef) + (entropy_loss * self.entropy_bonus_coef)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
            self.optimizer.step()


        return { ... } 

 