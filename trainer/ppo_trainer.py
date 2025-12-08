import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.optim import Optimizer
from typing import Dict
import logging
import numpy as np

from utils import device
from models.policy_network import PolicyNetwork
from engine.sampler import RolloutBuffer
from config import train_config as config # 引入配置

logger = logging.getLogger(__name__)

class PPOTrainer:
    def __init__(self,
                 agent: PolicyNetwork,
                 optimizer: Optimizer,
                 ppo_epochs: int = 4,
                 ppo_clip_eps: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_bonus_coef: float = 0.01,
                 grad_clip_norm: float = 0.5,
                 mini_batch_size: int = 64): # [新增]

        self.agent = agent
        self.optimizer = optimizer
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_eps = ppo_clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_bonus_coef = entropy_bonus_coef
        self.grad_clip_norm = grad_clip_norm
        self.mini_batch_size = mini_batch_size

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        # 1. 准备数据
        # 此时 Buffer 中的数据已经是 Tensor 且在 device 上 (除了 queries/text)
        # 我们只需要数值型数据
        query_embeddings = buffer.query_embeddings.detach()
        old_actions = buffer.actions[:, 0].detach()
        old_log_probs = buffer.log_probs[:, 0].detach()
        old_values = buffer.values[:, 0].detach()
        returns = buffer.returns[:, 0].detach()
        advantages = buffer.advantages[:, 0].detach()

        # [重要] Advantage Normalization (降低方差的关键)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 统计容器
        epoch_stats = {
            "actor_loss": [], "value_loss": [], "entropy_loss": [], 
            "approx_kl": [], "clip_frac": [], "explained_var": []
        }

        # 计算 Explained Variance (衡量 Critic 好坏: 1 is perfect, <0 is bad)
        y_pred = old_values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        epoch_stats["explained_var"].append(explained_var)

        dataset_size = query_embeddings.size(0)
        indices = list(range(dataset_size))

        # 2. PPO Epochs 循环
        for _ in range(self.ppo_epochs):
            # [新增] Mini-batch 迭代
            # 使用 Random Sampler 打乱数据
            sampler = BatchSampler(SubsetRandomSampler(indices), self.mini_batch_size, drop_last=False)

            for batch_indices in sampler:
                # 取出 Mini-batch 数据
                mb_query_embs = query_embeddings[batch_indices]
                mb_actions = old_actions[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_old_values = old_values[batch_indices]
                mb_returns = returns[batch_indices]
                mb_advantages = advantages[batch_indices]

                # 前向传播 (Actor & Critic)
                # 注意：ResNet 结构下，get_logits_and_values 更高效
                if hasattr(self.agent, 'get_logits_and_values'):
                    logits, new_values = self.agent.get_logits_and_values(mb_query_embs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                else:
                    # 兼容旧接口
                    _, new_log_probs, new_values, entropy = self.agent(mb_query_embs, mb_actions)
                    entropy = entropy.mean()

                # --- 计算 Actor Loss ---
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 统计 Clip Fraction (有多少比例的样本被 Clip 了)
                clip_frac = (torch.abs(ratio - 1.0) > self.ppo_clip_eps).float().mean().item()

                # --- 计算 Value Loss (带 Clipping) ---
                # 限制 Value 更新幅度，防止 Critic 跑飞 (Standard PPO Trick)
                v_loss_unclipped = F.mse_loss(new_values, mb_returns)
                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values,
                    -self.ppo_clip_eps,
                    self.ppo_clip_eps,
                )
                v_loss_clipped = F.mse_loss(v_clipped, mb_returns)
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max

                # --- 总 Loss ---
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_bonus_coef * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                # --- 记录数据 ---
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                
                epoch_stats["actor_loss"].append(actor_loss.item())
                epoch_stats["value_loss"].append(value_loss.item())
                epoch_stats["entropy_loss"].append(entropy.item())
                epoch_stats["approx_kl"].append(approx_kl)
                epoch_stats["clip_frac"].append(clip_frac)

        # 返回平均统计值
        return {k: np.mean(v) for k, v in epoch_stats.items()}