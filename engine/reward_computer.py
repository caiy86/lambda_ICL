import logging
from typing import Callable
import torch

from engine.sampler import RolloutBuffer
from models.llm_wrapper import LLMWrapper
from config import train_config as config
from utils import device

logger = logging.getLogger(__name__)

class RewardComputer:

    def __init__(self,
                 gamma: float, 
                 lambda_: float,
                 system_prompt: str = "You are a helpful assistant.",
                 prompt_strategy: str = "multi_turn"):

        self.gamma = gamma
        self.lambda_ = lambda_
        self.system_prompt = system_prompt
        self.device = device
        
        logger.info("RewardComputer initialized with GAE (gamma=%.2f, lambda=%.2f)", gamma, lambda_)

    # def compute_rewards_and_advantages(self, 
    #                                         buffer: RolloutBuffer, 
    #                                         llm_wrapper: LLMWrapper, 
    #                                         ) -> RolloutBuffer: 

    #             batch_size = buffer.batch_size
                
    #             prompts = []
    #             targets = []
    #             for i in range(batch_size):
    #                 query_data = buffer.queries[i]
    #                 example_data = buffer.selected_examples_text[i]
    #                 prompt_str = llm_wrapper.build_chat_prompt(
    #                     system_prompt=self.system_prompt,
    #                     examples=example_data,
    #                     query=query_data['query'],
    #                 )
    #                 prompts.append(prompt_str)
    #                 targets.append(query_data['answer'])
                
    #             per_sample_loss = llm_wrapper.get_batch_loss(prompts, targets)
    #             llm_reward = -per_sample_loss 
                
    #             buffer.final_llm_loss = per_sample_loss.to(self.device)
                
    #             buffer.rewards[:, 0] = llm_reward.to(self.device)
                
    #             buffer.returns[:, 0] = buffer.rewards[:, 0]
                
    #             buffer.advantages[:, 0] = buffer.returns[:, 0] - buffer.values[:, 0]
                
    #             return buffer

    def compute_rewards_and_advantages(self, 
                                        buffer: RolloutBuffer, 
                                        llm_wrapper: LLMWrapper, 
                                        check_correct_fn: Callable) -> RolloutBuffer: 
            batch_size = buffer.batch_size
            prompts = []
            targets = []
            
            for i in range(batch_size):
                query_data = buffer.queries[i]
                example_data = buffer.selected_examples_text[i]
                prompt_str = llm_wrapper.build_chat_prompt(
                    system_prompt=self.system_prompt,
                    examples=example_data,
                    query=query_data['query'],
                )
                prompts.append(prompt_str)
                targets.append(query_data['answer'])
                
            target_losses = llm_wrapper.get_batch_loss(prompts, targets)
            
            generated_texts, _ = llm_wrapper.generate_for_evaluation(
                prompts, 
                max_new_tokens=config.MAX_GEN_TOKENS
            )
            
            rewards = []
            for i in range(batch_size):
                is_correct = check_correct_fn(target_answer=targets[i], pred_text=generated_texts[i])
                r_metric = 1.0 if is_correct else -1.0
                nll_val = target_losses[i].item()
                prob_val = torch.exp(torch.tensor(-nll_val)).item()
                r_loss = 2.0 * prob_val - 1.0

                total_r = config.METRIC_WEIGHT * r_metric +  config.LOSS_WEIGHT * r_loss
                # total_r = r_metric * (1.0 + config.SCALE_FACTOR * r_loss)
                rewards.append(total_r)
      
            buffer.final_llm_loss = target_losses.to(self.device)
            buffer.rewards[:, 0] = torch.tensor(rewards, device=self.device)
            
            buffer.returns[:, 0] = buffer.rewards[:, 0]
            buffer.advantages[:, 0] = buffer.returns[:, 0] - buffer.values[:, 0]
            
            return buffer