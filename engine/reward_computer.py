import torch.nn.functional as F
from typing import List, Dict,Optional
import logging

from engine.sampler import RolloutBuffer
from models.llm_wrapper import LLMWrapper
from models.embedding_model import EmbeddingModel
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

    def compute_rewards_and_advantages(self, 
                                            buffer: RolloutBuffer, 
                                            llm_wrapper: LLMWrapper, 
                                            ) -> RolloutBuffer: 

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
                
                per_sample_loss = llm_wrapper.get_batch_loss(prompts, targets)
                llm_reward = -per_sample_loss 
                
                buffer.final_llm_loss = per_sample_loss.to(self.device)
                
                buffer.rewards[:, 0] = llm_reward.to(self.device)
                
                buffer.returns[:, 0] = buffer.rewards[:, 0]
                
                buffer.advantages[:, 0] = buffer.returns[:, 0] - buffer.values[:, 0]
                
                return buffer
