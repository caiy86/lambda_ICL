import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import logging

from utils import device

logger = logging.getLogger(__name__)

class LLMWrapper:

    def __init__(self, model_name: str = 'Qwen/Qwen3-0.6B'):

        logger.info(f"Loading LLM environment: {model_name}...")
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto", 
            trust_remote_code=True 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        logger.info(f"LLM '{model_name}' loaded successfully.")

    def build_chat_prompt(self, 
                          system_prompt: str, 
                          examples: List[Dict[str, str]], 
                          query: str) -> str:

        messages = [{"role": "system", "content": system_prompt}]
        for ex in examples:
            messages.extend([
                {"role": "user", "content": ex['query']},
                {"role": "assistant", "content": ex['answer']}
            ])
        messages.append({"role": "user", "content": query})

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False
        )   
        return prompt_str

    @torch.no_grad()
    def get_batch_loss(self, prompts: List[str], targets: List[str]) -> torch.Tensor:
        
        self.tokenizer.padding_side = 'right'

        full_texts = [p + t + self.tokenizer.eos_token for p, t in zip(prompts, targets)] 
               
        inputs = self.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(self.device)
        
        labels = inputs.input_ids.clone()
        
        prompt_token_lengths = [
            len(self.tokenizer.encode(p)) for p in prompts
        ]

        for i in range(len(prompts)):
            labels[i, :prompt_token_lengths[i]] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )

        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        loss = loss.view(shift_labels.shape)
        
        loss_mask = (shift_labels != -100)
        per_sample_loss = loss.sum(dim=1) / loss_mask.sum(dim=1)
        per_sample_loss = per_sample_loss.nan_to_num(float('inf'))
        
        self.tokenizer.padding_side = 'left'
        
        return per_sample_loss.detach().cpu()

    @torch.no_grad()
    def generate_for_evaluation(self, 
                                prompts: List[str], 
                                max_new_tokens: int = 150) -> tuple[List[str], torch.Tensor]:
        
        self.tokenizer.padding_side = 'left'
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False, 
            
            return_dict_in_generate=True,
            output_scores=True 
        )

        input_token_lengths = inputs.input_ids.shape[1]
        
        scores = torch.stack(outputs.scores, dim=1) 
        generated_ids = outputs.sequences[:, input_token_lengths:] 

        log_softmax_scores = F.log_softmax(scores, dim=-1)

        chosen_token_log_probs = log_softmax_scores.gather(
            dim=-1, 
            index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        
        padding_mask = (generated_ids != self.tokenizer.pad_token_id)
        
        chosen_token_log_probs = chosen_token_log_probs * padding_mask

        sum_log_probs = chosen_token_log_probs.sum(dim=1)
        num_non_pad_tokens = padding_mask.sum(dim=1)
        
        num_non_pad_tokens = torch.clamp(num_non_pad_tokens, min=1e-9)
        
        avg_log_probs = sum_log_probs / num_non_pad_tokens
        nlls = -avg_log_probs 
        
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )

        return generated_texts, nlls