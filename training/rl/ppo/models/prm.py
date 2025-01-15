import logging
import numpy as np
import torch
from torch import nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProcessRM(nn.Module):

    def __init__(self, model_name_or_path, lora_weights, model_max_length):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.lora_weights = lora_weights
        
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = "ки" 

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, 
            add_eos_token=False, 
            padding_side='left',
            model_max_length=model_max_length
            )
        self.tokenizer.pad_token_id = 151655 # "<|image_pad|>"
        self.candidate_tokens = self.tokenizer.encode(
            f" {self.good_token} {self.bad_token}"
            ) 
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1] 
        logging.info("Loading PRM...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            ).eval()
        self.model = PeftModel.from_pretrained(self.model, self.lora_weights)
        
    @torch.no_grad()
    def get_reward(self, obs: list[np.ndarray[str]], actions: list[np.ndarray[str]]):
        # Prepare text inputs for PRM
        inputs_for_prm = []
        for o, a in zip(obs.copy(), actions.copy()):
            inputs_for_prm.append(f"{o}{a} {self.step_tag}")
        
        # Prepare token inputs
        input_ids = self.tokenizer(
            inputs_for_prm, 
            return_tensors="pt", 
            padding=True
            ).to("cuda")
        
        # Get logits
        logits = self.model(**input_ids).logits[:, :, self.candidate_tokens]
        score = logits.softmax(dim=-1)[:, :, 0]  # get the probs for good_tokens
        
        step_scores = []
        for i in range(np.shape(score)[0]):
            step_score = score[i][input_ids["input_ids"][i] == self.step_tag_id]
            last_step_score = step_score[-1]  # NOTE interestingly, only last step score is taken as the reward
            step_scores.append([last_step_score.item()])
        step_scores = np.array(step_scores)
        
        return step_scores
