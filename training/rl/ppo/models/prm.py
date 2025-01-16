import logging
import numpy as np
import torch
from torch import nn
from peft import PeftModel
from utils.constants import Prompts, STEP_TAG
from transformers import AutoModelForCausalLM, AutoTokenizer



class ProcessRM(nn.Module):

    def __init__(self, model_name_or_path, lora_weights, model_max_length):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.lora_weights = lora_weights
        
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = STEP_TAG
        self.step_tag_real = "ки" 
        self.instruction = Prompts.INSTRUCTION_STEP_BY_STEP.value
        self.system_prompt = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value

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
        # NOTE openR implementation used to encode " ки" (with space), I changed it in my setting
        self.step_tag_real_id = self.tokenizer.encode(self.step_tag_real)[-1] 
        logging.info("Loading PRM...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            ).eval()
        self.model = PeftModel.from_pretrained(self.model, self.lora_weights)
        
    @torch.no_grad()
    def get_reward(
        self, 
        obs: list[np.ndarray[str]], 
        actions: list[np.ndarray[str]]  # actions are 100% ended with "\n\n" due to generation setting
        ):
        
        # Get input texts with templates
        inputs_for_prm = self.generate_input_texts(obs, actions)
        
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
            step_score = score[i][input_ids["input_ids"][i] == self.step_tag_real_id]
            last_step_score = step_score[-1]  # NOTE interestingly, only last step score is taken as the reward
            step_scores.append([last_step_score.item()])
        step_scores = np.array(step_scores)
        
        return step_scores
    
    def generate_input_texts(self, obs, actions):
        """Generate input texts from obs and actions.

        1. Split each obs string on all occurrences of "\nAnswer:\n".
        - Keep the first chunk as 'first_part'.
        - If there's exactly one additional chunk, use that directly as 'second_part'.
        - If there are multiple additional chunks, re-join them with "\nAnswer:\n".
        2. Re-append "\nAnswer:\n" to the 'first_part' to form the chat prompt.
        3. Apply the chat template to build the user prompt (questions).
        4. Take the corresponding actions element, append it to the 'second_part'.
        5. Replace self.step_tag with self.step_tag_real in the second_part if needed.
        6. Concatenate the templated question + second_part to get the final input text.
        7. Return a list of final input_text strings (length: n).

        Args:
            obs (np.ndarray): A numpy array of shape (n,) or (n,1) containing observation strings.
            actions (np.ndarray): A numpy array of shape (n,) or (n,1) containing action strings.

        Returns:
            list[str]: Final processed strings (length: n).
        """
        # Convert obs and actions to Python lists (if (n,1), they become (n,))
        obs_list = obs.flatten().tolist()
        actions_list = actions.flatten().tolist()

        # Split on *all* occurrences of "\nAnswer:\n"
        splitted_data = []
        for s in obs_list:
            splitted_parts = s.split("\nAnswer:\n")
            first_part = splitted_parts[0]

            # Decide how to form the second_part
            if len(splitted_parts) == 1:
                # No second part
                second_part = ""
            elif len(splitted_parts) == 2:
                # Exactly one second part
                second_part = splitted_parts[1]
            else:
                # Multiple second parts, join them back together
                second_part = "\nAnswer:\n".join(splitted_parts[1:])
            splitted_data.append((first_part, second_part))

        # Build messages by attaching "\nAnswer:\n" to the first_part only
        messages = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": first_part + "\nAnswer:\n"},
            ]
            for (first_part, _) in splitted_data
        ]

        # Apply the chat template to generate initial questions
        questions = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Combine each templated question with the updated second_part + actions
        input_texts = []
        for (first_part, second_part), question, action_str in zip(splitted_data, questions, actions_list):
            # Append the action to the second part
            second_part += action_str
            second_part = second_part.replace(self.step_tag, self.step_tag_real)
            input_text = question + second_part
            input_texts.append(input_text)

        return input_texts

