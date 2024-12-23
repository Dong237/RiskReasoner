import torch
import logging
import itertools
from peft import PeftModel
from typing import Literal, Optional, List
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig
)
from utils.helper import (
    jload,
    jdump,
    setup_logging
)
from utils.constants import (
    Prompts,
    STEP_TAG, 
    SPLIT_TOKEN, 
    SEARCH_PATTERN
)


class BaseGenerator:
    def __init__(
        self,
        model_name_or_path: str = "Qwen2.5-Math-7B-Instruct",
        max_new_tokens: int = 2048,
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 1.0, 
        model_max_length: int = 2048,    
        lora_weights: Optional[str] = None,
        ):

        self.model_name_or_path = model_name_or_path
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_max_length = model_max_length
 
        self.model = None
        self.tokenizer = None
        self.lora_weights = lora_weights
        
        self.system_prompt = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value
        self.instruction = Prompts.INSTRUCTION_STEP_BY_STEP.value
        
        self.step_tag = STEP_TAG
        self.split_token = SPLIT_TOKEN
        self.search_pattern = SEARCH_PATTERN
        
        setup_logging()
    
    def __call__(self, data_all: List[dict]):
        return self.generate(data_all)

    def generate(self, *args, **kwargs):
        """Takes in the whole json dataset and performs the experiment"""
        raise NotImplementedError
    
    ### Making prediction for a single input
    def predict_token_and_probs(self, *args, **kwargs):
        """Predict the classification token and probabilities of the two tokens by processing logits"""
        raise NotImplementedError
    
    @staticmethod
    def _get_variation(word: Literal["good", "bad"]):
        """
        Generate all variations of the given word with different cases 
        (lowercase, capitalized, uppercase) and combinations of leading 
        and trailing spaces.
        """
        # Define variations
        cases = [word.lower(), word.capitalize(), word.upper()]  # e.g., "good", "Good", "GOOD"
        spaces = ["", " ", "  "]  # No space, one space, two spaces

        # Generate all combinations with leading and trailing spaces
        variations = []
        for leading_space, case, trailing_space in itertools.product(spaces, cases, spaces):
            variations.append(f"{leading_space}{case}{trailing_space}")
        return variations

    def get_tokens_id(self, tokenizer, good_token, bad_token, pred_token):
        """
        Given the two tokens from ``choices`` and the predicted token,
        get the corresponding token ids. The ids of the two binary tokens
        are used later for performing masking and retrieving the probabilities.
        """
        good_tokens, bad_tokens = self._get_variation(good_token), self._get_variation(bad_token)
        good_tokens_id = [tokenizer(token).input_ids[0] for token in good_tokens]
        bad_tokens_id = [tokenizer(token).input_ids[0] for token in bad_tokens]
        if pred_token in good_tokens:
            idx = good_tokens.index(pred_token)
        elif pred_token in bad_tokens:
            idx = bad_tokens.index(pred_token)
        else:
            return None, None
        good_token_id, bad_token_id = good_tokens_id[idx], bad_tokens_id[idx]
        return good_token_id, bad_token_id

    @staticmethod
    def find_continuous_indices(tensor_pattern, tensor_sequence):
        """
        Finds the starting indices of a continuous subsequence (tensor_pattern) 
        in a given sequence (tensor_sequence).
        """
        pattern_length = len(tensor_pattern)

        # Create sliding windows of size equal to the pattern length over the sequence
        sliding_windows = tensor_sequence.unfold(0, pattern_length, 1)  
        
        # Compare each window with the pattern
        is_match = (sliding_windows == tensor_pattern).all(dim=1)  # Check for full matches
        
        # Find the first match
        match_indices = torch.where(is_match)[0]
        if len(match_indices) > 0:
            start_index = match_indices[0].item()  # Convert to Python int
            return list(range(start_index, start_index + pattern_length))
        return []
    
    
    ### Loading Model and Tokenizer
    def start_service(self):
        self.model = self.load_model_to_device(device=None)
        self.tokenizer = self.load_tokenizer()
        logging.info("Model and tokenizer loaded successfully.")

    def load_model_to_device(self, device: int):
        logging.info("Loading model...")
        if device:
            device_map = f"cuda:{device}"
        else:
            device_map = "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
            device_map=device_map
        )
        if self.lora_weights:
            logging.info(f"Loading LoRA weights from {self.lora_weights}")
            model = PeftModel.from_pretrained(model, self.lora_weights) 
        return model
    
    def load_tokenizer(self):
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            model_max_length=self.model_max_length,
            truncation=True,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
            )
        return tokenizer
    
    
    ### Batch Generation
    def generate_for_batch(
        self, 
        prompt_batch: List[str], 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        strategy: Literal["greedy", "sampling"],
        return_inputs: bool = True
        ):
        model_inputs = self.get_batch_model_inputs(prompt_batch, model, tokenizer)
        with torch.no_grad():
            generated_dict = self.model.generate(
                **model_inputs,
                generation_config=self.get_generation_config(
                    strategy=strategy
                    ),
                return_dict_in_generate=True,
                output_logits=True,
                output_scores=True,
            )
        if return_inputs:
            return (model_inputs, generated_dict)
        else:
            return generated_dict
    
    def get_batch_model_inputs(self, prompt_batch, model, tokenizer):
        # Prepare the batch prompts
        messages = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
                ] 
            for prompt in prompt_batch
        ]

        # Convert prompts to text using the tokenizer's chat template
        texts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize the inputs as a batch 
        # and pad to the max length in this batch
        model_inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
            ).to(model.device)
        return model_inputs
    
    def get_generation_config(self, strategy=Literal["greedy", "sampling"]):
        if strategy == "greedy":
            generation_config = GenerationConfig(
                temperature=None,     
                top_p=None,           
                top_k=None,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,      
            )
        elif strategy == "sampling":
            generation_config = GenerationConfig(
                temperature=self.temperature,     
                top_p=self.top_p,           
                top_k=self.top_k,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,      
            )
        else:
            raise ValueError(f"Invalid generation strategy: {strategy}")
        return generation_config
    

    ## Save and Load data
    def save(self, data: List[dict], path: str):
        jdump(data, path)
        logging.info(f"Data saved to {path}")

    def load(self, path: str):
        data = jload(path)
        return data