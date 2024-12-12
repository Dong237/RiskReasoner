import os
import re
import torch
torch.cuda.empty_cache()
import logging
import itertools
from tqdm import tqdm
from peft import PeftModel
from typing import Literal, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.helper import jdump
from utils.constants import (
    Prompts, 
    STEP_TAG, 
    SPLIT_TOKEN, 
    SEARCH_PATTERN
)


SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value
INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value


class Generator:    
    def __init__(
        self, 
        model_name_or_path: str = "Qwen2.5-Math-7B-Instruct",
        max_new_tokens: int = 2048,
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 1.0, 
        model_max_length: int = 2048,
        batch_size: int = 1,        
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
        self.batch_size = batch_size 
    
    def __call__(self, data_all: List[dict]):
        return self.generate(data_all)
    
    
    def generate(self, data_all: List[dict]):
        self._start_service()
        results = []
        for start_idx in tqdm(range(0, len(data_all), self.batch_size), desc="Processing batches"):
            end_idx = min(start_idx + self.batch_size, len(data_all))
            batch = data_all[start_idx:end_idx]
            real_batch_size = end_idx - start_idx  # the last batch might be smaller than BATCH_SIZE

            # Prepare inputs for the batch
            prompts = [INSTRUCTION + item["query_cot"] for item in batch]
            choices_list = [item["choices"] for item in batch]
            gold_labels = [item["gold"] for item in batch]
            record_ids = [item["id"] for item in batch]

            # Make predictions for the batch
            batch_results = self._batch_predict( 
                prompts, 
                choices_list, 
                record_ids, 
                gold_labels,
                real_batch_size,
                )

            # Append batch results to overall results
            results.extend(batch_results)
        return results
    
    def _batch_predict(
        self,
        prompts, 
        choices_list, 
        record_ids, 
        gold_labels, 
        real_batch_size
        ):
        
        """
        Predict for a batch of data and return the results.
        """
        # Prepare the batch prompts
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
                ] 
            for prompt in prompts
        ]

        # Convert prompts to text using the tokenizer's chat template
        texts = self.tokenizer.apply_chat_template(
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
            ).to(self.model.device)

        # Configure generation settings for deterministic decoding
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        # Generate responses for the batch
        with torch.no_grad():
            generated_dict = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=2048,
                return_dict_in_generate=True,
                output_logits=True,
                output_scores=True,
            )

        # Process each result in the batch
        results = []
        for idx in range(real_batch_size):
            # Make prediction
            good_prob, bad_prob, pred_label = self._predict_token_and_probs(
                model_inputs, 
                generated_dict, 
                choices_list, 
                idx
                )
            
            # Prepare the result dictionary
            result = {
                "id": record_ids[idx],
                "pred_prob": [good_prob, bad_prob],
                "pred_label": pred_label,
                "label": gold_labels[idx],
                "query": prompts[idx],
            }
            results.append(result)

        return results
    
    def _predict_token_and_probs(
        self,
        model_inputs, 
        generated_dict, 
        choices_list, 
        idx
        ):
        
        #######################
        ##  Text Prediction  ##
        #######################
        # Extract the generated text for the current prompt
        generated_ids = generated_dict.sequences[idx]
        generated_ids = generated_ids[model_inputs["input_ids"][idx].size()[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        text_prediction = response.lower().split(SPLIT_TOKEN.lower())[-1].replace(":", "").strip()
        good_token, bad_token = choices_list[idx]
        if good_token in text_prediction:
            pred_label = 0
        elif bad_token in text_prediction:
            pred_label = 1
        else:
            pred_label = "miss"

        #######################
        ##  Prob Prediction  ##
        #######################
        # Search for the prediction in similar form of "final assessment: xxx"
        match = re.search(SEARCH_PATTERN, response, re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            # To clean up the matched text, this is vital!
            matched_text = matched_text.replace(STEP_TAG, "").replace("\n", "").replace("*", "")
            pred_token = matched_text.split(":")[-1]
            
            good_token_id, bad_token_id = self._get_tokens_id(
                self.tokenizer, good_token, bad_token, pred_token
                )
            
            if not good_token_id or not bad_token_id:
                # Make a random guess if output has no valid prediction
                good_prob, bad_prob = 0.5, 0.5
            else:
                # Now start to get the logits for the pred_token 
                matched_generated_ids = self.tokenizer(
                    matched_text, return_tensors="pt"
                    )["input_ids"].to(self.model.device)
                # FIXME during index matching below, I skip over the first token since its variant is quite complicated
                # and unstable. But luckily, there are always more than 2 matched tokens in the list so this should be fine.
                indices = self._find_continuous_indices(matched_generated_ids[0][1:], generated_ids)
                # Note that in normal cases, indices[-1] is not the last index of generated_ids
                # because '<|im_end|>' should be the last token generated.
                target_logits = generated_dict.logits[indices[-1]][idx]
                
                mask = torch.full_like(target_logits, float('-inf'))
                mask[good_token_id], mask[bad_token_id] = 0, 0
                
                masked_logits = target_logits + mask

                # Compute probabilities
                probabilities = torch.softmax(masked_logits, dim=-1)
                good_prob = probabilities[good_token_id].item()
                bad_prob = probabilities[bad_token_id].item()
        else:
            # make a random guess if regex failed / output has no prediction
            good_prob, bad_prob = 0.5, 0.5
            
        return good_prob, bad_prob, pred_label
    
    
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


    def _get_tokens_id(self, tokenizer, good_token, bad_token, pred_token):
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
    def _find_continuous_indices(tensor_pattern, tensor_sequence):
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
    
    
    def _start_service(self):
        """
        Load the model and tokenizer from the specified path, optionally loading LoRA weights.
        """
        logging.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
            device_map="cuda"
        )
        
        if self.lora_weights:
            logging.info(f"Loading LoRA weights from {self.lora_weights}")
            model = PeftModel.from_pretrained(model, self.lora_weights) 
            
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            model_max_length=self.model_max_length,
            truncation=True,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
            )
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
  


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # Create a sample data dictionary
    data = [
    {
        "id":2065,
        "query":"Assess the client's loan status based on the following loan records from Lending Club. Respond with only 'good' or 'bad', and do not provide any additional information. For instance, 'The client has a stable income, no previous debts, and owns a property.' should be classified as 'good'. \nText: The Installment is 285.05. The Loan Purpose is credit_card. The Loan Application Type is Individual. The Interest Rate is 17.86%. The Last Payment Amount is 285.05. The Loan Amount is 7900.0. The Revolving Balance is 10878.0. The Delinquency In 2 years is 1.0. The Inquiries In 6 Months is 0.0. The Mortgage Accounts is 0.0. The Grade is D. The Open Accounts is 10.0. The Revolving Utilization Rate is 74.00%. The Total Accounts is 11.0. The Fico Range Low is 660.0. The Fico Range High is 664.0. The Address State is WA. The Employment Length is 3 years. The Home Ownership is RENT. The Verification Status is Verified. The Annual Income is 22000.0. As reference, the predicted probability of this client's loan status being good given by the machine learning model LightGBM is 48.21%, and the probability of it being bad is 51.79%. \nAnswer:",
        "query_cot":"Text: The Installment is 285.05. The Loan Purpose is credit_card. The Loan Application Type is Individual. The Interest Rate is 17.86%. The Last Payment Amount is 285.05. The Loan Amount is 7900.0. The Revolving Balance is 10878.0. The Delinquency In 2 years is 1.0. The Inquiries In 6 Months is 0.0. The Mortgage Accounts is 0.0. The Grade is D. The Open Accounts is 10.0. The Revolving Utilization Rate is 74.00%. The Total Accounts is 11.0. The Fico Range Low is 660.0. The Fico Range High is 664.0. The Address State is WA. The Employment Length is 3 years. The Home Ownership is RENT. The Verification Status is Verified. The Annual Income is 22000.0. As reference, the predicted probability of this client's loan status being good given by the machine learning model LightGBM is 48.21%, and the probability of it being bad is 51.79%. \nAnswer:",
        "answer":"bad",
        "choices":[
            "good",
            "bad"
        ],
        "gold":1,
        "text":"The Installment is 285.05. The Loan Purpose is credit_card. The Loan Application Type is Individual. The Interest Rate is 17.86%. The Last Payment Amount is 285.05. The Loan Amount is 7900.0. The Revolving Balance is 10878.0. The Delinquency In 2 years is 1.0. The Inquiries In 6 Months is 0.0. The Mortgage Accounts is 0.0. The Grade is D. The Open Accounts is 10.0. The Revolving Utilization Rate is 74.00%. The Total Accounts is 11.0. The Fico Range Low is 660.0. The Fico Range High is 664.0. The Address State is WA. The Employment Length is 3 years. The Home Ownership is RENT. The Verification Status is Verified. The Annual Income is 22000.0. "
    },
        {
        "id":290,
        "query":"Assess the client's loan status based on the following loan records from Lending Club. Respond with only 'good' or 'bad', and do not provide any additional information. For instance, 'The client has a stable income, no previous debts, and owns a property.' should be classified as 'good'. \nText: The Installment is 583.51. The Loan Purpose is debt_consolidation. The Loan Application Type is Individual. The Interest Rate is 15.99%. The Last Payment Amount is 20000.95. The Loan Amount is 24000.0. The Revolving Balance is 21435.0. The Delinquency In 2 years is 0.0. The Inquiries In 6 Months is 0.0. The Mortgage Accounts is 0.0. The Grade is C. The Open Accounts is 9.0. The Revolving Utilization Rate is 75.50%. The Total Accounts is 16.0. The Fico Range Low is 675.0. The Fico Range High is 679.0. The Address State is CA. The Employment Length is 2 years. The Home Ownership is RENT. The Verification Status is Verified. The Annual Income is 75000.0. As reference, the predicted probability of this client's loan status being good given by the machine learning model LightGBM is 99.82%, and the probability of it being bad is 0.18%. \nAnswer:",
        "query_cot":"Text: The Installment is 583.51. The Loan Purpose is debt_consolidation. The Loan Application Type is Individual. The Interest Rate is 15.99%. The Last Payment Amount is 20000.95. The Loan Amount is 24000.0. The Revolving Balance is 21435.0. The Delinquency In 2 years is 0.0. The Inquiries In 6 Months is 0.0. The Mortgage Accounts is 0.0. The Grade is C. The Open Accounts is 9.0. The Revolving Utilization Rate is 75.50%. The Total Accounts is 16.0. The Fico Range Low is 675.0. The Fico Range High is 679.0. The Address State is CA. The Employment Length is 2 years. The Home Ownership is RENT. The Verification Status is Verified. The Annual Income is 75000.0. As reference, the predicted probability of this client's loan status being good given by the machine learning model LightGBM is 99.82%, and the probability of it being bad is 0.18%. \nAnswer:",
        "answer":"good",
        "choices":[
            "good",
            "bad"
        ],
        "gold":0,
        "text":"The Installment is 583.51. The Loan Purpose is debt_consolidation. The Loan Application Type is Individual. The Interest Rate is 15.99%. The Last Payment Amount is 20000.95. The Loan Amount is 24000.0. The Revolving Balance is 21435.0. The Delinquency In 2 years is 0.0. The Inquiries In 6 Months is 0.0. The Mortgage Accounts is 0.0. The Grade is C. The Open Accounts is 9.0. The Revolving Utilization Rate is 75.50%. The Total Accounts is 16.0. The Fico Range Low is 675.0. The Fico Range High is 679.0. The Address State is CA. The Employment Length is 2 years. The Home Ownership is RENT. The Verification Status is Verified. The Annual Income is 75000.0. "
    }
    ]
    
    generator = Generator(
        model_name_or_path="/data/youxiang/huggingface/Qwen2.5-7B-Instruct",
        batch_size=16,
    )
    
    results = generator(data)
    jdump(results, "result_cot.json")