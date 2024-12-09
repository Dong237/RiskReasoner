import os
import torch
torch.cuda.empty_cache()

import logging
from typing import Literal
from tqdm import tqdm
import pandas as pd
from peft import PeftModel
# from verifier import Verifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.helper import jload, jdump
from utils.constants import Prompts, SPLIT_TOKEN

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value
INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value


class Generator:    
    def __init__(
        self, 
        model_name_or_path: str = "Qwen2.5-Math-7B-Instruct",
        verifier =  None, # Verifier =
        device: str = "cuda", 
        max_new_tokens: int = 2048,
        temperature: float = 0.7, 
        top_k: int = 50, 
        top_p: float = 1.0, 
        model_max_length: int = 2048,
        N: int = 1,        
        lora_weights: str = None
        ):

        self.model_name_or_path = model_name_or_path
        self.device = device
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_max_length = model_max_length
 
        self.model = None
        self.tokenizer = None
        self.lora_weights = lora_weights
        
        self.verifier = verifier
        self.N = N

    
    def __call__(self, data: Literal[dict, pd.Series]):
        # self._start_service()
        prompt = INSTRUCTION + data["query_cot"]
        choices = data["choices"] 
        gold_label = data["gold"]
        
        response_N = self._generate_for_single_question(prompt, choices, gold_label)
        data_tb_verified = {
            "prompt": prompt,
            "response_N": response_N,
        }
        
        # if self.verifier:
        #     response_best = self.verifier.verify(data_tb_verified)
        return data_tb_verified
    

    def start_service(self):
        logging.info("Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
            device_map="cuda"
        )
        if self.lora_weights:
            logging.info(f"Loading LoRA weights from {self.lora_weights}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_weights) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            model_max_length=self.model_max_length,
            truncation=True,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
            )
        logging.info("Model and tokenizer loaded successfully.")

    
    def _generate_response(self, model_inputs):

        # Generate responses using the model (batch inference)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )
        # Process the generated output and decode the results
        generated_ids = [
            output_ids[len(input_ids):]  # Slice the generation to remove the input part
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # Decode all generated outputs in batch
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses


    def _generate_probs(self, model_inputs, choices):
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            logits = outputs.logits

        good_token, bad_token = choices
        good_token_id = self.tokenizer.convert_tokens_to_ids(good_token)
        bad_token_id = self.tokenizer.convert_tokens_to_ids(bad_token)
        # Extract probabilities for "good" and "bad" from the logits (for each prompt in the batch)
        # NOTE this extraction is not clean since the last token is not always good or bad
        last_token_logits = logits[:, -1, :]  # Get logits for the last token in each sequence

        # Logit Masking for "good" and "bad"
        mask = torch.full_like(last_token_logits, float('-inf'))
        mask[:, good_token_id] = 0
        mask[:, bad_token_id] = 0
        masked_logits = last_token_logits + mask

        # Compute probabilities for each prompt
        probabilities = torch.softmax(masked_logits, dim=-1)
        good_probs = probabilities[:, good_token_id].cpu().numpy()
        bad_probs = probabilities[:, bad_token_id].cpu().numpy()
        return good_probs, bad_probs


    def _generate_for_single_question(self, prompt, choices, gold_label):
        results = []

        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},  # Common system message
                {"role": "user", "content": prompt}  # User's input prompt
            ]
            for _ in range(self.N)
        ]

        # Apply the chat template (assumes the tokenizer has an apply_chat_template method)
        prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True  # Add generation-specific prompt (like stop tokens, etc.)
        )
        
        model_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

        responses = self._generate_response(model_inputs)
        good_probs, bad_probs = self._generate_probs(model_inputs, choices)

        good_token, bad_token = choices
        for i, response in enumerate(responses):
            last_sentence = response.lower().split(SPLIT_TOKEN.lower())[-1]
            if good_token in last_sentence:
                pred_label = 0
            elif bad_token in last_sentence:
                pred_label = 1
            else:
                pred_label = "miss"
            results.append(
                {
                    "id": i,
                    "reasoning_steps": response,
                    "pred_label": pred_label,
                    "gold_label": gold_label,
                    "probs": [float(good_probs[i]), float(bad_probs[i])]
                    }
                )
        return results
        

if __name__ == "__main__":
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

    # Initialize the generator without the verifier part
    generator = Generator(
        model_name_or_path="/data/tangbo/plms/Qwen2.5-7B-Instruct/", 
        N=32,
        )
    generator.start_service()
    data_tb_verified_all = []
    for data_atom in tqdm(data, desc="Processing data..."):
        # Call the generator's __call__ method to test it
        data_tb_verified = generator(data_atom)
        data_tb_verified_all.append(data_tb_verified)
        
    jdump(data_tb_verified_all, "test.json")
    print("Final data saved")