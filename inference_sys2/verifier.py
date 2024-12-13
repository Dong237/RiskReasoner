import os
import torch
torch.cuda.empty_cache()
import logging
from tqdm import tqdm
from typing import List
from functools import reduce
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import Prompts, SPLIT_TOKEN, STEP_TAG
from utils.helper import jload, jdump, compute_binary_metrics_from_results, setup_logging

SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value
INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value

STEP_TAG_REAL = "ки"
GOOD_TOKEN, BAD_TOKEN = '+', '-'


class Verifier:
    
    def __init__(
        self,         
        model_name_or_path: str = "Qwen2.5-Math-7B-Instruct",
        lora_weights: str = None,
        model_max_length: int = 4096, # 2048 is too small according to experience with Qwen2.5-7B
        batch_sze: int = 8,
        ):
        
        self.model_name_or_path = model_name_or_path
        self.lora_weights = lora_weights
        
        self.model = None
        self.tokenizer = None
        
        self.model_max_length = model_max_length
        self.batch_sze = batch_sze
    
    
    @staticmethod
    def _get_solution_level_score(step_scores: List[list], method="min"):
        good_probs = [probs[0] for probs in step_scores]
        if method == "min":
            return min(good_probs)
        elif method == "product":
            return reduce(lambda x, y: x * y, good_probs)
        else:
            raise ValueError("Invalid method provided for calculating the solution level score")
    
    def _prepare_model_inputs(
        self, 
        prompt, 
        response_N
        ):
        
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},  
                {"role": "user", "content": prompt},
            ]
            for _ in range(len(response_N))
        ]
        
        questions = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # We handle tokenization ourselves
            add_generation_prompt=True  # Add generation-specific prompt (like stop tokens, etc.)
        )
        
        input_text_batch = []
        for question, response in zip(questions, response_N):
            response["reasoning_steps"] = response["reasoning_steps"] + STEP_TAG
            # NOTE replace STEP_TAG with STEP_TAG_REAL to align with the training process
            input_text = question + response["reasoning_steps"].replace(STEP_TAG, STEP_TAG_REAL)
            input_text_batch.append(input_text)
            
        tokenized_inputs = self.tokenizer(
            input_text_batch, 
            return_tensors="pt",
            truncation=True, 
            padding='max_length',
            max_length=self.model_max_length
            ).to(self.model.device)
        
        return tokenized_inputs
        
        
    def get_verifier_scores(
        self, 
        prompt, 
        batch_response, 
        step_tag_real_id,
        good_token_id,
        bad_token_id,
        ):
        """
        Process a specified batch of responses, store the predicted probs of each step
        and the solution-level score in the response dict.
        """
        model_inputs = self._prepare_model_inputs(prompt, batch_response)
        indices_batch, indices_vocab = torch.where(model_inputs['input_ids']==step_tag_real_id)
        
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            logits = outputs.logits
        
        mask = torch.full_like(logits, float('-inf')) # .to(self.model.device)  # -inf for masking
        # Set mask to 0 at good_token_id and bad_token_id positions for each step tag
        mask[:, :, good_token_id] = 0
        mask[:, :, bad_token_id] = 0
        masked_logits = logits + mask
        step_tag_logits = masked_logits[indices_batch, indices_vocab, :]
        probabilities = torch.softmax(step_tag_logits, dim=-1)
        
        for batch_idx in range(model_inputs['input_ids'].size(0)):  # Iterating over batch_size
            # Get the indices for this batch element
            batch_indices = (indices_batch == batch_idx)
            step_probs = probabilities[batch_indices]  
            step_tag_probs = []
            for step_prob in step_probs:
                # NOTE remember that step_probs_for_tag[0] is the probability of good_token_id
                step_probs_for_tag = [
                    step_prob[good_token_id].item(),
                    step_prob[bad_token_id].item()
                ]
                step_tag_probs.append(step_probs_for_tag)
            
            batch_response[batch_idx]["step_tag_probs"] = step_tag_probs
            batch_response[batch_idx]["solution_level_score"] = self._get_solution_level_score(step_tag_probs)
    
    
    def _get_best_response(self, response_N):
        resposne_best = max(response_N, key=lambda x: x['solution_level_score'])
        return resposne_best
    
    
    def verify(self, data, save_N=True):
        
        step_tag_real_id = self.tokenizer.encode(f"{STEP_TAG_REAL}")[-1]
        good_token_id = self.tokenizer.encode(f"{GOOD_TOKEN}")[-1]
        bad_token_id = self.tokenizer.encode(f"{BAD_TOKEN}")[-1]
        
        num_responses = len(data["response_N"])
        real_batch_size = min(num_responses, self.batch_sze)
        for start_idx in range(0, num_responses, real_batch_size):
            # Get the current batch
            end_idx = min(start_idx + real_batch_size, num_responses)
            batch_responses = data["response_N"][start_idx:end_idx]
            self.get_verifier_scores(
                data["prompt"], 
                batch_responses, 
                step_tag_real_id,
                good_token_id,
                bad_token_id,
                )
        
        if save_N:
            data_scored = {
                "prompt": data["prompt"],
                "response_N": data["response_N"]
            }
        else:
            data_scored = None
        
        response_best = self._get_best_response(data["response_N"])
        # return the dict with keys adapted for further metrics calculation
        response_best_keys_adapted = {
            "id": response_best["id"],
            "prompt": data["prompt"],
            "reasoning_steps": response_best["reasoning_steps"],
            "pred_label": response_best["pred_label"],
            "pred_prob": [float(prob) for prob in response_best["probs"]],  # get the prob for "good"
            "label": response_best["gold_label"],
            "solution_level_score": response_best["solution_level_score"],
            
        }
        return response_best_keys_adapted, data_scored
    
    
    def start_verifier(self):
        logging.info("Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
            device_map="cuda",
            low_cpu_mem_usage=True
        )
        self.model.eval()  
        if self.lora_weights:
            logging.info(f"Loading LoRA weights from {self.lora_weights}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.lora_weights,
                freeze_base_model=True
                ) 
        # Use greedy decoding for verify to keep its consistency
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            model_max_length=self.model_max_length,
            # padding_side="left",
            use_fast=False,
            trust_remote_code=True,
            )
        self.tokenizer.pad_token_id = 0
        logging.info("Model and tokenizer loaded successfully.")


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    setup_logging()
    # model_name_or_path = "/data/youxiang/huggingface/Qwen2.5-7B-Instruct"
    # lora_weights = "/data/youxiang/repos/RiskReasoner/models/RiskPRM"
    
    model_name_or_path = "models/RiskPRM_v2_full"
    
    verifier = Verifier(model_name_or_path, lora_weights=None)
    verifier.start_verifier()
    
    data = {
        "prompt": 'You are given a text describing a customer\'s credit report and predicted loan status (probabilities) from a trained machine learning system as reference (and reference only).\n\nPlease analyze the provided information STEP BY STEP following these instructions:\n1. **During reasoning process**: For each piece of data, evaluate its significance to the customer\'s creditworthiness. Then use your prior knowledge to interpret the data and consider how each factor impacts the final credit risk assessment.\n2. **When presenting your reasoning steps**:\n    - Analyse each attribute from the report, discuss its importance, and determine whether it positively or negatively affects the loan status.\n    - Combine all factors to form a holistic assessment, taking into account the prediction probabilities provided by the machine learning model.\n3. **In the end of your output**: after presenting the reasoning steps, give your final answer in the following format: \'Final assessment: [choose from "good" or "bad"]\'. End you output with one of the assessment token, this means that your last output token can only be either "good" or "bad", not any other texts or symbols. For example: don\'t use **Final assessment: good**, instead just say Final assessment: good (without \'**\').\n\nHere is the customer\'s credit report:\nText: The Installment is 583.51. The Loan Purpose is debt_consolidation. The Loan Application Type is Individual. The Interest Rate is 15.99%. The Last Payment Amount is 20000.95. The Loan Amount is 24000.0. The Revolving Balance is 21435.0. The Delinquency In 2 years is 0.0. The Inquiries In 6 Months is 0.0. The Mortgage Accounts is 0.0. The Grade is C. The Open Accounts is 9.0. The Revolving Utilization Rate is 75.50%. The Total Accounts is 16.0. The Fico Range Low is 675.0. The Fico Range High is 679.0. The Address State is CA. The Employment Length is 2 years. The Home Ownership is RENT. The Verification Status is Verified. The Annual Income is 75000.0. As reference, the predicted probability of this client\'s loan status being good given by the machine learning model LightGBM is 99.82%, and the probability of it being bad is 0.18%. \nAnswer:',
        "response_N": [
            {
                "id": 0,
                "reasoning_steps": '### Step-by-Step Analysis:\n\n1. **Installment (583.51)**: This is the monthly payment towards the loan. A higher installment amount generally indicates a larger loan or higher monthly payments, which could pose a higher risk if the borrower cannot manage these payments. However, without more context on the total loan amount, it is initially neutral.\n\n2. **Loan Purpose (debt_consolidation)**: Debt consolidation can indicate that the borrower is attempting to simplify their financial obligations, which could be a positive sign if they are consolidating high-interest debts. This typically reduces overall interest expenses and might improve creditworthiness.\n\n3. **Loan Application Type (Individual)**: Being an individual application rather than a joint application does not inherently impact the risk level; however, it could imply less diversification in terms of financial support.\n\n4. **Interest Rate (15.99%)**: A relatively high interest rate suggests that the loan is likely to have higher costs over time. This increases the risk if the borrower struggles to make payments, but it also indicates that the lender sees some risk in the loan.\n\n5. **Last Payment Amount (20000.95)**: This seems unusually high for a loan payment, suggesting either an error or an exceptional payment. If verified, it could be a positive indicator that the borrower has sufficient funds to pay off a large portion of the loan, reducing future risk.\n\n6. **Loan Amount (24000.0)**: The loan amount is substantial, which is a significant risk factor. Higher loan amounts increase the burden on the borrower, especially if the interest rate is high.\n\n7. **Revolving Balance (21435.0)**: A high revolving balance indicates that the borrower is using a considerable amount of available credit, which could suggest financial strain. High utilization rates are typically viewed as negative for credit scores.\n\n8. **Delinquency In 2 years (0.0)**: No delinquencies in the past two years is a strong positive. It suggests timely payments and good credit history.\n\n9. **Inquiries In 6 Months (0.0)**: Zero inquiries in the past six months is positive, indicating that the borrower has not been aggressively seeking new credit, which can lower credit scores.\n\n10. **Mortgage Accounts (0.0)**: No mortgage accounts might be neutral unless the borrower is looking to take out a mortgage soon, in which case it could be a positive if they can qualify.\n\n11. **Grade (C)**: The grade C is below average, indicating a higher risk compared to better grades like A or B.\n\n12. **Open Accounts (9.0)**: Nine open accounts could be a mix of positive and negative. It shows stability in credit usage but also increases the complexity of managing multiple accounts.\n\n13. **Revolving Utilization Rate (75.50%)**: A high utilization rate suggests that the borrower is using a significant portion of their available credit, which is risky.\n\n14. **Total Accounts (16.0)**: Having 16 total accounts indicates diverse credit usage, which is usually viewed positively.\n\n15. **FICO Range (675.0 - 679.0)**: The FICO range is in the mid-600s, which is not ideal but still within a range that can secure loans with potentially higher interest rates.\n\n16. **Address State (CA)**: The state does not directly impact creditworthiness unless there are specific regional economic factors.\n\n17. **Employment Length (2 years)**: Two years of employment is decent but not long-term experience, which could indicate instability.\n\n18. **Home Ownership (RENT)**: Renting can slightly reduce creditworthiness as lenders prefer borrowers who own their homes.\n\n19. **Verification Status (Verified)**: Verification status being "verified" is positive as it confirms the accuracy of the application details.\n\n20. **Annual Income (75000.0)**: An annual income of $75,000 is substantial, providing a strong ability to repay the loan.\n\n### Combining All Factors:\nThe high interest rate, substantial loan amount, and high revolving balance are significant negative factors. The lack of delinquencies, verified application, and stable employment provide some positive indicators. The FICO score, while not excellent, is not a critical failure. Given the high probability of the loan being good (99.82%) according to the LightGBM model, and the overwhelming majority of positive factors, the risk is manageable.\n\n### Final Assessment: good',
                "pred_label": 0,
                "gold_label": 0,
                "probs": [0.8, 0.2]
            },
            {
                "id": 1,
                "reasoning_steps": '1. **Installment (583.51)**: This is the monthly payment towards the loan. A higher installment amount generally indicates a larger loan or higher monthly payments, which could pose a higher risk if the borrower cannot manage these payments. However, without more context on the total loan amount, it is initially neutral.\n\n2. **Loan Purpose (debt_consolidation)**: Debt consolidation can indicate that the borrower is attempting to simplify their financial obligations, which could be a positive sign if they are consolidating high-interest debts. This typically reduces overall interest expenses and might improve creditworthiness.\n\n3. **Loan Application Type (Individual)**: Being an individual application rather than a joint application does not inherently impact the risk level; however, it could imply less diversification in terms of financial support.\n\n4. **Interest Rate (15.99%)**: A relatively high interest rate suggests that the loan is likely to have higher costs over time. This increases the risk if the borrower struggles to make payments, but it also indicates that the lender sees some risk in the loan.\n\n5. **Last Payment Amount (20000.95)**: This seems unusually high for a loan payment, suggesting either an error or an exceptional payment. If verified, it could be a positive indicator that the borrower has sufficient funds to pay off a large portion of the loan, reducing future risk.\n\n6. **Loan Amount (24000.0)**: The loan amount is substantial, which is a significant risk factor. Higher loan amounts increase the burden on the borrower, especially if the interest rate is high.\n\n7. **Revolving Balance (21435.0)**: A high revolving balance indicates that the borrower is using a considerable amount of available credit, which could suggest financial strain. High utilization rates are typically viewed as negative for credit scores.\n\n8. **Delinquency In 2 years (0.0)**: No delinquencies in the past two years is a strong positive. It suggests timely payments and good credit history.\n\n9. **Inquiries In 6 Months (0.0)**: Zero inquiries in the past six months is positive, indicating that the borrower has not been aggressively seeking new credit, which can lower credit scores.\n\n10. **Mortgage Accounts (0.0)**: No mortgage accounts might be neutral unless the borrower is looking to take out a mortgage soon, in which case it could be a positive if they can qualify.\n\n11. **Grade (C)**: The grade C is below average, indicating a higher risk compared to better grades like A or B.\n\n12. **Open Accounts (9.0)**: Nine open accounts could be a mix of positive and negative. It shows stability in credit usage but also increases the complexity of managing multiple accounts.\n\n13. **Revolving Utilization Rate (75.50%)**: A high utilization rate suggests that the borrower is using a significant portion of their available credit, which is risky.\n\n14. **Total Accounts (16.0)**: Having 16 total accounts indicates diverse credit usage, which is usually viewed positively.\n\n15. **FICO Range (675.0 - 679.0)**: The FICO range is in the mid-600s, which is not ideal but still within a range that can secure loans with potentially higher interest rates.\n\n16. **Address State (CA)**: The state does not directly impact creditworthiness unless there are specific regional economic factors.\n\n17. **Employment Length (2 years)**: Two years of employment is decent but not long-term experience, which could indicate instability.\n\n18. **Home Ownership (RENT)**: Renting can slightly reduce creditworthiness as lenders prefer borrowers who own their homes.\n\n19. **Verification Status (Verified)**: Verification status being "verified" is positive as it confirms the accuracy of the application details.\n\n20. **Annual Income (75000.0)**: An annual income of $75,000 is substantial, providing a strong ability to repay the loan.\n\n### Combining All Factors:\nThe high interest rate, substantial loan amount, and high revolving balance are significant negative factors. The lack of delinquencies, verified application, and stable employment provide some positive indicators. The FICO score, while not excellent, is not a critical failure. Given the high probability of the loan being good (99.82%) according to the LightGBM model, and the overwhelming majority of positive factors, the risk is manageable.\n\n### Final Assessment: good',
                "pred_label": 0,
                "gold_label": 0,
                "probs": [0.8, 0.2]
            }
            ]
    }
    
    responses = []
    data_with_scores = []
    data_tb_verified = jload("datasets/generator/test_balanced_posterior_generator_cot_N.json")
    for i, data in enumerate(data_tb_verified): #, desc="Processing data..."):
        response_best, data_scored = verifier.verify(data, save_N=True)
        responses.append(response_best)
        data_with_scores.append(data_scored)
    
    # save the whole reranked data including the prompts
    jdump(
        data_with_scores,
        "datasets/verified/test_balanced_posterior_generator_cot_N_data_with_scores.json"
        )
    
    # save the best responses
    jdump(
        responses,
        "datasets/verified/test_balanced_posterior_generator_cot_N_best_responses.json"
    )
    logging.info("Data saved successfully")
    
    metrics = compute_binary_metrics_from_results(responses)
    logging.info(f"Evaluation on the verifier-reranked results: {metrics}")