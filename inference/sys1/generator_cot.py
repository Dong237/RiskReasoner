import os
import re
import torch
torch.cuda.empty_cache()
import logging
from tqdm import tqdm
from typing import Literal, List
from utils.helper import jdump
from inference.base import BaseGenerator


class GeneratorCoT(BaseGenerator):    
    def __init__(
        self, 
        batch_size: int = 1,
        generation_strategy: Literal["greedy", "sampling"] = "greedy",
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.batch_size = batch_size 
        self.generation_strategy = generation_strategy
    
    def __call__(self, data_all: List[dict]):
        return self.generate(data_all)
    
    def generate(self, data_all: List[dict]):
        self.start_service()
        results = []
        for start_idx in tqdm(range(0, len(data_all), self.batch_size), desc="Processing batches"):
            end_idx = min(start_idx + self.batch_size, len(data_all))
            batch = data_all[start_idx:end_idx]
            real_batch_size = end_idx - start_idx  # the last batch might be smaller than BATCH_SIZE

            # Prepare inputs for the batch
            prompts = [self.instruction + item["query_cot"] for item in batch]
            choices_batch = [item["choices"] for item in batch]
            gold_labels = [item["gold"] for item in batch]
            record_ids = [item["id"] for item in batch]

            # Make predictions for the batch
            batch_results = self._batch_predict( 
                prompts, 
                choices_batch, 
                record_ids, 
                gold_labels,
                real_batch_size,
                )

            # Append batch results to overall results
            results.extend(batch_results)
        return results
    
    def _batch_predict(
        self,
        prompt_batch, 
        choices_batch, 
        record_ids, 
        gold_labels, 
        real_batch_size,
        return_prompt=True,
        ):
        
        """
        Predict for a batch of data and return the results.
        """
        model_inputs, generated_dict = self.generate_for_batch(
            prompt_batch, 
            self.model, 
            self.tokenizer, 
            self.generation_strategy
            )

        # Process each result in the batch
        results = []
        for idx in range(real_batch_size):
            # Make prediction
            probs, pred_label, response = self._predict_token_and_probs(
                self.model,
                self.tokenizer,
                model_inputs, 
                generated_dict, 
                choices_batch, 
                idx
                )
            
            # Prepare the result dictionary
            result = {
                "id": record_ids[idx],
                "reasoning_steps": response,
                "pred_prob": probs,
                "pred_label": pred_label,
                "gold_label": gold_labels[idx],
                "prompt": prompt_batch[idx] if return_prompt else None
            }
            results.append(result)

        return results
    
    def _predict_token_and_probs(
        self,
        model, 
        tokenizer, 
        model_inputs, 
        generated_dict, 
        choices, 
        idx
        ):
        
        #######################
        ##  Text Prediction  ##
        #######################
        # Extract the generated text for the current prompt
        generated_ids = generated_dict.sequences[idx]
        generated_ids = generated_ids[model_inputs["input_ids"][idx].size()[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        text_prediction = response.lower().split(self.split_token.lower())[-1].replace(":", "").strip()
        good_token, bad_token = choices[idx]
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
        good_prob, bad_prob = self._predict_probs(
            model, 
            tokenizer, 
            response, 
            generated_ids, 
            generated_dict.logits,
            good_token, 
            bad_token,
            idx
        )
            
        return [good_prob, bad_prob], pred_label, response

    def _predict_probs(
        self,
        model, 
        tokenizer, 
        response, 
        generated_ids, 
        generated_logits,
        good_token, 
        bad_token,
        idx
        ):
        match = re.search(self.search_pattern, response, re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            # To clean up the matched text, this is vital!
            matched_text = matched_text.replace(self.step_tag, "").replace("\n", "").replace("*", "") # get rid of possible '*' to get clean ids later
            pred_token = matched_text.split(":")[-1] 
            
            good_token_id, bad_token_id = self._get_tokens_id(
                tokenizer, good_token, bad_token, pred_token
                )
            
            if not good_token_id or not bad_token_id:
                # Make a random guess if output has no valid prediction
                logging.warning(f"Output has no valid prediction, making a random guess.")
                return 0.5, 0.5
            else:
                # Now start to get the logits for the pred_token 
                matched_generated_ids = tokenizer(
                    matched_text, return_tensors="pt"
                    )["input_ids"].to(model.device)
                # FIXME during index matching below, I skip over the first token since its variant is quite complicated
                # and unstable. But luckily, there are always more than 2 matched tokens in the list so this should be fine.
                indices = self._find_continuous_indices(matched_generated_ids[0][1:], generated_ids)
                try:
                    target_logits = generated_logits[indices[-1]][idx]
                except:
                    # This is quite rare but still can happen
                    logging.warning(f"Index matching failed, making a random guess.")
                    return 0.5, 0.5
                
                mask = torch.full_like(target_logits, float('-inf'))
                mask[good_token_id], mask[bad_token_id] = 0, 0
                masked_logits = target_logits + mask

                # Compute probabilities
                probabilities = torch.softmax(masked_logits, dim=-1)
                good_prob = probabilities[good_token_id].item()
                bad_prob = probabilities[bad_token_id].item()
                return good_prob, bad_prob
        else:
            # make a random guess if regex failed / output has no prediction
            logging.warning("Regex failed, making a random guess.")
            return 0.5, 0.5


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
    
    generator = GeneratorCoT(
        model_name_or_path="/data/youxiang/huggingface/Qwen2.5-7B-Instruct",
        batch_size=1,
    )
    
    results = generator(data)
    jdump(results, "result_cot.json")