import os
import torch
torch.cuda.empty_cache()
from tqdm import tqdm
from typing import Literal, List
from inference.base import BaseGenerator


class Generator(BaseGenerator):    
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

            # Prepare inputs for the batch
            prompts = [item["query"] for item in batch]  # NOTE no instruction for non-CoT case
            choices = batch[0]["choices"]
            gold_labels = [item["gold"] for item in batch]
            record_ids = [item["id"] for item in batch]

            # Make predictions for the batch
            batch_results = self.batch_predict( 
                prompts, 
                choices, 
                record_ids, 
                gold_labels,
                model=self.model, # using single device, just pass in self.model 
                )

            # Append batch results to overall results
            results.extend(batch_results)
        return results
    
    def batch_predict(
        self,
        prompt_batch, 
        choices, 
        record_ids, 
        gold_labels, 
        model=None,
        ):
        """
        Predict for a batch of data and return the results.
        """
        model_inputs = self.get_batch_model_inputs(
            prompt_batch, 
            model, 
            self.tokenizer, 
            )

        # Process each result in the batch
        results = []
        good_probs, bad_probs, labels = self.predict_token_and_probs(
            model_inputs,
            model, 
            self.tokenizer, 
            prompt_batch, 
            choices
            )
        
        # Return batch results
        for i in range(len(prompt_batch)):
            results.append(
                {   
                    "id": record_ids[i],
                    "prompt": prompt_batch[i], 
                    "pred_prob": [good_probs[i], bad_probs[i]], 
                    "pred_label": labels[i], 
                    "gold_label": int(gold_labels[i])
                    }
            )
        return results
    
    def predict_token_and_probs(self, model_inputs, model, tokenizer, prompts, choices):
        """
        Generate classification outputs for a batch of prompts, returning probabilities
        for 'good' and 'bad' tokens and their respective labels.
        """
        good_token, bad_token = choices
        good_probs, bad_probs = self._predict_probs(
            model_inputs, 
            model, 
            tokenizer, 
            good_token, 
            bad_token
            )
        labels = self._predict_token(
            model_inputs, 
            model, 
            tokenizer, 
            good_token, 
            bad_token
            )
        return good_probs, bad_probs, labels
    
    @staticmethod
    def _predict_probs(model_inputs, model, tokenizer, good_token, bad_token):
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits

        # Identify token IDs for "good" and "bad"
        good_token_id = tokenizer.convert_tokens_to_ids(good_token)
        bad_token_id = tokenizer.convert_tokens_to_ids(bad_token)

        # Extract probabilities for "good" and "bad" from the logits
        last_token_logits = logits[:, -1, :]  # Get logits for the last token of each sequence

        # Logit Masking: Mask out all other logits
        mask = torch.full_like(last_token_logits, float('-inf'))
        mask[:, good_token_id] = 0
        mask[:, bad_token_id] = 0
        masked_logits = last_token_logits + mask

        # Compute probabilities
        probabilities = torch.softmax(masked_logits, dim=-1)
        good_probs = probabilities[:, good_token_id].tolist()
        bad_probs = probabilities[:, bad_token_id].tolist()
        return good_probs, bad_probs
    
    def _predict_token(self, model_inputs, model, tokenizer, good_token, bad_token):
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                generation_config=self.get_generation_config(
                    strategy=self.generation_strategy
                    ),
                )

        # Decode generated text
        input_ids = model_inputs.input_ids
        generated_ids_trimmed = [
            output_ids[len(input_ids[i]):]
            for i, output_ids in enumerate(generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        # Determine predictions
        text_predictions = [response.strip().lower() for response in responses]
        labels = [
            0 if prediction == good_token else 1 if prediction == bad_token else "miss"
            for prediction in text_predictions
        ]
        return labels
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        batch_size=2,
    )
    
    results = generator(data)
    generator.save(results, "results.json")
    print("Final data saved")