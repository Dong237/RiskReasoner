import os
import torch
torch.cuda.empty_cache()
import logging
from tqdm import tqdm
from typing import Literal, Optional, List
from inference.sys1.generator_cot import GeneratorCoT
from utils.constants import Prompts

class GeneratorCoTN(GeneratorCoT):   
     
    def __init__(
        self, 
        N: int = 1,        
        cuda_visible_devices: Optional[Literal["0,1,2,3"]] = None,
        generation_strategy: Literal["greedy", "sampling"] = "sampling",
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.cuda_visible_devices = cuda_visible_devices
        self.N = N                                            # N is the N from "best-of-N" sampling
        self.generation_strategy = generation_strategy        # Overwrite the decoding method to "sampling"  
        
    def __call__(self, data_all: List[dict]):
        return self.generate(data_all)
    
    def generate(self, data_all: List[dict]):
        if self.cuda_visible_devices:
            pass
        else:
            # No multi-GPU scenario
            self.start_service(self.use_generated_ks)
            data_tb_verified = self._generate_for_all_questions(self.model, data_all)
        return data_tb_verified
            
    def _generate_for_all_questions(self, model, data_all: List[dict]):
        data_tb_verified = []
        for data in tqdm(data_all, desc="Processing all questions"):
            # NOTE since we are doing best-of-N or majority voting on single input 
            # question (i.e., already in batch) there is no proper way to do 
            # batch process of multiple questions for now
            result = self._generate_for_single_question(model, data)
            data_tb_verified.append(result)
        return data_tb_verified

    def _generate_for_single_question(self, model, data):
        
        if self.add_feature_explanations:
            cut = Prompts.INTRO_CUSTOMER_CREDIT_REPORT.value
            prompt = self.instruction.replace(cut, "") + self.explanation_features + cut + data["query_cot"]
        else:
            prompt = self.instruction + data["query_cot"]
        
        choices = data["choices"] 
        gold_label = data["gold"]        
        results = []
        choices = data["choices"] 
        gold_label = data["gold"]        
        
        div, rem = divmod(self.N, self.batch_size)
        batch_schedule = [self.batch_size] * div + ([rem] if rem else [])
        results = []
        for i, batch_size in enumerate(batch_schedule):
            results_batch = self.batch_predict(
                prompt_batch=[prompt] * batch_size, 
                choices_batch=[choices]*batch_size, 
                record_ids=list(range(batch_size)),
                gold_labels=[gold_label]*batch_size, 
                real_batch_size=batch_size,
                model=model,
                return_prompt=False,
                )
            for j in range(batch_size):
                results_batch[j]["id"] = i*batch_size + j
            results.extend(results_batch)
    
        return {
            "prompt": prompt,
            "response_N": results,
            }
     

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    num = 4
    folder = "datasets/posterior/split_output_test_balanced_posterior"
    file = f"questions_part_{num}.json"
    output_dir = f"datasets/generator/test_balanced_posterior_generator_cot_N_llama_r1_expl_4096_500_unsloth_{num}.json"
    
    # unsloth trained Qwen2.5-3B-Instruct
    # generator = GeneratorCoTN(
    #     model_name_or_path="/data1/huggingface/Qwen2.5-3B-Instruct",
    #     N=16,
    #     batch_size=8,
    #     max_new_tokens=2048, 
    #     # lora_weights="model_weights/qwen3B-grpo-unsloth/checkpoint-2000",
    #     add_feature_explanations=True,
    #     )
    
    # trl trained models
    generator = GeneratorCoTN(
        model_name_or_path="model_weights/llama-ks-2048-1.0/checkpoint-500",
        N=16,
        batch_size=8,
        max_new_tokens=4096, 
        # lora_weights="model_weights/llama-grpo-unsloth-4096/checkpoint-500",
        temperature=1.0,
        add_feature_explanations=True,
        generated_ks=True
        )
    
    generator.system_prompt = Prompts.SYSTEM_PROMPT_R1_FORMAT.value
    generator.instruction = Prompts.INSTRUCTION_STEP_BY_STEP_R1_KS.value
    data_path = os.path.join(folder, file)
    data = generator.load(os.path.join(folder, file))
    logging.info(f"Loaded data from {data_path}")
    
    results = generator(data)
    generator.save(results, output_dir)
    print(f"Final data saved to {output_dir}")