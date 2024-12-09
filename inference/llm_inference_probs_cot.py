"""
This is a simpel copy from ``llm_inference_probs.py``, except that the script uses
more complex instruction to elicit CoT process from the LLM and upgrade the inference
process to multi-GPU inference usign ``accelerate``.
"""

import random
import torch
torch.cuda.empty_cache()
import logging
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from peft import PeftModel
from transformers import (
    HfArgumentParser, 
    AutoModelForCausalLM, 
    AutoTokenizer
)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helper import (
    jdump, 
    setup_logging, 
    compute_binary_metrics_from_results
)
from utils.constants import Prompts
from accelerate import (
    init_empty_weights,
    infer_auto_device_map, 
    load_checkpoint_and_dispatch
    )

SEED = 42
random.seed(SEED)
MAX_MEMORY = {0: "64GiB", 1: "64GiB"}  # Adjust based on your setup

SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value
INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value
BATCH_SIZE = 32
## FIXME this batch inference script has potential bug when getting the probs for computing AUC KS etc. 
# Must check the code during next inference  

@dataclass
class Arguments:
    """
    Arguments for generating responses from the LLM.
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={
            "help": "Path to the directory containing the model weights and tokenizer config."
        }
    )
    train_data_path: str = field(
        default="datasets/train.parquet",
        metadata={
            "help": "Path to the .parquet file containing the training data for few shot learning."
        }
    )
    test_data_path: str = field(
        default="datasets/test.parquet",
        metadata={
            "help": "Path to the .parquet file containing the test data."
        }
    )
    few_shot: int = field(
        default=None,
        metadata={
            "help": "n-shot to use for while inferencing"
        }
    )
    inference_output_path: str = field(
        default="results/inferences/inference_results.json",
        metadata={
            "help": "Path to the json file for saving the inferencing results."
        }
    )
    evaluation_output_path: str = field(
        default="results/evaluations/evaluation_results.json",
        metadata={
            "help": "Path to the json file for saving the evaluation results."
        }
    )
    lora_weights: str = field(
        default=None,
        metadata={
            "help": "Path to the folder that contains LoRA weights."
        }
    )


# def _load_model_and_tokenizer(model_name_or_path, lora_weights=None):
#     """
#     Load the model and tokenizer, optimized for multi-GPU using the Accelerate library.
#     """
#     # Load the model with an empty initialization to prepare for dispatch
#     logging.info(f"Loading model from {model_name_or_path}...")
#     with init_empty_weights():
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path,
#             torch_dtype="auto"  # Automatically determines the best precision
#         )
    
#     # Optionally load LoRA weights for fine-tuning
#     if lora_weights:
#         logging.info(f"Loading LoRA weights from {lora_weights}")
#         model = PeftModel.from_pretrained(model, lora_weights)

#     # Automatically infer the device map for optimal multi-GPU partitioning
#     device_map = infer_auto_device_map(
#         model,
#         max_memory=MAX_MEMORY  # Adjust based on your setup
#     )
#     logging.info(f"Using device map: {device_map}")

#     # Dispatch the model to the appropriate devices based on the inferred device map
#     model = load_checkpoint_and_dispatch(
#         model,
#         checkpoint=model_name_or_path,
#         # device_map=device_map,
#         device_map="balanced",
#         no_split_module_classes=["Qwen2DecoderLayer"]  # Customize for your model type
#     )

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     tokenizer.padding_side = "left"  # Ensure compatibility with causal LM
#     tokenizer.truncation_side = "right"

#     logging.info("Model and tokenizer successfully loaded and dispatched.")
#     return model, tokenizer


def _load_model_and_tokenizer(model_name_or_path, lora_weights=None):
    """
    Load the model and tokenizer from the specified path, optionally loading LoRA weights.
    """
    try:
        logging.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="cuda"
        )
        if lora_weights:
            logging.info(f"Loading LoRA weights from {lora_weights}")
            model = PeftModel.from_pretrained(model, lora_weights) 
        # TODO need left padding here
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}")
        raise
    

def _generate(model, tokenizer, prompts, choices):
    """
    Generate classification output for a batch of prompts, returning probabilities
    for 'good' and 'bad' tokens in batches.
    """
    try:
        # Prepare the messages for input using the chat template for each prompt
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},  # Common system message
                {"role": "user", "content": prompt}  # User's input prompt
            ]
            for prompt in prompts
        ]
        
        # Apply the chat template for each prompt
        text_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize all prompts at once
        model_inputs = tokenizer(
            text_inputs, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

        # Configure generation settings
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

        #################################
        ## Generate probs prediction  ##
        #################################
        # Get logits from the model
        # FIXME always keep in mind that the most probable token is 
        # not necessarily "good" or "bad", could be e.g., "GOOD" or other forms
        # in this case (weak instruction-following ability e.g, chat-version) 
        # this binary approach will fail
        # Get logits from the model for the entire batch
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits

        good_token, bad_token = choices[0]
        good_token_id = tokenizer.convert_tokens_to_ids(good_token)
        bad_token_id = tokenizer.convert_tokens_to_ids(bad_token)

        # Extract probabilities for "good" and "bad" from the logits (for each prompt in the batch)
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

        ##############################
        ## Generate text prediction ##
        ##############################
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=2048,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Post-processing the predictions
        predictions = []
        for i, response in enumerate(responses):
            text_prediction = response.split("\n\n")[-1].strip().lower()
            if good_token in text_prediction:
                text_prediction_label = 0
            elif bad_token in text_prediction:
                text_prediction_label = 1
            else:
                text_prediction_label = "miss"
            predictions.append((good_probs[i], bad_probs[i], text_prediction_label))

        return predictions  # Return a list of tuples: (good_prob, bad_prob, text_prediction_label)
    
    except Exception as e:
        logging.error(f"Error during batch generation: {e}")
        raise



def _generate_few_shot_examples(dataframe, n):
    import pandas as pd  # Ensure pandas is imported
    
    split_token = "Text:"
    examples = "Here are some provided examples: \n\n"

    # Split the dataframe into "good" and "bad" based on the "answer" column
    good_df = dataframe[dataframe["answer"] == "good"]
    bad_df = dataframe[dataframe["answer"] == "bad"]
    
    # Calculate how many "good" and "bad" examples are needed
    num_good = (n // 2) + (n % 2)  # "good" gets the extra example if n is odd
    num_bad = n // 2

    # Sample the required number of examples from each subset
    good_samples = good_df.sample(n=num_good, random_state=42)
    bad_samples = bad_df.sample(n=num_bad, random_state=42)

    # Combine and shuffle the sampled examples
    sampled_df = pd.concat([good_samples, bad_samples]).sample(frac=1, random_state=42)  # Shuffle the combined samples

    # Construct the examples string
    for _, sample in sampled_df.iterrows():
        query = split_token + sample["query"].split(split_token)[-1]
        answer = sample["answer"]
        examples += query + answer + "\n\n"
    return examples
    

def main():
    # Set up logging
    setup_logging()

    # Parse arguments
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load model and tokenizer
    try:
        model, tokenizer = _load_model_and_tokenizer(
            args.model_name_or_path,
            lora_weights=args.lora_weights
        )
    except Exception as e:
        logging.error(f"Exiting program due to error: {e}")
        return

    # Load input data
    try:
        logging.info(f"Loading data from {args.test_data_path}...")
        data = pd.read_parquet(args.test_data_path)
        logging.info(f"Data loaded successfully. Total rows: {len(data)}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Prepare results list
    results = []    

    logging.info("Starting batch inference...")

    # Prepare the batches
    for i in tqdm(range(0, len(data), BATCH_SIZE), total=len(data)//BATCH_SIZE, desc="Processing batches"):
        batch = data.iloc[i:i+BATCH_SIZE]

        # Prepare the prompts and choices for the batch
        queries = [INSTRUCTION + row["query_cot"] for _, row in batch.iterrows()]
        choices = [row["choices"] for _, row in batch.iterrows()]

        # Generate batch predictions
        predictions = _generate(model, tokenizer, queries, choices)

        # Process and store the results
        for idx, (good_prob, bad_prob, text_prediction_label) in enumerate(predictions):
            row = batch.iloc[idx]
            result = {
                "id": row["id"],
                "pred_prob": [good_prob, bad_prob],
                "pred_label": text_prediction_label,
                "label": row["gold"],
                "query": queries[idx],
            }
            results.append(result)

    # Save evaluation and inference results as before
    logging.info("Evaluating the results...")
    metrics = compute_binary_metrics_from_results(results)
    logging.info(f"Evaluation on the inference results: {metrics}")

    # Adjust output paths
    if args.lora_weights:
        args.evaluation_output_path = args.evaluation_output_path.replace(".json", f"_lora.json")
        args.inference_output_path = args.inference_output_path.replace(".json", f"_lora.json")
    
    if args.few_shot:
        args.evaluation_output_path = args.evaluation_output_path.replace(".json", f"_{args.few_shot}_shot.json")
        args.inference_output_path = args.inference_output_path.replace(".json", f"_{args.few_shot}_shot.json")
    
    # Save results to JSON
    os.makedirs(os.path.dirname(args.evaluation_output_path), exist_ok=True)
    jdump(metrics, args.evaluation_output_path)
    logging.info("Evaluation results saved successfully.")
    
    os.makedirs(os.path.dirname(args.inference_output_path), exist_ok=True)
    jdump(results, args.inference_output_path)
    logging.info("Inference results saved successfully.")



if __name__ == "__main__":
    main()
