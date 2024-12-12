"""
The script achieves the similar goal of `llm_inference_probs.py, except that it uses
more complex instruction to elicit CoT process from the LLM to do binary classification.
"""


import os
import re
import random
import torch
torch.cuda.empty_cache()
import logging
import itertools
import pandas as pd
from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass, field
from peft import PeftModel
from transformers import (
    HfArgumentParser, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
from utils.helper import (
    jdump, 
    setup_logging, 
    compute_binary_metrics_from_results
)
from utils.constants import (
    Prompts, 
    STEP_TAG, 
    SPLIT_TOKEN, 
    SEARCH_PATTERN
)
from accelerate import (
    init_empty_weights,
    infer_auto_device_map, 
    load_checkpoint_and_dispatch
    )

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
SEED = 42
random.seed(SEED)
MAX_MEMORY = {0: "64GiB", 1: "64GiB"}  # Adjust based on your setup

SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value
INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value
# Define batch size
BATCH_SIZE = 16


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


# def load_model_and_tokenizer(model_name_or_path, lora_weights=None):
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


def load_model_and_tokenizer(model_name_or_path, lora_weights=None):
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
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=2048,
            truncation=True,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
            )
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}")
        raise


def get_variation(word: Literal["good", "bad"]):
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


def get_tokens_id(tokenizer, good_token, bad_token, pred_token):
    """
    Given the two tokens from ``choices`` and the predicted token,
    get the corresponding token ids. The ids of the two binary tokens
    are used later for performing masking and retrieving the probabilities.
    """
    good_tokens, bad_tokens = get_variation(good_token), get_variation(bad_token)
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


def find_continuous_indices(tensor_pattern, tensor_sequence):
    """
    Finds the starting indices of a continuous subsequence (tensor_pattern) 
    in a given sequence (tensor_sequence).
    """
    pattern_length = len(tensor_pattern)

    # Create sliding windows of size equal to the pattern length over the sequence
    sliding_windows = tensor_sequence.unfold(0, pattern_length, 1)  # Shape: [sequence_length - pattern_length + 1, pattern_length]
    
    # Compare each window with the pattern
    is_match = (sliding_windows == tensor_pattern).all(dim=1)  # Check for full matches
    
    # Find the first match
    match_indices = torch.where(is_match)[0]
    if len(match_indices) > 0:
        start_index = match_indices[0].item()  # Convert to Python int
        return list(range(start_index, start_index + pattern_length))
    return []


def predict_token_and_probs(
    model, 
    tokenizer, 
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
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
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
        matched_text = matched_text.replace(STEP_TAG, "").replace("\n", "")
        pred_token = matched_text.split(":")[-1]
        
        good_token_id, bad_token_id = get_tokens_id(
            tokenizer, good_token, bad_token, pred_token
            )
        
        if not good_token_id or not bad_token_id:
            # Make a random guess if output has no valid prediction
            good_prob, bad_prob = 0.5, 0.5
        else:
            # Now start to get the logits for the pred_token 
            matched_generated_ids = tokenizer(
                matched_text, return_tensors="pt"
                )["input_ids"].to(model.device)
            # FIXME during index matching below, I skip over the first token since its variant is quite complicated
            # and unstable. But luckily, there are always more than 2 matched tokens in the list so this should be fine.
            indices = find_continuous_indices(matched_generated_ids[0][1:], generated_ids)
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
            
    
def batch_predict(
    model, 
    tokenizer, 
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
    texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the inputs as a batch 
    # and pad to the max length in this batch
    model_inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True,
        truncation=True
        ).to(model.device)

    # Configure generation settings for deterministic decoding
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    # Generate responses for the batch
    with torch.no_grad():
        generated_dict = model.generate(
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
        good_prob, bad_prob, pred_label = predict_token_and_probs(
            model, 
            tokenizer, 
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
    

def main():
    # Set up logging
    setup_logging()

    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path,
        lora_weights=args.lora_weights
    )

    # Load input data
    data = pd.read_parquet(args.test_data_path)
    logging.info(f"Data loaded successfully. Total rows: {len(data)}")

    # Prepare results list
    results = []

    # Process data in batches
    logging.info("Starting batch inference...")
    for start_idx in tqdm(range(0, len(data), BATCH_SIZE), desc="Processing batches"):
        end_idx = min(start_idx + BATCH_SIZE, len(data))
        batch = data.iloc[start_idx:end_idx]
        real_batch_size = end_idx - start_idx  # the last batch might be smaller than BATCH_SIZE

        # Prepare inputs for the batch
        prompts = [INSTRUCTION + row["query_cot"] for _, row in batch.iterrows()]
        choices_list = [row["choices"] for _, row in batch.iterrows()]
        gold_labels = [row["gold"] for _, row in batch.iterrows()]
        record_ids = [row["id"] for _, row in batch.iterrows()]

        # Make predictions for the batch
        batch_results = batch_predict(
            model, 
            tokenizer, 
            prompts, 
            choices_list, 
            record_ids, 
            gold_labels,
            real_batch_size,
            )

        # Append batch results to overall results
        results.extend(batch_results)

    # Adjust the output paths based on the model type
    if args.lora_weights:
        args.evaluation_output_path = args.evaluation_output_path.replace(".json", f"_lora.json")
        args.inference_output_path = args.inference_output_path.replace(".json", f"_lora.json")

    # Print out evaluation results
    logging.info(f"Evaluating the results...")
    metrics = compute_binary_metrics_from_results(results)
    print(f"Evaluation on the inference results:{metrics}")
    os.makedirs(os.path.dirname(args.evaluation_output_path), exist_ok=True)
    jdump(metrics, args.evaluation_output_path)
    logging.info(f"Evaluation results saved to {args.evaluation_output_path}.")

    # Save results to a JSON file using jdump
    os.makedirs(os.path.dirname(args.inference_output_path), exist_ok=True)
    jdump(results, args.inference_output_path)
    logging.info(f"Inference results saved to {args.inference_output_path}.")


if __name__ == "__main__":
    main()