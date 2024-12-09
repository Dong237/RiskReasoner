
"""
This is a simpel copy from `llm_inference_probs.py, except that the script uses
more complex instruction to elicit CoT process from the LLM and upgrade the inference
process to multi-GPU inference usign `accelerate.
"""


import os
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
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}")
        raise
    

def _generate(model, tokenizer, prompt, choices):
    """
    Generate a classification output for the prompt, returning probabilities
    for 'good' and 'bad' tokens.
    """
    try:
        # Prepare the messages for input
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Prepare the text input for the model
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Configure generation settings for deterministic decoding
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
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits

        # Identify token IDs for "good" and "bad"
        good_token, bad_token = choices
        good_token_id = tokenizer.convert_tokens_to_ids(good_token)
        bad_token_id = tokenizer.convert_tokens_to_ids(bad_token)

        # Extract probabilities for "good" and "bad" from the logits
        last_token_logits = logits[0, -1, :]  # Get logits for the last token

        # Logit Masking: Mask out all other logits to have the final two probs sum to 1
        # TODO this is efficient but have certain level of risk of noise amplification
        # and loss of context, could consider using classification head when finetuning involed
        mask = torch.full_like(last_token_logits, float('-inf'))
        mask[good_token_id] = 0
        mask[bad_token_id] = 0
        masked_logits = last_token_logits + mask

        # Compute probabilities
        probabilities = torch.softmax(masked_logits, dim=-1)
        good_prob = probabilities[good_token_id].item()
        bad_prob = probabilities[bad_token_id].item()
        
        ##############################
        ## Generate text prediction ##
        ##############################
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=4096,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text_prediction = response.split("\n\n")[-1].strip().lower()
        if good_token in text_prediction:
            text_prediction_lable = 0 
        elif bad_token in text_prediction:
            text_prediction_lable = 1
        else:
            text_prediction_lable = "miss"
        return [good_prob, bad_prob], text_prediction_lable
    
    except Exception as e:
        logging.error(f"Error during generation for prompt '{prompt}': {e}")
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

    # Parse arguments using HfArgumentParser
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

    # Iterate over each row in the dataset with a progress bar
    logging.info("Starting inference...")
    
    # FIXME for CoT while there is no CoT few shot data, skip few shot for now
    if args.few_shot:
        logging.info("Generating few-shot examples...")
        data_to_sample_from = pd.read_parquet(args.train_data_path)
        examples = _generate_few_shot_examples(data_to_sample_from, args.few_shot)
    
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        query = INSTRUCTION+row["query_cot"]
        choices = row["choices"] 
        gold_label = row["gold"]
        record_id = row["id"]
        
        if gold_label==0:
            continue
        
        # Add few shot examples if specified
        query = examples + query if args.few_shot else query

        # Generate probabilities
        pred_prob, text_prediction_lable = _generate(model, tokenizer, query, choices)

        # Prepare the result dictionary
        result = {
            "id": record_id,
            "pred_prob": pred_prob,
            "pred_label": text_prediction_lable,
            "label": gold_label,
            "query": query,
        }
        results.append(result)

    # Adjust the output paths based on the model type
    if args.lora_weights:
        args.evaluation_output_path = args.evaluation_output_path.replace(".json", f"_lora.json")
        args.inference_output_path = args.inference_output_path.replace(".json", f"_lora.json")
        
    # Adjust the output paths based on whether few shot examples are used
    if args.few_shot:
        args.evaluation_output_path = args.evaluation_output_path.replace(".json", f"_{args.few_shot}_shot.json")
        args.inference_output_path = args.inference_output_path.replace(".json", f"_{args.few_shot}_shot.json")
    
    # Print out evaluation results
    logging.info(f"Evaluating the results...")
    metrics = compute_binary_metrics_from_results(results)
    print(f"Evaluation on the inference results:{metrics}")
    logging.info(f"Saving evaluation results to {args.evaluation_output_path}...")
    os.makedirs(os.path.dirname(args.evaluation_output_path), exist_ok=True)
    jdump(metrics, args.evaluation_output_path)
    logging.info("Evaluation results saved successfully.")
    
    # Save results to a JSON file using jdump
    logging.info(f"Saving inference results to {args.inference_output_path}...")
    os.makedirs(os.path.dirname(args.inference_output_path), exist_ok=True)
    jdump(results, args.inference_output_path)
    logging.info("Inference results saved successfully.")


if __name__ == "__main__":
    main()