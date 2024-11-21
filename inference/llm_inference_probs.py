import torch
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

from utils import (
    jdump, 
    setup_logging, 
    compute_binary_metrics_from_results
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


SYSTEM_PROMPT = "You are a helpful assistant that classifies binary prompts into 'good' or 'bad'."


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
    data_path: str = field(
        default="datasets/train.parquet",
        metadata={
            "help": "Path to the .parquet file containing the input data."
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
        ## Generate probs predictioin  ##
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
        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text_prediction = response.strip().lower()
        if text_prediction == good_token:
            text_prediction_lable = 0 
        elif text_prediction == bad_token:
            text_prediction_lable = 1
        else:
            text_prediction_lable = "miss"
        return [good_prob, bad_prob], text_prediction_lable
    
    except Exception as e:
        logging.error(f"Error during generation for prompt '{prompt}': {e}")
        raise


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
        logging.info(f"Loading data from {args.data_path}...")
        data = pd.read_parquet(args.data_path)
        logging.info(f"Data loaded successfully. Total rows: {len(data)}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Prepare results list
    results = []

    # Iterate over each row in the dataset with a progress bar
    logging.info("Starting inference...")
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        try:
            query = row["query"]
            choices = row["choices"] 
            gold_label = row["gold"]
            record_id = row["id"]

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
        except Exception as e:
            logging.error(f"Error processing row with id {row.get('id', 'unknown')}: {e}")

    if args.lora_weights:
        args.evaluation_output_path = args.evaluation_output_path.replace(".json", f"_lora.json")
        args.inference_output_path = args.inference_output_path.replace(".json", f"_lora.json")
    # Print out evaluation results
    logging.info(f"Evaluating the results...")
    metrics = compute_binary_metrics_from_results(results)
    print(f"Evaluation on the inference results:{metrics}")
    logging.info(f"Saving evaluation results to {args.evaluation_output_path}...")
    jdump(metrics, args.evaluation_output_path)
    logging.info("Evaluation results saved successfully.")
    
    # Save results to a JSON file using jdump
    logging.info(f"Saving inference results to {args.inference_output_path}...")
    jdump(results, args.inference_output_path)
    logging.info("Inference results saved successfully.")


if __name__ == "__main__":
    main()
