import os
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    jload, 
    jdump,
    setup_logging
)
from peft import PeftModel
import logging


SYSTEM_PROMPT_TEST = """You are a risk management assistant who is good at credit scoring.
You are given a text as description of a customer's credit report about his/her loan status below, and also the predicted loan status (probability) by a trained machine learning system.

Please analyse step by step with every possible details. Give your reasoning process in steps and your assessment in the end. \n\n
"""
SEED = 42
random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def _parse_args():
    # TODO add generation configuration 
    parser = argparse.ArgumentParser(
        description="Script for generating responses from the LLM"
        )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Path to the directory containing the model weights and tokenizer config."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="datasets/test_dataset.json",
        help="Path to the json file containing the test dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/inference_results.json",
        help="Path to the json file for saving the inferencing results"
    )
    parser.add_argument(
        '--lora_weights',
        default=None,
        help='Path to the folder that contains lora weights'
    )
    return parser.parse_args()


def _load_model_and_tokenizer(model_name_or_path, lora_weights=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="cuda"
    )
    if lora_weights:
        logging.info(f"Loading LoRA weights from {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights) 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def _generate(prompt, model, tokenizer):
    # TODO test whether it is possible to apply batch inference using messages variable
    messages = [
        {
            "role": "system", 
            "content": SYSTEM_PROMPT_TEST
            },
        {
            "role": "user", 
            "content": prompt
            },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # make to use greedy decoding
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=2048,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    setup_logging()
    args = _parse_args()
    if args.lora_weights:
        logging.info("Starting the inference process with SFT-model")
    else:
        logging.info("Starting the inference process with Base-model")
    test_dataset =  jload(args.test_data_path)
    model, tokenizer = _load_model_and_tokenizer(
        args.model_name_or_path,
        lora_weights=args.lora_weights
        )
    results = []
    for data in tqdm(test_dataset, desc="Getting response from the test dataset:"):
        response = _generate(
            data["query_cot"], 
            model, 
            tokenizer,
            )
        results.append(response)
    jdump(results, args.output_path)
    logging.info(f"Inference results saved as {args.output_path}")

if __name__ == "__main__":
    main()