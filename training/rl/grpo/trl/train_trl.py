import numpy as np
import logging
import os
import random
import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

from datasets import load_dataset
import swanlab
from swanlab.integration.transformers import SwanLabCallback
import torch
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, TrlParser
from grpo_trainer import GRPOTrainer
from utils.constants import Prompts, SPLIT_TOKEN, SEARCH_PATTERN, STEP_TAG
from utils.helper import setup_logging

INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP_R1.value
SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_R1_FORMAT.value
REPORT_INTRO = Prompts.INTRO_CUSTOMER_CREDIT_REPORT.value
EXPLANATIONS = Prompts.EXPLANATION_FEATURES.value
COMPLETIONS_TO_LOG = []

@dataclass
class DatasetArguments:

    dataset_id_or_path: str = "datasets/posterior/train_posterior.parquet"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    test_size: float = 0.1

@dataclass
class SwanlabArguments:

    swanlab: bool
    workspace: str
    project: str
    experiment_name: str


def format_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        try:

            if random.random() < 0.1:  
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n=====================================================\n")
                    f.write(completion)  

            # regex to match <think>\n text </think> specifically
            format_regex = r"^<think>\n([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n"
            match_format = re.search(format_regex, completion, re.DOTALL)  
            match_conclusion = re.search(SEARCH_PATTERN, completion, re.DOTALL)

            if match_format and match_conclusion:
                rewards.append(1.0)  
            else:
                rewards.append(0.0) 
        except Exception:
            rewards.append(0.0) 
    return rewards

def acc_reward_func(completions, label, **kwargs):
    try:
        record = {"completions": completions, "label": label}
        with open("acc_reward_samples.jsonl", "a", encoding="utf-8") as f:
            json_str = json.dumps(record, ensure_ascii=False)
            f.write(json_str + "\n")
    except Exception as e:
        logging.warning(f"Error writing to JSONL file: {e}")
        
    rewards = []
    completion_to_log = ""
    for completion, label_item in zip(completions, label):
        try:
            match = re.search(SEARCH_PATTERN, completion)
            if match is None:
                rewards.append(0.0) 
                continue
            conclusion = match.group().strip() 
            if conclusion.split(SPLIT_TOKEN)[-1].split(":")[-1].strip() == label_item:
                rewards.append(1.0)
            else:
                rewards.append(0.0) 
        except Exception:
            rewards.append(0.0) 

        completion_to_log += completion + "\n\n" + "="*200
    COMPLETIONS_TO_LOG.append(swanlab.Text(completion_to_log, caption=label_item))
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):  # 如果输出目录存在
        # 获取最后一个检查点
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def get_dataset(
    dataset_path: str,
    tokenizer,
    test_size: float = 0.1,
    instruction: str = INSTRUCTION,
    text_field: str = "query_cot"
) -> Tuple:
    """
    Load, process, and tokenize a local JSON dataset.

    Args:
        dataset_path (str): Path to the local JSON dataset file.
        instruction (str): Instruction to prepend to each query.
        tokenizer: HuggingFace tokenizer instance.
        test_size (int, optional): Proportion of samples to use for evaluation. Defaults to 100.
        num_proc (int, optional): Number of processes for parallel tokenization. Defaults to 4.
        text_field (str, optional): Key in the JSON data to use for the prompt. Defaults to "query_cot".

    Returns:
        Tuple: Tokenized training and evaluation datasets.
    """
    # Check if the dataset file exists
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    # Load the dataset from the local JSON file
    dataset = load_dataset(
        'parquet',  # Specify the file format
        data_files=dataset_path,  # Provide the path to your local Parquet file
        split="train"  # Specify the desired split, if any
    )

    # Add the 'query' and 'gold_label' fields
    def apply_template(item):
        vanilla_prompt = instruction + EXPLANATIONS + REPORT_INTRO + item[text_field]
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": vanilla_prompt}
            ]
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "label": item['choices'][int(item['gold'])],
        }

    dataset = dataset.map(apply_template, remove_columns=dataset.column_names)
    train_test_split = dataset.train_test_split(test_size=test_size)
    train_dataset = train_test_split["train"]  # 获取训练集
    test_dataset = train_test_split["test"]  # 获取测试集
    return train_dataset, test_dataset

def grpo_function(
    model_args: ModelConfig,
    dataset_args: DatasetArguments,
    training_args: GRPOConfig,
    callbacks: List,
):

    logging.info(f"Model parameters {model_args}")
    logging.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        (
            dataset_args.tokenizer_name_or_path
            if dataset_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision, 
        trust_remote_code=model_args.trust_remote_code,  
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, test_dataset = get_dataset(
        dataset_args.dataset_id_or_path, 
        tokenizer
        )

    logging.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,  
        reward_funcs=[
            format_reward_func,  
            acc_reward_func, 
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )        

    logging.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )

    if training_args.resume_from_checkpoint:
        last_checkpoint = get_checkpoint(training_args) 
        logging.info(f"Using checkpoint at {last_checkpoint} to resume training.")
    else:
        last_checkpoint = None
        
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)  
    
    swanlab.init()
    swanlab.log({"Prediction": COMPLETIONS_TO_LOG})
    swanlab.finish()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logging.info("*** Training complete ***")
    logging.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logging.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()
    # TODO: the saved tokenizer_config.json leads to sentencepiece error, had to replace it manually
    # further inspection needed
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info(f"Tokenizer saved to {training_args.output_dir}")
    logging.info("*** Training complete! ***")

def main():
    setup_logging()
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig, SwanlabArguments))
    model_args, dataset_args, training_args, swanlab_args = (
        parser.parse_args_and_config()
    )
    logging.info("Using default logging tool from transformers: {training_args.report_to}")
    if swanlab_args.swanlab:
        swanlab_callback = SwanLabCallback(
            workspace=swanlab_args.workspace,
            project=swanlab_args.project,
            experiment_name=swanlab_args.experiment_name,
        )
        callbacks = [swanlab_callback] # [NLPSwanLabCallback()]
    else:
        callbacks = None

    grpo_function(model_args, dataset_args, training_args, callbacks=callbacks)

if __name__ == "__main__":
    main()
