import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from utils.constants import Prompts, SPLIT_TOKEN, SEARCH_PATTERN_RL_FORMAT
from utils.helper import setup_logging
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP_R1_KS.value
SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_R1_FORMAT.value
REPORT_INTRO = Prompts.INTRO_CUSTOMER_CREDIT_REPORT.value
EXPLANATIONS = Prompts.EXPLANATION_FEATURES.value

GOOD_DEFAULT_RISK_BOUND = 30
BAD_DEFAULT_RISK_BOUND = 70

try:
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
except Exception as e:
    raise ImportError(
        "请安装指定的 TRL 版本和 unsloth 库: "
        "pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b\n"
        "pip install unsloth"
    )


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
        # regex to match <think>\n text </think> specifically
        format_regex = r'^(?=(?:(?!<think>).)*<think>(?:(?!<think>).)*$)<think>\n([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n'
        match_format = re.search(format_regex, completion, re.DOTALL)  
        match_conclusion = re.search(SEARCH_PATTERN_RL_FORMAT, completion, re.DOTALL)

        if match_format and match_conclusion:
            rewards.append(1.5)  
        else:
            rewards.append(-1.5) 
    return rewards

def acc_reward_func(completions, label, **kwargs):
    try:
        record = {"completions": completions, "label": label}
        with open("completions_unsloth.jsonl", "a", encoding="utf-8") as f:
            json_str = json.dumps(record, ensure_ascii=False)
            f.write(json_str + "\n")
    except Exception as e:
        logging.warning(f"Error writing to JSONL file: {e}")
        
    rewards = []
    for completion, label_item in zip(completions, label):
        match = re.search(SEARCH_PATTERN_RL_FORMAT, completion)
        if match is None:
            rewards.append(-0.5) 
            continue
        prediction = match.group(1).strip() 
        if prediction == label_item:
            rewards.append(2.0)
        else:
            rewards.append(-2.0) 
    return rewards

def ks_reward_func(completions, label, **kwargs):
    rewards = []
    for completion, label_item in zip(completions, label):
        match = re.search(SEARCH_PATTERN_RL_FORMAT, completion)
        if match is None:
            rewards.append(-0.5) 
            continue
        default_risk = int(match.group(2))
        if label_item == "good" and default_risk < GOOD_DEFAULT_RISK_BOUND:
            rewards.append(1.0)
        elif label_item == "bad" and default_risk > BAD_DEFAULT_RISK_BOUND:
            rewards.append(1.0) 
        else:
            rewards.append(-1.0)
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

# 定义 GRPO 训练函数
def grpo_function(
    model_args: ModelConfig,
    dataset_args: DatasetArguments,
    training_args: GRPOConfig,
    callbacks: List,
):

    logging.info(f"Model parameters {model_args}")
    logging.info(f"Training/evaluation parameters {training_args}")

    # 从预训练模型加载模型和分词器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,  
        max_seq_length=training_args.max_completion_length,  
        fast_inference=True,  
        load_in_4bit=True,  
        max_lora_rank=model_args.lora_r,  
        gpu_memory_utilization=training_args.vllm_gpu_memory_utilization, 
        )
    logging.info(f"Loaded model {model_args.model_name_or_path}")

    # PEFT 模型
    model = FastLanguageModel.get_peft_model(
        model,
        r = model_args.lora_r, 
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], 
        lora_alpha = model_args.lora_alpha,  
        use_gradient_checkpointing = "unsloth",  
        random_state = training_args.seed,  
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, test_dataset = get_dataset(
        dataset_args.dataset_id_or_path, 
        tokenizer
        )

    trainer = GRPOTrainer(
        model=model,  
        reward_funcs=[
            format_reward_func,  
            acc_reward_func, 
            ks_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )   
    
    logging.info(
        f"""*** Starting training {
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            } for {training_args.num_train_epochs} epochs***"""
    )

    if training_args.resume_from_checkpoint:
        last_checkpoint = get_checkpoint(training_args) 
        logging.info(f"Using checkpoint at {last_checkpoint} to resume training.")
    else:
        last_checkpoint = None

    train_result = trainer.train()

    # 记录和保存指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logging.info("*** Training complete ***")

    # 保存模型和分词器
    logging.info("*** Save model ***")
    trainer.model.config.use_cache = True
    model.save_lora(training_args.output_dir)
    logging.info(f"Model saved to {training_args.output_dir}")
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info(f"Tokenizer saved to {training_args.output_dir}")
    logging.info("*** Training complete! ***")

def main():
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig, SwanlabArguments))
    model_args, dataset_args, training_args, swanlab_args = (
        parser.parse_args_and_config()
    )

    if swanlab_args.swanlab:
        swanlab_callback = SwanLabCallback(
            workspace=swanlab_args.workspace,
            project=swanlab_args.project,
            experiment_name=swanlab_args.experiment_name,
        )
        callbacks = [swanlab_callback]
    else:
        callbacks = None

    grpo_function(model_args, dataset_args, training_args, callbacks=callbacks)

if __name__ == "__main__":
    main()
