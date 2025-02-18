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
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from utils.constants import Prompts, SPLIT_TOKEN, SEARCH_PATTERN, STEP_TAG
from utils.helper import setup_logging

INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value
SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_R1_FORMAT.value

@dataclass
class DatasetArguments:
    """数据集参数的数据类"""

    # 数据集 ID 或路径
    dataset_id_or_path: str = "datasets/posterior/train_posterior.parquet"
    # 数据集拆分
    dataset_splits: str = "train"
    # 分词器名称或路径
    tokenizer_name_or_path: str = None
    #
    test_size: float = 0.1

@dataclass
class SwanlabArguments:
    """SwanLab参数的数据类"""

    # 是否使用 SwanLab
    swanlab: bool
    # SwanLab 用户名
    workspace: str
    # SwanLab 的项目名
    project: str
    # SwanLab 的实验名
    experiment_name: str


def format_reward_func(completions, **kwargs):
    """
    格式奖励函数，检查模型输出格式是否匹配: <think>...</think><answer>...</answer>

    参数:
        completions (list[str]): 生成的输出
    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出
    for completion in completions:
        try:
            # 在生成的输出前添加<think>标签，便于后续正则表达式匹配
            # completion = "<think>" + completion

            if random.random() < 0.1:  # 1% 的概率将生成输出写入文件
                # 创建生成输出目录（如果不存在）
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n=====================================================\n")
                    f.write(completion)  # 写入生成的输出

            # 定义正则表达式模式，用于匹配 <think> 标签
            format_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n"
            match_format = re.search(format_regex, completion, re.DOTALL)  
            match_conclusion = re.search(SEARCH_PATTERN, completion, re.DOTALL)

            if match_format and match_conclusion:
                rewards.append(1.0)  # 如果格式不正确，奖励为 0
            else:
                rewards.append(1.0)  # 如果格式正确，奖励为 1
        except Exception:
            rewards.append(0.0)  # 如果发生异常，奖励为 0

    return rewards

def acc_reward_func(completions, label, **kwargs):
    """
    方程奖励函数，检查计算结果是否正确，数字是否符合使用要求（每个数字只用一次，只使用所提供的数字）

    参数:
        completions (list[str]): 生成的输出
        target (list[str]): 预期的答案
        nums (list[str]): 可用的数字

    返回:
        list[float]: 奖励分数
    """
    try:
        record = {"completions": completions, "label": label}
        with open("acc_reward_samples.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logging.warning(f"Error writing to JSONL file: {e}")
        
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出、预期的答案和可用的数字
    for completion, label_item in zip(completions, label):
        try:
            # 在生成的输出前添加 <think> 标签，便于后续正则表达式匹配
            # completion = "<think>" + completion
            # 定义正则表达式模式，用于匹配 <answer> 标签
            match = re.search(SEARCH_PATTERN, completion)
            if match is None:
                rewards.append(0.0)  # 如果没有匹配到 <answer> 标签，奖励为 0
                continue
            conclusion = match.group().strip()  # 提取 <answer> 标签中的内容
            if conclusion.split(SPLIT_TOKEN)[-1].split(":")[-1].strip() == label_item:
                rewards.append(1.0)

                # 10% 的概率将成功的样本写入文件
                if random.random() < 0.10:
                    # 创建生成输出目录（如果不存在）
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n=========================================\n")
                        f.write(completion)  # 写入生成的输出
            else:
                rewards.append(0.0)  # 如果不正确，奖励为 0
        except Exception:
            rewards.append(0.0)  # 如果评估失败，奖励为 0

    return rewards


def get_checkpoint(training_args: GRPOConfig):
    """
    获取最后一个检查点

    参数:
        training_args (GRPOConfig): 训练参数
    返回:
        str: 最后一个检查点的路径，如果没有检查点，则返回 None
    """
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
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction + item[text_field]}
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
    # 记录模型参数
    logging.info(f"Model parameters {model_args}")
    # 记录训练/评估参数
    logging.info(f"Training/evaluation parameters {training_args}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        (
            # 如果有指定分词器，则使用指定的分词器，否则使用模型名称
            dataset_args.tokenizer_name_or_path
            if dataset_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,  # 使用指定的模型版本
        trust_remote_code=model_args.trust_remote_code,  # 允许使用远程代码
    )
    # 如果分词器没有填充标记，则使用结束标记作为填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    train_dataset, test_dataset = get_dataset(dataset_args.dataset_id_or_path, tokenizer)

    # 参考自 huggingface/open-r1, 把attn_implementation（是否使用flash_attention）等参数传入模型初始化参数
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

    # 设置 GRPOTrainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,  # 模型名称或路径
        # 奖励函数列表，用于计算奖励分数
        reward_funcs=[
            format_reward_func,  # 格式奖励函数
            acc_reward_func,  # 方程奖励函数
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )        

    logging.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )

    # 训练模型
    if training_args.resume_from_checkpoint:
        last_checkpoint = get_checkpoint(training_args) 
        logging.info(f"Using checkpoint at {last_checkpoint} to resume training.")
    else:
        last_checkpoint = None
        
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)  

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
    trainer.save_model(training_args.output_dir)
    logging.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # 等待所有进程加载
    # TODO: the saved tokenizer_config.json leads to sentencepiece error, had to replace it manually
    # further inspection needed
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info(f"Tokenizer saved to {training_args.output_dir}")

    logging.info("*** Training complete! ***")

class NLPSwanLabCallback(SwanLabCallback):    
    def on_predict(self, args, state, control, metrics, **kwargs):
        logging.warning(f"This is a predict point: {metrics}.")
        print(metrics)
    pass
    def on_step_end(self, args, state, control, **kwargs):
        logging.info(f"This is a step point: {kwargs.keys()}.")
        print(kwargs)

def main():
    """主函数，用于执行主训练循环"""
    # 解析命令行参数和配置文件
    setup_logging()
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig, SwanlabArguments))
    model_args, dataset_args, training_args, swanlab_args = (
        parser.parse_args_and_config()
    )
    logging.info("Using default logging tool from transformers: {training_args.report_to}")
    # 如果使用 SwanLab，则创建 SwanLab 回调对象，用于训练信息记录
    if swanlab_args.swanlab:
        swanlab_callback = SwanLabCallback(
            workspace=swanlab_args.workspace,
            project=swanlab_args.project,
            experiment_name=swanlab_args.experiment_name,
        )
        callbacks = [swanlab_callback] # [NLPSwanLabCallback()]
    else:
        callbacks = None

    # 运行主训练循环
    grpo_function(model_args, dataset_args, training_args, callbacks=callbacks)

if __name__ == "__main__":
    main()
