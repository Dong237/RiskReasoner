import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List

from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from utils.constants import Prompts, SPLIT_TOKEN, SEARCH_PATTERN, STEP_TAG
from utils.helper import setup_logging
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value
SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_R1_FORMAT.value
COMPLETIONS_TO_LOG = []

# 如果在使用过程中出现错误，请确保安装指定的 TRL 版本和 unsloth 库
# 可以使用以下命令进行安装：
# pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
# pip install unsloth

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
            completion = "<think>" + completion

            if random.random() < 0.1:  # 1% 的概率将生成输出写入文件
                # 创建生成输出目录（如果不存在）
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)  # 写入生成的输出

            # 定义正则表达式模式，用于匹配 <think> 和 <answer> 标签
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)  # 使用正则表达式进行匹配

            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)  # 如果格式不正确，奖励为 0
            else:
                rewards.append(1.0)  # 如果格式正确，奖励为 1
        except Exception:
            rewards.append(0.0)  # 如果发生异常，奖励为 0

    return rewards

def equation_reward_func(prompts, completions, target, nums, **kwargs):
    """
    方程奖励函数，检查计算结果是否正确，数字是否符合使用要求（每个数字只用一次，只使用所提供的数字）

    参数:
        completions (list[str]): 生成的输出
        target (list[str]): 预期的答案
        nums (list[str]): 可用的数字

    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出、预期的答案和可用的数字
    for prompt, completion, gt, numbers in zip(prompts, completions, target, nums):
        try:
            # 在生成的输出前添加 <think> 标签，便于后续正则表达式匹配
            completion = "<think>" + completion
            # 定义正则表达式模式，用于匹配 <answer> 标签
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)  # 如果没有匹配到 <answer> 标签，奖励为 0
                continue
            equation = match.group(1).strip()  # 提取 <answer> 标签中的内容
            # 提取方程中的所有数字
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # 检查所有数字是否被使用且只使用一次
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue

            # 定义允许的字符模式，只允许数字、运算符、括号和空白字符
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)  # 如果方程包含不允许的字符，奖励为 0
                continue

            # 计算方程的结果
            result = eval(equation, {"__builtins__": None}, {})

            if random.random() < 0.3:  # 30% 的概率打印出来
                print('-'*20, f"\nQuestion:\n{prompt}",
                    f"\nCompletion:\n{completion}", f"\nResult:\n{result}", f"\nTarget:\n{gt}", f"\nNumbers:\n{numbers}")


            # 检查方程是否正确且与预期答案匹配（误差小于 1e-5）
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)  # 如果正确，奖励为 1                
                # 10% 的概率将成功的样本写入文件
                if random.random() < 0.10:
                    
                    # 创建生成输出目录（如果不存在）
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"\nQuestion:\n{prompt}\nCompletion:\n{completion}\nResult:\n{result}\nTarget:\n{gt}\nNumbers:\n{numbers}")  # 写入生成的输出
                    # print('-'*20, f"\nQuestion:\n{prompt}", f"\nCompletion:\n{completion}", f"\nResult:\n{result}", f"\nTarget:\n{gt}", f"\nNumbers:\n{numbers}")


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
        ],  # 如果内存不足，可以移除 QKVO
        lora_alpha = model_args.lora_alpha,  
        use_gradient_checkpointing = "unsloth",  
        random_state = training_args.seed,  
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        'parquet',  # Specify the file format
        data_files=dataset_args.dataset_id_or_path,  # Provide the path to your local Parquet file
        split=dataset_args.dataset_splits  # Specify the desired split, if any
    )
    dataset = dataset.shuffle(seed=training_args.seed).select(range(50000))

    def generate_r1_prompt(numbers, target):
        """
        生成 R1 Countdown 游戏提示词

        参数:
            numbers (list[int]): 数字列表
            target (int): 目标值
        返回:
            dict: 生成的一个数据样本
        """
        # 定义提示词前缀
        r1_prefix = [
            {
                "role": "user",
                "content": f"使用给定的数字 {numbers}，创建一个等于 {target} 的方程。你可以使用基本算术运算（+、-、*、/）一次或多次，但每个数字只能使用一次。在 <think> </think> 标签中展示你的思考过程，并在 <answer> </answer> 标签中返回最终方程，例如 <answer> (1 + 2) / 3 </answer>。在 <think> 标签中逐步思考。",
            },
            {
                "role": "assistant",
                "content": "让我们逐步解决这个问题。\n<think>",  # 结尾使用 `<think>` 促使模型开始思考
            },
        ]

        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),  # 提示词，continue_final_message=True 表示将提示词中的最后一个消息继续到最终的输出中
            "target": target,
            "nums": numbers,
        }

    # 将数据集转换为 R1 Countdown 游戏提示词
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]  # 获取训练集
    test_dataset = train_test_split["test"]  # 获取测试集

    # 设置 GRPOTrainer
    trainer = GRPOTrainer(
        model = model,
        reward_funcs=[
            format_reward_func,  # 格式奖励函数
            equation_reward_func,  # 方程奖励函数
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

    # 训练模型
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
    """主函数，用于执行主训练循环"""
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
