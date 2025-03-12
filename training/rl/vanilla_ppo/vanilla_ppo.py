# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import shutil
import torch
from torch import nn
from tqdm import tqdm
from accelerate.utils import DummyOptim
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union, Literal
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from trl import (
    PPOConfig, 
    PPOTrainer,
    get_kbit_device_map,
    get_quantization_config,
    create_reference_model,
    AutoModelForCausalLMWithValueHead
)
from peft import get_peft_model, LoraConfig
from utils.constants import Prompts, SPLIT_TOKEN

# Define your instruction and system prompt
INSTRUCTION = Prompts.INSTRUCTION_STEP_BY_STEP.value
SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value

@dataclass
class ScriptArguments:
    dataset_path: str = "datasets/posterior/train_posterior.json"
    eval_size: int = 1000

@dataclass
class PPOArguments(PPOConfig):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    learning_rate: float = 1e-5
    steps: int = 50_000 
    num_ppo_epochs: int = 6
    mini_batch_size: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    output_dir: str = "models/ppo/SparseRM"
    remove_unused_columns: bool = False # Must set this to false in cutom reward case
    total_episodes: int = 10000
    missing_eos_penalty: float = 1.0
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    whiten_rewards: bool = False
    kl_coef: float = 0.02
    cliprange: float = 0.1
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = "/data/youxiang/huggingface/Qwen2.5-7B-Instruct"
    model_revision: str = "main"
    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = "bfloat16"
    trust_remote_code: bool = True
    model_max_length: int = 1024
    max_new_tokens: int = 1024
    attn_implementation: Optional[str] = None
    use_peft: bool = True
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "v_proj"]  # Adjusted for your model
    )
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False

def get_peft_config(model_config: ModelArguments) -> Optional[LoraConfig]:
    if not model_config.use_peft:
        return None
    
    if 'chat' in model_config.model_name_or_path.lower():
        modules_to_save = None
    else:
        modules_to_save = ["wte", "lm_head"]
        
    peft_config = LoraConfig(
        task_type=model_config.lora_task_type,
        r=model_config.lora_r,
        target_modules=model_config.lora_target_modules,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        modules_to_save=modules_to_save
    )
    return peft_config

def preprocess(
    dataset_path: str,
    tokenizer,
    eval_size: int = 100,
    num_proc: int = 16,
    instruction: str = INSTRUCTION,
    text_field: str = "query_cot"
) -> Tuple:
    """
    Load, process, and tokenize a local JSON dataset.

    Args:
        dataset_path (str): Path to the local JSON dataset file.
        instruction (str): Instruction to prepend to each query.
        tokenizer: HuggingFace tokenizer instance.
        eval_size (int, optional): Number of samples to use for evaluation. Defaults to 100.
        num_proc (int, optional): Number of processes for parallel tokenization. Defaults to 4.
        text_field (str, optional): Key in the JSON data to use for the prompt. Defaults to "query_cot".

    Returns:
        Tuple: Tokenized training and evaluation datasets.
    """
    # Check if the dataset file exists
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    # Load the dataset from the local JSON file
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    # Add the 'query' and 'gold_label' fields
    def add_query_and_gold(example):
        example['prompt'] = instruction + example[text_field]
        example['gold_label'] = example['choices'][int(example['gold'])]
        return example
    dataset = dataset.map(add_query_and_gold, remove_columns=dataset.column_names)

    # Split the dataset into training and evaluation sets
    train_size = len(dataset) - eval_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))

    # Define the tokenization function
    def tokenize_function(examples):
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            for prompt in examples["prompt"]
        ]

        # Convert prompts to text using the tokenizer's chat template
        texts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Standard tokenization
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=True,
        )

        # Explicitly carry over gold_label
        tokenized["gold_label"] = examples["gold_label"]
        return tokenized

    # Tokenize the training dataset
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=['prompt'],
        num_proc=num_proc
    )

    # Tokenize the evaluation dataset
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=['prompt'],
        num_proc=num_proc
    )

    return train_dataset, eval_dataset

def compute_custom_rewards(batch) -> List[float]:
    """
    Compute custom rewards based on the generated sequences and gold labels.
    
    Args:
        batch (Dict[str, List[Any]]): Batch of input ids, generated text sequences and gold labels.
    Returns:
        List[float]: List of reward scores of tensor type (1.0 if correct, else 0.0).
    """
    
    device = batch["input_ids"][0].device
    generated_sequences = batch["response"]
    gold_labels = batch["gold_label"]
    rewards = []
    
    for gen_seq, gold in zip(generated_sequences, gold_labels):
        # Extract the last sentence
        last_sentence = gen_seq.lower().split(SPLIT_TOKEN.lower())[-1]
        # Check if the expected answer is in the last sentence
        if gold.strip().lower() in last_sentence.strip():
            reward = torch.tensor(1.0).to(device)
        else:
            reward = torch.tensor(0.0).to(device)
        rewards.append(reward)
    return rewards

def collator(data):
    # keys_to_collect = ['gold_label', 'input_ids', 'attention_mask']
    # ren =  dict((key, [d[key] for d in data if key in d]) for key in keys_to_collect)
    data_dict = {
        "gold_label": [],
        "input_ids": [],
        "attention_mask": [],
    }
    for item in data:
        data_dict["gold_label"].append(item["gold_label"]) 
        data_dict["input_ids"].append(torch.tensor(item["input_ids"]))
        data_dict["attention_mask"].append(torch.tensor(item["attention_mask"]))
    return data_dict
    
def train():
    
    # Parse arguments
    parser = HfArgumentParser(
        (ScriptArguments, PPOArguments, ModelArguments)
        )
    script_args, ppo_args, model_args = parser.parse_args_into_dataclasses()
    
    # Remove output_dir if exists
    shutil.rmtree(ppo_args.output_dir, ignore_errors=True)

    #####################
    # Model & Tokenizer #
    #####################
    torch_dtype = (
        model_args.torch_dtype 
        if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    
    quantization_config = get_quantization_config(model_args)
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map= get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        model_max_length=model_args.model_max_length, 
        truncation=True,
        padding_side="left", 
        trust_remote_code=model_args.trust_remote_code
    )
    
    tokenizer.add_special_tokens(
        {"eos_token": "<|im_end|>"}
        )
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch_dtype,
        use_cache=False,
        # device_map="auto",
    )
    ref_model = create_reference_model(model)
    
    peft_config = get_peft_config(model_args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    ## Alternative code block if ref_model is not needed
    # peft_config = get_peft_config(model_args)
    # if peft_config is None:
    #     ref_model = create_reference_model(model)
    # else:
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()
    #     ref_model = None  # Reference model is not needed when using PEFT

    # Load model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model, # model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs
    )
    model.gradient_checkpointing_enable()
    model = model.to(torch.bfloat16) # FIXME this is a dirty fix for mix-precision training issue
    
    ###########
    # Dataset #
    ###########
    # Process dataset only on the main process
    with PartialState().local_main_process_first():
        train_dataset, eval_dataset = preprocess(
            script_args.dataset_path, 
            tokenizer, 
            script_args.eval_size
        )
    
    ################
    # Training Loop #
    ################
    # Initialize PPOTrainer as per older TRL
    ppo_trainer = PPOTrainer(
        config=ppo_args,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        # optimizer=dummy_optimizer,
    )
    
    # Define generation kwargs
    generation_kwargs = {

        "min_length": -1,
        "top_k": 0.0,  # NOTE keepping this 0 appears to solve negative KL issue
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": model_args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "return_prompt": False,
        "remove_padding": True  # NOTE it is also required to keep this True to solve negative KL issue
    }

    epochs = ppo_args.num_ppo_epochs
    for epoch in tqdm(range(epochs), desc="Epoch"):
        for batch in tqdm(ppo_trainer.dataloader, desc="Batch"):
            # Get query tensors
            query_list = batch["input_ids"]
            
            #### Get response from model
            response_tensors = ppo_trainer.generate(query_list, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
            #### Compute reward scores
            reward_list = compute_custom_rewards(batch)

            #### Run PPO step
            stats = ppo_trainer.step(query_list, response_tensors, reward_list)
            ppo_trainer.log_stats(stats, batch, reward_list)
    
    #### Save model
    ppo_trainer.save_pretrained(ppo_args.output_dir)
    
    # If using PEFT, save the adapters separately (optional)
    if model_args.use_peft:
        model.save_pretrained(ppo_args.output_dir, adapter=True)
    
    print(f"Model training results saved to: {ppo_args.output_dir}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Set CUDA device
    train()