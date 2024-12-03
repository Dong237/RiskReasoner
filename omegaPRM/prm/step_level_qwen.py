# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import time
import json
import math
import random
import logging
import os
from typing import Literal
from typing import Dict, Optional, List
from wandb.sdk import login
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.helper import is_wandb_logged_in
from utils.constants import Prompts

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers 
from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


torch.cuda.empty_cache()
random.seed(42)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
LOCAL_RANK = None
STEP_TAG = "\n\n"     # Step tag that may also exist in the Instruction
STEP_TAG_REAL = "ки"  # Step tag that only exist in the reasoning steps 
GOOD_TOKEN, BAD_TOKEN = '+', '-'
SYSTEM_PROMPT = Prompts.SYSTEM_PROMPT_CREDIT_SCORING.value


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen2.5-1.5B-Instruct-GPTQ-Int8"
        )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = field(
        default=False, 
        metadata={"help": "Whether to preprocess data upfront or not."}
    )
    validation_fraction: float = 0.1


@dataclass
class WandbArguments:
    use_wandb: bool = False
    key: str = ""
    relogin: bool = False
    force: bool = False
    timeout: int = None
    wandb_project: str = "RiskReasoner"
    wandb_run_name: str = ""
    wandb_watch: Literal["false", "gradients", "all"] = "false "
    wandb_log_model: Literal["false", "checkpoint", "end"] = "false"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # model_max_length determines the input data size and thereby has a
    # great impact on the memory usage, setting it properly (not too large)
    # will give you more space on GPUs for increasing the batch size
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    logging_strategy: str = "steps"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 10
    save_strategy: str = "steps"
    save_total_limit: int = 3


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj","v_proj"] # ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def rank0_print(*args):
    if LOCAL_RANK == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(sources, tokenizer, max_len=2048):
    """Prepare input data for training."""
    
    # Find all occurrences of the step_tag_id
    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]
    
    # Note that there must be a space in front of the STEP_TAG_REAL
    step_tag_real_id = tokenizer.encode(f" {STEP_TAG_REAL}")[-1] 
    good_token_id = tokenizer.encode(f"{GOOD_TOKEN}")[-1]
    bad_token_id = tokenizer.encode(f"{BAD_TOKEN}")[-1]
    input_ids, labels, attentionn_masks = [], [], []
    
    for example in sources:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},  # Common system message
            {"role": "user", "content": example['question']}  # User's input prompt
        ]
        question_with_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # We handle tokenization ourselves
            add_generation_prompt=True  # Add generation-specific prompt (like stop tokens, etc.)
        )
        input_text = question_with_template + example['process'].replace(STEP_TAG, STEP_TAG_REAL)
        tokenized_inputs = tokenizer(
            input_text, 
            truncation=True, 
            padding='max_length',
            max_length=max_len
            )
        indices = find_all_indices(tokenized_inputs['input_ids'], step_tag_real_id)
        # example['label'] = example['label'][:len(indices)]
        tokenized_inputs['labels'] = [IGNORE_TOKEN_ID] * len(tokenized_inputs['input_ids'])
        for i, index in enumerate(indices):
            if example['label'][i] in [GOOD_TOKEN, 1]:
                tokenized_inputs['labels'][index] = good_token_id
            elif example['label'][i] in [BAD_TOKEN, 0]:
                tokenized_inputs['labels'][index] = bad_token_id
            else:
                raise ValueError('Invalid label')
            tokenized_inputs['attention_mask'][index] = 0
    
        input_ids.append(tokenized_inputs["input_ids"])
        labels.append(tokenized_inputs["labels"])
        attentionn_masks.append(tokenized_inputs["attention_mask"])
        
    return dict(
        input_ids=torch.tensor(input_ids, dtype=torch.int),
        labels=torch.tensor(labels, dtype=torch.int),
        attention_mask=torch.tensor(attentionn_masks, dtype=torch.int),
    )


class SupervisedDataset(Dataset):
    """SupervisedDataset precomputes everything upfront, 
    leading to faster data access but higher memory usage."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        # sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(raw_data, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """LazySupervisedDataset processes data on demand, using less memory 
    initially but potentially slowing down the first access to each example 
    due to on-the-fly processing."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.raw_data, self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # Load the full dataset
    full_data = json.load(open(data_args.data_path, "r"))

    # Shuffle the data to ensure random distribution
    random.shuffle(full_data)

    # If validation is specified, split the dataset
    if data_args.validation_fraction:
        logging.info("Using validation dataset while training")
        # Split the data into training and validation sets
        assert data_args.validation_fraction <= 1.0, "Validation fraction must be between 0 and 1"
        validation_size = math.ceil(len(full_data) * data_args.validation_fraction)
        eval_data = full_data[:validation_size]
        train_data = full_data[validation_size:]
        # Create datasets
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None
        train_data = full_data

    # Create training dataset
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    preds, labels = eval_pred
    auc = roc_auc_score(labels, preds[:, 1])
    ll = log_loss(labels, preds)
    acc = accuracy_score(labels, preds > 0.5)
    return {"auc": auc, "log_loss": ll, "accuracy": acc}


def preprocess_logits_for_metrics(logits, labels, candidate_tokens):
    """Preprocess logits for custom metrics computation."""
    indices = torch.nonzero(
        (labels == candidate_tokens[0]) | (labels == candidate_tokens[1]), as_tuple=True
    )
    gold_labels = (labels[indices[0], indices[1]] == candidate_tokens[0]).long()
    step_logits = logits[indices[0], indices[1]][:, [candidate_tokens[1], candidate_tokens[0]]]
    return torch.softmax(step_logits, dim=-1)[:, 1], gold_labels


def train():
    global local_rank

    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, WandbArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        wandb_args,
    ) = parser.parse_args_into_dataclasses()

    # Setting wandb logging
    if wandb_args.use_wandb:
        logging.info("Using Wandb for logging training information...")
        if not is_wandb_logged_in():
            logging.info("Wandb login...")
            _ = login(
                key = wandb_args.key,
                relogin = wandb_args.relogin,
                force = wandb_args.force,
                timeout = wandb_args.timeout,
            )

        if len(wandb_args.wandb_run_name)==0:
            wandb_args.wandb_run_name = time.strftime(
                '%Y-%m-%d-%H:%M:%S %p %Z', 
                time.gmtime(time.time())
                )
        training_args.report_to = ["wandb"]
        training_args.run_name = wandb_args.wandb_run_name
        
        os.environ["WANDB_PROJECT"] = wandb_args.wandb_project
        os.environ["WANDB_WATCH"] = wandb_args.wandb_watch
        os.environ["WANDB_LOG_MODEL"] = wandb_args.wandb_log_model

    # TODO To understand this (This serves for single-gpu qlora?)
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    # Setting device map
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        logging.info(f"Device map is {device_map}")
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    # Ensure that ZeRO3 and LoRA are not used at the same time
    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False # TODO: why set it to false?

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=not deepspeed.is_deepspeed_zero3_enabled(),
        quantization_config=GPTQConfig(
            bits=4, 
            disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    # TODO think about why we would ever want to do this 
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = 0 # this is from OpenR
    
    # Setting up LoRA configuration
    if training_args.use_lora:
        # NOTE that if you use LoRA to finetune the base language model, e.g., 
        # Qwen-7B, instead of chat models, e.g., Qwen-7B-Chat, the script automatically 
        # switches the embedding and output layer as trainable parameters. This is 
        # because the base language model has no knowledge of special tokens brought 
        # by ChatML format. Thus these layers should be updated for the model to understand 
        # and predict the tokens. 

        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # However, if the preferred batch size fits into memory, there’s no reason 
        # to apply memory-optimizing techniques because they can slow down the training
        if training_args.gradient_checkpointing:
            # Enables the gradients for the input embeddings. This is useful for fine-tuning 
            # adapter weights while keeping the model weights fixed.
            model.enable_input_require_grads()  

    # Loading data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args, 
        max_len=training_args.model_max_length
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Starting the trainner
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, 
        output_dir=training_args.output_dir, 
        bias=lora_args.lora_bias
        )


if __name__ == "__main__":
    train()