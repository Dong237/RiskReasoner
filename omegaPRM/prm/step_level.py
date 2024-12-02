
import torch
from datasets import load_dataset
import argparse
import os
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)


# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

step_tag = '\n\n'
good_token, bad_token = '+', '-'


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/tangbo/plms/Qwen2.5-7B-Instruct/")
    parser.add_argument("--train_data_path", type=str, default="omegaPRM/prm/test.json")
    parser.add_argument("--val_data_path", type=str, default="omegaPRM/prm/test.json")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--total_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_path", type=str, default="models/risk_prm")
    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return tokenizer, model


def setup_lora(model):
    """Apply LoRA configuration to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    return get_peft_model(model, lora_config)


def preprocess_function(example, tokenizer, step_tag_id, candidate_tokens):
    """Prepare input data for training."""
    input_text = f"{example['question']} {example['process']}"
    tokenized_inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=2048)

    # Find all occurrences of the step_tag_id
    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]

    indices = find_all_indices(tokenized_inputs['input_ids'], step_tag_id)
    example['label'] = example['label'][:len(indices)]

    tokenized_inputs['labels'] = [-100] * len(tokenized_inputs['input_ids'])
    for i, index in enumerate(indices):
        if example['label'][i] in ['+', 1]:
            tokenized_inputs['labels'][index] = candidate_tokens[0]
        elif example['label'][i] in ['-', 0]:
            tokenized_inputs['labels'][index] = candidate_tokens[1]
        else:
            raise ValueError('Invalid label')
        tokenized_inputs['attention_mask'][index] = 0
    return tokenized_inputs


def compute_metrics(eval_pred, candidate_tokens):
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


def preprocess_logits_for_metrics(logits,labels):
    print('aa')
    # return logits,labels
    labels_index = torch.argwhere(torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1]))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1)
    # labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [candidate_tokens[1], candidate_tokens[0]]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold


def main():
    args = parse_arguments()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    # model = setup_lora(model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Define step tag and candidate tokens
    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]

    # Load dataset and preprocess
    data_files = {
    'train': args.train_data_path,
    "test": args.val_data_path, 
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    tokenized_datasets = raw_datasets.map(
        lambda example: preprocess_function(
            example, tokenizer, step_tag_id, candidate_tokens
            )
    )
    tokenized_datasets = {
        split: ds.remove_columns(["question", "process", "label"]) for split, ds in tokenized_datasets.items()
    }

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # Training arguments
    batch_size = args.total_batch_size
    gradient_accumulation_steps = batch_size // args.per_device_train_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        gradient_accumulation_steps //= world_size

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

if __name__ == "__main__":
    main()
