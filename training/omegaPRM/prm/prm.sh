#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
DIR=`pwd`

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The IP address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="/data/youxiang/huggingface/Qwen2.5-7B-Instruct" # "/data/tangbo/plms/Qwen2.5-7B-Instruct-GPTQ-Int8"
DATA="datasets/omegaPRM_v2/risk_reasoner_v2.jsonl"
OUTPUT_DIR="models/RiskPRM_v2_full"
DS_CONFIG_PATH="omegaPRM/prm/ds_config_zero3.json"
WANDB_KEY="e28afd6154b7ecd865dde62fead55bba5994bc9a"

DISTRIBUTED_ARGS="\
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

echo "Distributed Arguments: $DISTRIBUTED_ARGS"


# non-lazy processing of data + logging with tensorboard
torchrun $DISTRIBUTED_ARGS omegaPRM/prm/step_level_qwen.py \
    --model_name_or_path $MODEL \
    --hidden_dropout_prob 0.1 \
    --attention_probs_dropout_prob 0.1 \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --validation_fraction 0.1 \
    --logging_strategy "steps" \
    --logging_steps 30 \
    --eval_strategy "steps" \
    --eval_steps 30 \
    --save_strategy "steps" \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048 \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH}
    # --use_lora \
    # --key $WANDB_KEY \
    # --use_wandb True \
    # --wandb_run_name "prm-run1" \
    
