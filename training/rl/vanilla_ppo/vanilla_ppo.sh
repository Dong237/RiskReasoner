#!/bin/bash
############################################################################
# Example: accelerate_ppo.sh
############################################################################
# This script will run your PPO training with Accelerate and DeepSpeed Zero2.
# Customize the environment variables, script arguments, and config as needed.
############################################################################

# Which GPUs to use (if you have multiple). For example: "0,1,2,3".
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,7

# Sometimes helps reduce connection overhead in certain environments
export CUDA_DEVICE_MAX_CONNECTIONS=1

############################################################################
# Paths & Defaults
############################################################################
PPO_SCRIPT="training/rl/ppo.py"            
ACCELERATE_CONFIG_PATH="training/rl/ds_config_zero2.yaml"  # The Accelerate config for DS-ZeRO2

# Script arguments (adjust to match your script's argument parser)
DATASET_PATH="datasets/posterior/train_posterior.json"
MODEL_PATH="/data/youxiang/huggingface/Qwen2.5-7B-Instruct"
OUTPUT_DIR="models/ppo/SparseRM"
EVAL_SIZE=500
STEPS=50000
NUM_PPO_EPOCHS=5

############################################################################
# Optional: Parse command-line arguments
############################################################################
function usage() {
  echo "Usage: bash accelerate_ppo.sh [--model MODEL_PATH] [--data DATA_PATH] [--config ACCELERATE_CONFIG_PATH]"
}

while [[ "$1" != "" ]]; do
  case $1 in
    --model )
      shift
      MODEL_PATH=$1
      ;;
    --data )
      shift
      DATASET_PATH=$1
      ;;
    --config )
      shift
      ACCELERATE_CONFIG_PATH=$1
      ;;
    -h|--help )
      usage
      exit 0
      ;;
    * )
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

############################################################################
# Run Accelerate Launch
############################################################################
accelerate launch --config_file $ACCELERATE_CONFIG_PATH \
  $PPO_SCRIPT \
    --dataset_path $DATASET_PATH \
    --eval_size $EVAL_SIZE \
    --model_name_or_path $MODEL_PATH \
    --steps $STEPS \
    --num_ppo_epochs $NUM_PPO_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --mini_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lora_r 16 \
    --lora_alpha 16 \
    --learning_rate 1e-4 \
    --kl_coef 0.02 \
    --cliprange 0.1 \
    1> >(tee logs/ppo/${NUM_PPO_EPOCHS}_ppo_epochs.log) \
    2> >(tee logs/ppo/${NUM_PPO_EPOCHS}_ppo_epochs.err >&2)
