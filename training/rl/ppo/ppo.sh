#!/usr/bin/env bash
###############################################################################
# Script for running train.py with your specific PPO
# hyperparameters. Includes argument parsing, environment variable usage,
# date-logging, and optional log-file redirection.
###############################################################################
export CUDA_VISIBLE_DEVICES=1

# Exit on error (e), treat unset variables as errors (u), and fail on any command
# in a pipeline that fails (o pipefail).
set -euo pipefail

###############################################################################
# 1. Parse Command-Line Arguments with Default Values
###############################################################################
# You can override these defaults by providing flags, e.g.:
#   ./run_ppo.sh --dataset_name my_dataset
#
# This approach uses a simple "while" loop parsing each "--key value" pair.

DATASET_NAME="train_posterior"
DATASET_PATH="datasets/posterior/train_posterior.json"
MODEL_NAME_OR_PATH="/data1/huggingface/Qwen2.5-7B-Instruct"
PRM_TYPE="Qwen"
PRM_MODEL_NAME_OR_PATH="/data1/huggingface/Qwen2.5-7B-Instruct"
PRM_LORA_WEIGHTS="model_weights/RiskPRM_v3_lora"
ALGO="APPO"
MINI_BATCH_SIZE=4
PPO_EPOCH=3
NUM_ENV_STEPS=100000
EPISODE_LENGTH=25
MAX_NEW_TOKENS=512
MODEL_MAX_LENGTH=4096
N_ROLLOUT_THREADS=8
CRITIC_LR="5e-5"
LR="1e-6"
SAVE_INTERVAL=50

LOG_FILE=""  # if set, we’ll redirect stdout/stderr to this file

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dataset_name)
      DATASET_NAME="$2"
      shift; shift
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift; shift
      ;;
    --model_name_or_path)
      MODEL_NAME_OR_PATH="$2"
      shift; shift
      ;;
    --prm_type)
      PRM_TYPE="$2"
      shift; shift
      ;;
    --prm_model_name_or_path)
      PRM_MODEL_NAME_OR_PATH="$2"
      shift; shift
      ;;
    --prm_lora_weights)
      PRM_LORA_WEIGHTS="$2"
      shift; shift
      ;;
    --algorithm_name)
      ALGO="$2"
      shift; shift
      ;;
    --mini_batch_size)
      MINI_BATCH_SIZE="$2"
      shift; shift
      ;;
    --ppo_epoch)
      PPO_EPOCH="$2"
      shift; shift
      ;;
    --num_env_steps)
      NUM_ENV_STEPS="$2"
      shift; shift
      ;;
    --episode_length)
      EPISODE_LENGTH="$2"
      shift; shift
      ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
      shift; shift
      ;;
    --model_max_length)
      MODEL_MAX_LENGTH="$2"
      shift; shift
      ;;
    --n_rollout_threads)
      N_ROLLOUT_THREADS="$2"
      shift; shift
      ;;
    --critic_lr)
      CRITIC_LR="$2"
      shift; shift
      ;;
    --lr)
      LR="$2"
      shift; shift
      ;;
    --save_interval)
      SAVE_INTERVAL="$2"
      shift; shift
      ;;
    --log_file)
      LOG_FILE="$2"
      shift; shift
      ;;
    *)  # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

###############################################################################
# 2. Handle Environment Variables for GPU Selection
#    (Optional: only needed if you want to pick which GPU(s) to use)
###############################################################################
# If you want to pin to GPU 0, for example:
# export CUDA_VISIBLE_DEVICES=0
# Or take it from an environment variable, e.g.:
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  # Fallback if not set, we default to GPU 0
  export CUDA_VISIBLE_DEVICES=0
fi

###############################################################################
# 3. Print Summary of the Run
###############################################################################
DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
echo "==================================================================="
echo "Starting train.py at $DATE_STR"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET_NAME ($DATASET_PATH)"
echo "Model Path: $MODEL_NAME_OR_PATH"
echo "PRM: $PRM_TYPE, $PRM_MODEL_NAME_OR_PATH, LoRA: $PRM_LORA_WEIGHTS"
echo "Algorithm: $ALGO"
echo "Mini-Batch: $MINI_BATCH_SIZE, PPO Epoch: $PPO_EPOCH"
echo "Num Env Steps: $NUM_ENV_STEPS, Episode Length: $EPISODE_LENGTH"
echo "Max New Tokens: $MAX_NEW_TOKENS, Model Max Length: $MODEL_MAX_LENGTH"
echo "Rollout Threads: $N_ROLLOUT_THREADS"
echo "Critic LR: $CRITIC_LR, LR: $LR"
echo "Save Interval: $SAVE_INTERVAL"
if [ -n "$LOG_FILE" ]; then
  echo "Will redirect output to: $LOG_FILE"
fi
echo "==================================================================="

###############################################################################
# 4. Optionally Redirect Output to a Log File
###############################################################################
if [ -n "$LOG_FILE" ]; then
  # We use 'exec' to redirect both stdout and stderr to the log file
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

###############################################################################
# 5. Run the Training Script
###############################################################################
python training/rl/ppo/train.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_path "$DATASET_PATH" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --prm_type "$PRM_TYPE" \
    --prm_model_name_or_path "$PRM_MODEL_NAME_OR_PATH" \
    --prm_lora_weights "$PRM_LORA_WEIGHTS" \
    --algorithm_name "$ALGO" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --ppo_epoch "$PPO_EPOCH" \
    --num_env_steps "$NUM_ENV_STEPS" \
    --episode_length "$EPISODE_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --n_rollout_threads "$N_ROLLOUT_THREADS" \
    --critic_lr "$CRITIC_LR" \
    --lr "$LR" \
    --save_interval "$SAVE_INTERVAL" \
    1> >(tee logs/ppo/${NUM_ENV_STEPS}steps.log) \
    2> >(tee logs/ppo/${NUM_ENV_STEPS}steps.err >&2)

###############################################################################
# 6. Done!
###############################################################################
echo "Training finished at $(date +"%Y-%m-%d_%H-%M-%S")"
