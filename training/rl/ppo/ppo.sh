#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python -u training/rl/ppo/train.py --seed 10 \
                        --dataset_name "prealgebra" \
                        --dataset_path "/data/youxiang/repos/openr/envs/MATH/dataset/test500.jsonl" \
                        --model_name_or_path "/data/youxiang/huggingface/Qwen2.5-Math-7B-Instruct" \
                        --prm_type "MS" \
                        --prm_model_name_or_path "/data/youxiang/huggingface/Qwen2.5-Math-7B-Instruct" \
                        --prm_checkpoint_path "CHECKPOINT_PATH" \
                        --algorithm_name "APPO" \
                        --experiment_name "ms_single" \
                        --num_mini_batch 4 \
                        --ppo_epoch 1