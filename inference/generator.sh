#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=4

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" 

MODEL="/data/youxiang/huggingface/Qwen2.5-7B-Instruct"
DATA="datasets/posterior/split_output_test_balanced_posterior/questions_part_4.json"
OUTPUT_PATH="results/posterior/generator_cot_N/part_4.json"


python3 inference_sys2/inference.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --output_path $OUTPUT_PATH \
    --N 16 \
    --max_new_tokens 2048 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 50 \
    --model_max_length 2048 \