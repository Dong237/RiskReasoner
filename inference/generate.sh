#!/bin/bash

# Set CUDA environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Define paths for the model, input data, and output results
GENERATOR_TYPE="GeneratorCoT"
MODEL="/data/youxiang/huggingface/Qwen2.5-7B-Instruct"
DATA="datasets/posterior/train_posterior.json"
OUTPUT_PATH="datasets/generator/train_posterior_generator_cot.json"


# Run the inference script with the GeneratorCoTN type and required arguments
python3 inference/generate.py \
    --generator_type $GENERATOR_TYPE \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --output_path $OUTPUT_PATH \
    --batch_size 16 \
    --max_new_tokens 2048 \
    --model_max_length 2048 \
    --generation_strategy "greedy"
