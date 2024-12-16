#!/bin/bash

# This bash script starts experiments on every LLM automatically, 
# including CALM model which requires the downloaded LoRA weights 

export CUDA_VISIBLE_DEVICES=1

# List of models to test (excluding the special model)
MODELS=(
    # "/data/youxiang/huggingface/Qwen2.5-Math-7B-Instruct"
    # "/data/youxiang/huggingface/Llama-3.1-8B-Instruct"
    "/data/youxiang/huggingface/Qwen2.5-7B-Instruct"
    # "/data/youxiang/huggingface/Llama-2-7b-chat-hf"
    # "/data/youxiang/huggingface/Qwen2.5-14B-Instruct-GPTQ-Int8"
    # "/data/tangkai/models/Qwen2.5-72B-Instruct-GPTQ-Int4"
)

# Dataset paths
TEST_DATA_PATH="datasets/posterior/test_balanced_posterior.parquet"
TRAIN_DATA_PATH="datasets/posterior/train_posterior.parquet"

# Extract dataset name
DATASET_NAME=$(basename "$TEST_DATA_PATH" .parquet) # Extracts "test_balanced"

# Create results directories if they don't exist
mkdir -p results/llms/inference
mkdir -p results/llms/evaluation

# Loop over models (excluding the special model)
for MODEL_NAME_OR_PATH in "${MODELS[@]}"; do
    # Extract model name
    MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH") # Extracts e.g., "Qwen2.5-Math-7B-Instruct"

    # Set dynamic paths
    INFERENCE_OUTPUT_PATH="results/llms/inference/${MODEL_NAME}_${DATASET_NAME}_cot.json"
    EVALUATION_OUTPUT_PATH="results/llms/evaluation/${MODEL_NAME}_${DATASET_NAME}_cot.json"

    echo "Running experiment for $MODEL_NAME without LoRA weights"
    python3 inference/llm_inference_probs_cot.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --test_data_path "$TEST_DATA_PATH" \
        --train_data_path "$TRAIN_DATA_PATH" \
        --inference_output_path "$INFERENCE_OUTPUT_PATH" \
        --evaluation_output_path "$EVALUATION_OUTPUT_PATH" \
        
done

echo "All experiments completed!"
