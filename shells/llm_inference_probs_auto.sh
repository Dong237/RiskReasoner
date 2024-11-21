#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3

# List of models to test
MODELS=(
    "/data/youxiang/huggingface/Qwen2.5-Math-7B-Instruct"
    "/data/tangbo/plms/Qwen2.5-7B-Instruct/"
    "/data/youxiang/huggingface/Llama-2-7b-chat-hf"
    "/data/youxiang/huggingface/Qwen2.5-14B-Instruct-GPTQ-Int8"
    "/data/tangkai/models/Qwen2.5-72B-Instruct-GPTQ-Int4"
)

# Special case for model with LoRA weights
SPECIAL_MODEL="/data/youxiang/huggingface/Llama-2-7b-chat-hf"
LORA_WEIGHTS="/data/youxiang/repos/RiskReasoner/lora_weights/CALM"

# Dataset
DATA_PATH="datasets/llms/test_balanced.parquet"

# Extract dataset name
DATASET_NAME=$(basename "$DATA_PATH" .parquet) # Extracts "test_balanced"

# Create results directories if they don't exist
mkdir -p results/inference
mkdir -p results/evaluation

# Loop over models
for MODEL_NAME_OR_PATH in "${MODELS[@]}"; do
    # Extract model name
    MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH") # Extracts e.g., "Qwen2.5-Math-7B-Instruct"

    # Set dynamic paths
    INFERENCE_OUTPUT_PATH="results/inference/${MODEL_NAME}_${DATASET_NAME}.json"
    EVALUATION_OUTPUT_PATH="results/evaluation/${MODEL_NAME}_${DATASET_NAME}.json"

    # Check if this is the special case with LoRA weights
    if [ "$MODEL_NAME_OR_PATH" == "$SPECIAL_MODEL" ]; then
        echo "Running experiment for $MODEL_NAME with LoRA weights: $LORA_WEIGHTS"
        python inference/llm_inference_probs.py \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --data_path "$DATA_PATH" \
            --inference_output_path "$INFERENCE_OUTPUT_PATH" \
            --evaluation_output_path "$EVALUATION_OUTPUT_PATH" \
            --lora_weights "$LORA_WEIGHTS"
    else
        echo "Running experiment for $MODEL_NAME without LoRA weights"
        python inference/llm_inference_probs.py \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --data_path "$DATA_PATH" \
            --inference_output_path "$INFERENCE_OUTPUT_PATH" \
            --evaluation_output_path "$EVALUATION_OUTPUT_PATH"
    fi
done

echo "All experiments completed!"
