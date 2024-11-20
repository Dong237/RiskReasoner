#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3

# Set variables
# MODEL_NAME_OR_PATH="/data/youxiang/huggingface/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME_OR_PATH="/data/tangbo/plms/Qwen2.5-7B-Instruct/"
# MODEL_NAME_OR_PATH="/data/youxiang/huggingface/Llama-2-7b-chat-hf"
# LORA_WEIGHTS="/data/youxiang/repos/RiskReasoner/lora_weights/CALM"
# MODEL_NAME_OR_PATH="/data/youxiang/huggingface/Llama-3.1-8B-Instruct"
MODEL_NAME_OR_PATH="/data/youxiang/huggingface/Qwen2.5-14B-Instruct-GPTQ-Int8"
DATA_PATH="datasets/llms/test_balanced.parquet"

# Extract model name and dataset name
MODEL_NAME=$(basename "$MODEL_NAME_OR_PATH") # Extracts "Qwen2.5-Math-7B-Instruct"
DATASET_NAME=$(basename "$DATA_PATH" .parquet) # Extracts "test_balanced"

# Set dynamic paths
INFERENCE_OUTPUT_PATH="results/inference/${MODEL_NAME}_${DATASET_NAME}.json"
EVALUATION_OUTPUT_PATH="results/evaluation/${MODEL_NAME}_${DATASET_NAME}.json"


# Print paths for debugging
echo "Inference Output Path: $INFERENCE_OUTPUT_PATH"
echo "Evaluation Output Path: $EVALUATION_OUTPUT_PATH"

# Run inference
python inference/llm_inference_probs.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --data_path "$DATA_PATH" \
    --inference_output_path "$INFERENCE_OUTPUT_PATH" \
    --evaluation_output_path "$EVALUATION_OUTPUT_PATH" \
    # --lora_weights "$LORA_WEIGHTS"