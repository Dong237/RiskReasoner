#!/bin/bash

# Bash script to run train_xgb.py

# Define paths
TRAIN_PATH="datasets/experts/train_expert.parquet"  # Replace with the actual train parquet file path
TEST_PATH="datasets/experts/test_expert_balanced.parquet"   # Replace with the actual test parquet file path
OUTPUT_PATH="results/expert_systems/expert_systems.json"  # Replace with the desired output path for metrics
ENCODING_THRESHOLD=20

# Execute the Python script
python3 inference/expert_systems.py \
    --training_data_path "$TRAIN_PATH" \
    --testing_data_path "$TEST_PATH" \
    --encoding_threshold "$ENCODING_THRESHOLD" \
    --metrics_output_path "$OUTPUT_PATH" \
    --few_shot 8 \

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Metrics saved to $OUTPUT_PATH."
else
    echo "Error during training. Please check the script and inputs."
fi
