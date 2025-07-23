#!/bin/bash

# Set paths for data and output
TRAINING_DATA_PATH="datasets/prior/experts/train_expert.parquet"
TESTING_DATA_PATH="datasets/prior/experts/test_expert_balanced.parquet"
METRICS_OUTPUT_DIR="results/expert_systems"
METRICS_OUTPUT_PATH="${METRICS_OUTPUT_DIR}/kan_classification_metrics.json"

# Create output directory if it doesn't exist
mkdir -p ${METRICS_OUTPUT_DIR}

# Set encoding threshold
ENCODING_THRESHOLD=5

# KAN model hyperparameters
GRID_SIZE=10
K_VALUE=3
OPTIMIZER="LBFGS"
STEPS=100
VALIDATION_SPLIT=0.2

# Run the Python script with full dataset
echo "Starting KAN model training on full dataset..."
python XAI/kan_classification.py \
  --training_data_path ${TRAINING_DATA_PATH} \
  --testing_data_path ${TESTING_DATA_PATH} \
  --encoding_threshold ${ENCODING_THRESHOLD} \
  --metrics_output_path ${METRICS_OUTPUT_PATH} \
  --grid_size ${GRID_SIZE} \
  --k_value ${K_VALUE} \
  --optimizer ${OPTIMIZER} \
  --steps ${STEPS} \
  --validation_split ${VALIDATION_SPLIT}