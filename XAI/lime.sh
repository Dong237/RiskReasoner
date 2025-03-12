#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Usage:
#   ./lime.sh
#
# Description:
#   This script demonstrates how to call the explain_lime.py
#   script with the appropriate arguments for:
#       - Training data path
#       - Testing data path
#       - Metrics JSON path
#       - Output metric (the metric to select the best model)
#       - Sample index (the test sample index to explain)
# -----------------------------------------------------------------------------

# Define your paths and parameters here
TRAINING_DATA_PATH="datasets/prior/experts/train_expert.parquet"
TESTING_DATA_PATH="datasets/prior/experts/test_expert_balanced.parquet" 
METRICS_JSON_PATH="results/expert_systems/expert_systems_balanced.json"
OUTPUT_METRIC="ROC_AUC"    # e.g., ROC_AUC, accuracy, F1_score, etc.
SAMPLE_INDEX=0             # Example: explain the first sample in the test set

# Call the Python script
python explain_lime.py \
    --training_data_path "${TRAINING_DATA_PATH}" \
    --testing_data_path "${TESTING_DATA_PATH}" \
    --metrics_json_path "${METRICS_JSON_PATH}" \
    --output_metric "${OUTPUT_METRIC}" \
    --sample_index "${SAMPLE_INDEX}"
