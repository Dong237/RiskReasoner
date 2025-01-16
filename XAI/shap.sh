#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Usage:
#   ./shap.sh
#
# Description:
#   This script demonstrates how to call the shap_demo.py
#   script with the appropriate arguments for:
#       - Training data path
#       - Testing data path
#       - Metrics JSON path
#       - Output metric (the metric to select the best model)
#       - Number of test samples to explain with SHAP
# -----------------------------------------------------------------------------

# Define your paths and parameters here
TRAINING_DATA_PATH="datasets/prior/experts/train_expert.parquet"
TESTING_DATA_PATH="datasets/prior/experts/test_expert_balanced.parquet" 
METRICS_JSON_PATH="results/expert_systems/expert_systems_balanced.json"
OUTPUT_METRIC="ROC_AUC"       # e.g., ROC_AUC, accuracy, F1_score, etc.
NUM_SAMPLES_TO_EXPLAIN=5      # How many test samples to explain with SHAP

# Call the Python script
python explain_shap.py \
    --training_data_path "${TRAINING_DATA_PATH}" \
    --testing_data_path "${TESTING_DATA_PATH}" \
    --metrics_json_path "${METRICS_JSON_PATH}" \
    --output_metric "${OUTPUT_METRIC}" \
    --num_samples_to_explain "${NUM_SAMPLES_TO_EXPLAIN}"
