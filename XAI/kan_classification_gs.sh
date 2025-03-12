#!/bin/bash

# Set paths for data and output
TRAINING_DATA_PATH="datasets/prior/experts/train_expert.parquet"
TESTING_DATA_PATH="datasets/prior/experts/test_expert_balanced.parquet"
BEST_METRICS_OUTPUT_PATH="results/expert_systems/kan_metrics_best_triple_layers.json"
ALL_METRICS_OUTPUT_PATH="results/expert_systems/kan_metrics_all_triple_layers.json"

# Create output directory if it doesn't exist
mkdir -p results/expert_systems

# Set encoding threshold and other parameters
ENCODING_THRESHOLD=5
KAN_SEARCH_LAYER=3
GRID_SIZE_LOWER=5
GRID_SIZE_UPPER=50
K_LOWER=1
K_UPPER=10
OPTIMIZER="LBFGS"
STEPS=100
VALIDATION_SPLIT=0.2

# Run the Python script with full dataset
echo "Starting KAN model training with grid search on full dataset..."
python XAI/kan_classification_grid_search.py \
  --training_data_path "${TRAINING_DATA_PATH}" \
  --testing_data_path "${TESTING_DATA_PATH}" \
  --encoding_threshold "${ENCODING_THRESHOLD}" \
  --best_metrics_output_path "${BEST_METRICS_OUTPUT_PATH}" \
  --all_metrics_output_path "${ALL_METRICS_OUTPUT_PATH}" \
  --kan_search_layer "${KAN_SEARCH_LAYER}" \
  --grid_size_lower "${GRID_SIZE_LOWER}" \
  --grid_size_upper "${GRID_SIZE_UPPER}" \
  --k_lower "${K_LOWER}" \
  --k_upper "${K_UPPER}" \
  --optimizer "${OPTIMIZER}" \
  --steps "${STEPS}" \
  --validation_split "${VALIDATION_SPLIT}"

echo "KAN model training completed."