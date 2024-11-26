#!/bin/bash

# Bash script to run preprocessor/posterior.py with specified arguments
# Meaning of each arguments can be found at the begining of the Python script

# Define the path to the Python script
PROGRAM="preprocessor/posterior.py"

# Define the arguments
EXPERT_SYSTEMS_RESULT="/data/youxiang/repos/RiskReasoner/results/expert_systems/expert_systems_balanced.json"
METRIC="KS_score"
DATA_PATH="datasets"
ENCODING_THRESHOLD="20"

# Run the Python program
python "$PROGRAM" \
    --expert_systems_result "$EXPERT_SYSTEMS_RESULT" \
    --metric "$METRIC" \
    --data_path "$DATA_PATH" \
    --encoding_threshold "$ENCODING_THRESHOLD"
