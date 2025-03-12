#!/bin/bash

# Set the model and other parameters
OUTPUT_DIR="datasets/omegaPRM"
MODEL_NAME="/data/tangbo/plms/Qwen2.5-7B-Instruct/"
MODEL_TYPE="vllm"
DEVICE="cuda"
MAX_NEW_TOKENS=2048
TEMPERATURE=1
C_PUCT=0.125
ALPHA=0.5
BETA=0.9
LENGTH_SCALE=500
NUM_ROLLOUTS=16
MAX_SEARCH_COUNT=20
ROLLOUT_BUDGET=320
SAVE_DATA_TYPE="both"
LOG_FILE_PREFIX="logs/omega_prm_single_gpu"

# Split files directory
SPLIT_DIR="datasets/posterior/split_output"

# # Create output directory if it doesn't exist
# mkdir -p $OUTPUT_DIR
# mkdir -p  log

# Start the OmegaPRM process on each GPU with separate split files
# for i in {5,7} # Do not forget to change the input data file names accordingly
for i in 2; do
    SPLIT_FILE="$SPLIT_DIR/questions_part_${i}.json"
    GPU_ID=$i  # $((i-1))
    OUTPUT_DIR_GPU="${OUTPUT_DIR}_gpu${GPU_ID}"
    mkdir -p "$OUTPUT_DIR_GPU"  # Create the output directory for this GPU
    LOG_FILE_PREFIX="log/omega_prm_gpu_$GPU_ID"

    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Using GPU ${i}."
    # Run the OmegaPRM process in the background on the specified GPU
    # CUDA_VISIBLE_DEVICES=$GPU_ID 
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 omegaPRM/run_omegaprm.py \
        --question_file "$SPLIT_FILE" \
        --output_dir "$OUTPUT_DIR_GPU" \
        --model_name "$MODEL_NAME" \
        --model_type "$MODEL_TYPE" \
        --device "$DEVICE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --c_puct "$C_PUCT" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --length_scale "$LENGTH_SCALE" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --max_search_count "$MAX_SEARCH_COUNT" \
        --rollout_budget "$ROLLOUT_BUDGET" \
        --save_data_type "$SAVE_DATA_TYPE" \
        --log_file_prefix "$LOG_FILE_PREFIX" &  # & at the end of the command places the process into the background
done

# Wait for all processes to finish
wait

echo "All OmegaPRM processes complete."
