export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU
export BNB_CUDA_VERSION=121
export LD_LIBRARY_PATH=/home/brain/anaconda3/envs/unsloth/lib/python3.10/site-packages/nvidia/nvjitlink/lib
LOG_FILE="training.log"


python training/rl/grpo/unsloth/train_unsloth.py \
    --config training/rl/grpo/unsloth/hyperp_unsloth.yaml \
    2>&1 | tee training/rl/grpo/unsloth/${LOG_FILE}