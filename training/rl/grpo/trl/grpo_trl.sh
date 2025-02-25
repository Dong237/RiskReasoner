# 如果你要限制计算卡编号，请在这里设置，例如只使用 cuda:1-3，如果不用限制，就删除下面这行
export CUDA_VISIBLE_DEVICES=0,1,2
# export VLLM_USE_MODELSCOPE=True
export MKL_THREADING_LAYER=GNU
export LD_LIBRARY_PATH=/home/brain/anaconda3/envs/alignment/lib/python3.10/site-packages/nvidia/nvjitlink/lib
LOG_FILE="training.log"

accelerate launch \
    --num_processes 2 \
    --config_file training/rl/r1/trl/ds_config_zero3.yaml \
    training/rl/r1/train_trl.py \
    --config training/rl/grpo/trl/hyperp_trl_3B.yaml \
    2>&1 | tee training/rl/grpo/trl/${LOG_FILE}

    # 1> >(tee logs/ppo/${NUM_PPO_EPOCHS}_ppo_epochs.log) \
    # 2> >(tee logs/ppo/${NUM_PPO_EPOCHS}_ppo_epochs.err >&2)