# 如果你要限制计算卡编号，请在这里设置，例如只使用 cuda:1-3，如果不用限制，就删除下面这行
export CUDA_VISIBLE_DEVICES=0,1,2,4,5
export VLLM_USE_MODELSCOPE=True
export MKL_THREADING_LAYER=GNU
export LD_LIBRARY_PATH=/home/brain/anaconda3/envs/pack/lib/python3.9/site-packages/nvidia/nvjitlink/lib
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
# export NCCL_SOCKET_IFNAME=eth0  # Set the correct network interface
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=0
LOG_FILE="training.log"

accelerate launch \
    --main_process_port 0  \
    --num_processes 4 \
    --config_file training/rl/grpo/trl/ds_config_zero3.yaml \
    training/rl/grpo/trl/train_trl.py \
    --config training/rl/grpo/trl/hyperp_trl_3B.yaml \
    2>&1 | tee training/rl/grpo/trl/${LOG_FILE}