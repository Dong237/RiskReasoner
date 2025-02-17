# 如果你要限制计算卡编号，请在这里设置，例如只使用 cuda:1-3，如果不用限制，就删除下面这行
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
# export VLLM_USE_MODELSCOPE=True
export MKL_THREADING_LAYER=GNU
export LD_LIBRARY_PATH=/home/brain/anaconda3/envs/alignment/lib/python3.10/site-packages/nvidia/nvjitlink/lib


accelerate launch \
    --num_processes 6 \
    --config_file /data/youxiang/repos/RiskReasoner/training/rl/r1/ds_config_zero3.yaml \
    training/rl/r1/train_grpo.py \
    --config /data/youxiang/repos/RiskReasoner/training/rl/r1/hyperparams.yaml \
    2>&1 | tee /data/youxiang/repos/RiskReasoner/training/rl/r1/riskreasoning.log

    # 1> >(tee logs/ppo/${NUM_PPO_EPOCHS}_ppo_epochs.log) \
    # 2> >(tee logs/ppo/${NUM_PPO_EPOCHS}_ppo_epochs.err >&2)