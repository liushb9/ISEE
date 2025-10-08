#!/bin/bash
# Multi-body ACT training script for six tasks with mixed state dimensions (14D and 16D)

# Use the correct task name that matches SIM_TASK_CONFIGS.json
TASK_NAME="sim-six_tasks-integrated_clean-1200"
SEED=${1:-0}
GPU_ID=${2:-0}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "Starting multi-body ACT training..."
echo "Task: $TASK_NAME"
echo "Seed: $SEED"
echo "GPU: $GPU_ID"

python3 imitate_episodes.py \
    --task_name $TASK_NAME \
    --ckpt_dir ./act_ckpt/act-multibody-seed${SEED} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 2000 \
    --seed ${SEED}

echo "Training completed!"
