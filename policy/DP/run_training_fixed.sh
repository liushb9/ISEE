#!/bin/bash

echo "Running fixed training script for six_tasks..."

# 检查是否在正确的目录
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the RoboTwin/policy/DP directory"
    exit 1
fi

# 检查数据集是否存在
if [ ! -d "data/six_tasks-demo_clean-300.zarr" ]; then
    echo "Error: Dataset data/six_tasks-demo_clean-300.zarr not found!"
    echo "Please ensure the dataset exists before running training."
    exit 1
fi

echo "Dataset found. Starting training..."

# 运行训练
python train.py --config-name=robot_dp_14.yaml \
    task.name=six_tasks \
    task.dataset.zarr_path="data/six_tasks-demo_clean-300.zarr" \
    training.debug=False \
    training.seed=0 \
    training.device="cuda:0" \
    exp_name=six_tasks-robot_dp-train \
    logging.mode=online \
    setting=demo_clean \
    expert_data_num=300 \
    head_camera_type=D435

echo "Training completed!"