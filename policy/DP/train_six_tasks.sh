#!/bin/bash

# 专门用于训练six_tasks的脚本
# 自动检测数据维度并使用正确的配置

task_name="six_tasks"
task_config=${1:-"demo_clean"}
expert_data_num=${2:-1200}
seed=${3:-0}
gpu_id=${4:-0}

head_camera_type=D435
DEBUG=False
save_ckpt=True

echo "=== 训练 six_tasks 任务 ==="
echo "任务配置: ${task_config}"
echo "数据数量: ${expert_data_num}"
echo "随机种子: ${seed}"
echo "GPU ID: ${gpu_id}"

# 检查数据集是否存在
if [ ! -d "./data/six_tasks.zarr" ]; then
    echo "❌ six_tasks.zarr数据集未找到！"
    echo "请先运行: python merge_zarr.py"
    exit 1
fi

# 检测数据维度
echo "🔍 检测数据集维度..."
actual_dim=$(python -c "
import zarr
import numpy as np
try:
    root = zarr.open('./data/six_tasks.zarr', mode='r')
    if 'data' in root and 'action' in root['data']:
        action_shape = root['data']['action'].shape
        if len(action_shape) >= 2:
            print(action_shape[1])
        else:
            print('10')
    else:
        print('10')
except Exception as e:
    print(f'Error: {e}')
    print('10')
")

echo "检测到action维度: ${actual_dim}"

# 根据维度选择配置
if [ "$actual_dim" = "10" ]; then
    action_dim=10
    config_name="robot_dp_10"
    echo "✅ 使用10维配置 (endpose格式)"
elif [ "$actual_dim" = "14" ]; then
    action_dim=14
    config_name="robot_dp_14"
    echo "⚠️  使用14维配置 (原始关节数据)"
elif [ "$actual_dim" = "16" ]; then
    action_dim=16
    config_name="robot_dp_16"
    echo "⚠️  使用16维配置 (原始关节数据)"
else
    action_dim=10
    config_name="robot_dp_10"
    echo "⚠️  未知维度，使用默认10维配置"
fi

exp_name=${task_name}-robot_dp-train
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m=== 训练配置 ===\033[0m"
echo -e "\033[33m任务名称: ${task_name}\033[0m"
echo -e "\033[33m配置名称: ${config_name}\033[0m"
echo -e "\033[33mAction维度: ${action_dim}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33m数据集路径: data/six_tasks.zarr\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[32m开始训练...\033[0m"
python train.py --config-name=${config_name}.yaml \
                        task.name=${task_name} \
                        task.dataset.zarr_path="data/six_tasks.zarr" \
                        training.debug=$DEBUG \
                        training.seed=${seed} \
                        training.device="cuda:0" \
                        exp_name=${exp_name} \
                        logging.mode=${wandb_mode} \
                        setting=${task_config} \
                        expert_data_num=${expert_data_num} \
                        head_camera_type=$head_camera_type

echo -e "\033[32m训练完成！\033[0m"
