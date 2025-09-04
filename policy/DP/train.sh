#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

head_camera_type=D435

DEBUG=False
save_ckpt=True

# 自动检测数据维度
if [ "$1" = "six_tasks" ]; then
    # 对于six_tasks，检查数据集中的实际维度
    if [ -d "./data/six_tasks.zarr" ]; then
        echo "检测six_tasks.zarr数据集..."
        # 使用Python检测实际维度
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
            print('10')  # 默认值
    else:
        print('10')  # 默认值
except:
    print('10')  # 默认值
")
        echo "检测到action维度: ${actual_dim}"
        
        if [ "$actual_dim" = "10" ]; then
            action_dim=10
            echo "✅ 使用10维配置 (endpose格式)"
        elif [ "$actual_dim" = "14" ]; then
            action_dim=14
            echo "⚠️  使用14维配置 (原始关节数据)"
        elif [ "$actual_dim" = "16" ]; then
            action_dim=16
            echo "⚠️  使用16维配置 (原始关节数据)"
        else
            action_dim=10
            echo "⚠️  未知维度，使用默认10维配置"
        fi
    else
        echo "❌ six_tasks.zarr数据集未找到！"
        exit 1
    fi
fi

alg_name=robot_dp_$action_dim
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33maction_dim: ${action_dim}\033[0m"
echo -e "\033[33mconfig: ${config_name}.yaml\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 特殊处理six_tasks任务，使用已存在的six_tasks.zarr数据集
if [ "$1" = "six_tasks" ]; then
    if [ ! -d "./data/six_tasks.zarr" ]; then
        echo "Error: six_tasks.zarr dataset not found!"
        exit 1
    fi
    echo "Using existing six_tasks.zarr dataset for training..."
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
    exit 0
fi

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \