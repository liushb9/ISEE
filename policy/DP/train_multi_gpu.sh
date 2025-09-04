#!/bin/bash

# 多卡训练启动脚本
# 使用方法: bash train_multi_gpu.sh [任务名] [配置] [数据量] [种子] [GPU数量] [GPU_IDs]
# 例如: bash train_multi_gpu.sh six_tasks demo_clean 1200 0 3 "0,2,4"

task_name=${1:-"six_tasks"}
task_config=${2:-"demo_clean"}
expert_data_num=${3:-1200}
seed=${4:-0}
num_gpus=${5:-2}  # 默认使用2张GPU
gpu_ids=${6:-""}  # 新增：指定具体的GPU ID，例如 "0,2,4"

head_camera_type=D435
DEBUG=False
save_ckpt=True

echo "=== 多卡训练配置 ==="
echo "任务名称: ${task_name}"
echo "任务配置: ${task_config}"
echo "数据数量: ${expert_data_num}"
echo "随机种子: ${seed}"
echo "GPU数量: ${num_gpus}"
if [ ! -z "$gpu_ids" ]; then
    echo "指定GPU ID: ${gpu_ids}"
fi
echo "=================="

# 检查数据集是否存在
if [ "$task_name" = "six_tasks" ]; then
    # 检查两种可能的路径：连字符和下划线
    if [ -d "./data/six-tasks.zarr" ]; then
        echo "✅ 找到数据集: ./data/six-tasks.zarr"
        dataset_path="./data/six-tasks.zarr"
    elif [ -d "./data/six_tasks.zarr" ]; then
        echo "✅ 找到数据集: ./data/six_tasks.zarr"
        dataset_path="./data/six_tasks.zarr"
    else
        echo "❌ six_tasks.zarr数据集未找到！"
        echo "请检查以下路径："
        echo "  - ./data/six-tasks.zarr"
        echo "  - ./data/six_tasks.zarr"
        echo "请先运行: python merge_zarr.py"
        exit 1
    fi
fi

# 检测可用的GPU数量
available_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "系统可用GPU数量: ${available_gpus}"

# 如果指定了GPU ID，验证其有效性
if [ ! -z "$gpu_ids" ]; then
    # 解析GPU ID列表
    IFS=',' read -ra gpu_array <<< "$gpu_ids"
    specified_gpu_count=${#gpu_array[@]}
    
    echo "指定的GPU ID: ${gpu_ids}"
    echo "指定GPU数量: ${specified_gpu_count}"
    
    # 验证指定的GPU ID是否有效
    for gpu_id in "${gpu_array[@]}"; do
        if [ "$gpu_id" -ge "$available_gpus" ] || [ "$gpu_id" -lt 0 ]; then
            echo "❌ 错误: GPU ID ${gpu_id} 超出范围 [0, $((available_gpus-1))]"
            exit 1
        fi
    done
    
    # 更新GPU数量为实际指定的数量
    num_gpus=$specified_gpu_count
    echo "✅ 使用指定的 ${num_gpus} 张GPU: ${gpu_ids}"
else
    # 没有指定GPU ID，使用默认逻辑
    if [ $num_gpus -gt $available_gpus ]; then
        echo "⚠️  请求的GPU数量 (${num_gpus}) 超过可用数量 (${available_gpus})"
        echo "将使用所有可用GPU: ${available_gpus}"
        num_gpus=$available_gpus
        # 生成连续的GPU ID列表
        gpu_ids=$(seq -s, 0 $((available_gpus-1)))
    else
        # 生成连续的GPU ID列表
        gpu_ids=$(seq -s, 0 $((num_gpus-1)))
    fi
fi

# 检测数据维度
if [ "$task_name" = "six_tasks" ]; then
    echo "🔍 检测数据集维度..."
    actual_dim=$(python -c "
import zarr
import numpy as np
try:
    # 尝试连字符路径
    root = zarr.open('./data/six-tasks.zarr', mode='r')
    if 'data' in root and 'action' in root['data']:
        action_shape = root['data']['action'].shape
        if len(action_shape) >= 2:
            print(action_shape[1])
        else:
            print('10')
    else:
        print('10')
except Exception as e:
    try:
        # 尝试下划线路径
        root = zarr.open('./data/six_tasks.zarr', mode='r')
        if 'data' in root and 'action' in root['data']:
            action_shape = root['data']['action'].shape
            if len(action_shape) >= 2:
                print(action_shape[1])
            else:
                print('10')
        else:
            print('10')
    except Exception as e2:
        print(f'Error: {e2}')
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
else
    # 对于其他任务，使用默认配置
    action_dim=14
    config_name="robot_dp_14"
fi

exp_name=${task_name}-robot_dp-train
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m=== 训练配置 ==="
echo -e "\033[33m任务名称: ${task_name}\033[0m"
echo -e "\033[33m配置名称: ${config_name}\033[0m"
echo -e "\033[33mAction维度: ${action_dim}\033[0m"
echo -e "\033[33mGPU数量: ${num_gpus}\033[0m"
echo -e "\033[33m使用GPU ID: ${gpu_ids}\033[0m"
echo -e "\033[33m数据集路径: data/${task_name}-${task_config}-${expert_data_num}.zarr\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_ids}

echo -e "\033[32m开始多卡训练...\033[0m"
echo -e "\033[32m使用GPU: ${CUDA_VISIBLE_DEVICES}\033[0m"
echo -e "\033[32m实际GPU数量: ${num_gpus}\033[0m"

# 启动多卡训练
python train.py --config-name=${config_name}.yaml \
                        task.name=${task_name} \
                        +task.dataset.zarr_path="${dataset_path}" \
                        training.debug=$DEBUG \
                        training.seed=${seed} \
                        training.device="cuda:0" \
                        exp_name=${exp_name} \
                        logging.mode=${wandb_mode} \
                        setting=${task_config} \
                        expert_data_num=${expert_data_num} \
                        head_camera_type=$head_camera_type

echo -e "\033[32m多卡训练完成！\033[0m"
