#!/bin/bash

# 单GPU训练脚本
# 用法: bash train_single_gpu.sh <task_name> <setting> <expert_data_num> <seed> <gpu_id>

# 检查参数
if [ $# -ne 5 ]; then
    echo "❌ 参数错误！"
    echo "用法: bash train_single_gpu.sh <task_name> <setting> <expert_data_num> <seed> <gpu_id>"
    echo "示例: bash train_single_gpu.sh six_tasks demo_clean 1200 0 0"
    exit 1
fi

# 解析参数
TASK_NAME=$1
SETTING=$2
EXPERT_DATA_NUM=$3
SEED=$4
GPU_ID=$5

echo "=== 单GPU训练配置 ==="
echo "任务名称: $TASK_NAME"
echo "配置名称: $SETTING"
echo "数据数量: $EXPERT_DATA_NUM"
echo "随机种子: $SEED"
echo "GPU ID: $GPU_ID"

# 检查数据集路径
echo "● 检测数据集路径..."
if [ -d "./data/six-tasks.zarr" ]; then
    echo "✅ 找到数据集: ./data/six-tasks.zarr"
    DATASET_PATH="./data/six-tasks.zarr"
elif [ -d "data/six-tasks.zarr" ]; then
    echo "✅ 找到数据集: data/six-tasks.zarr"
    DATASET_PATH="data/six-tasks.zarr"
elif [ -d "data/six_tasks-${SETTING}-${EXPERT_DATA_NUM}.zarr" ]; then
    echo "✅ 找到数据集: data/six_tasks-${SETTING}-${EXPERT_DATA_NUM}.zarr"
    DATASET_PATH="data/six_tasks-${SETTING}-${EXPERT_DATA_NUM}.zarr"
else
    echo "❌ 未找到数据集！"
    echo "请检查以下路径是否存在："
    echo "  - ./data/six-tasks.zarr"
    echo "  - data/six-tasks.zarr"
    echo "  - data/six_tasks-${SETTING}-${EXPERT_DATA_NUM}.zarr"
    exit 1
fi

# 检查GPU可用性
echo "● 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=, read -r gpu_id gpu_name total_mem free_mem; do
    echo "  GPU $gpu_id: $gpu_name, 总内存: ${total_mem}MB, 可用内存: ${free_mem}MB"
done

# 检查指定GPU是否存在
if ! nvidia-smi -i $GPU_ID > /dev/null 2>&1; then
    echo "❌ GPU $GPU_ID 不存在或不可用！"
    echo "可用GPU列表："
    nvidia-smi --query-gpu=index --format=csv,noheader
    exit 1
fi

echo "✅ 使用GPU: $GPU_ID"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="/home/shengbang/RoboTwin/policy/DP:$PYTHONPATH"

# 创建输出目录
OUTPUT_DIR="outputs/${TASK_NAME}-robot_dp-train-${SETTING}-${EXPERT_DATA_NUM}-seed${SEED}"
mkdir -p $OUTPUT_DIR

echo "● 输出目录: $OUTPUT_DIR"

# 开始训练
echo "🚀 开始单GPU训练..."
echo "使用GPU: $GPU_ID"
echo "数据集: $DATASET_PATH"

# 运行训练命令
python train.py \
    --config-name=six_tasks \
    +task.dataset.zarr_path=$DATASET_PATH \
    training.seed=$SEED \
    training.device=cuda:0 \
    exp_name=${TASK_NAME}-robot_dp-train \
    logging.mode=online \
    setting=$SETTING \
    expert_data_num=$EXPERT_DATA_NUM \
    head_camera_type=D435

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "🎉 训练完成！"
    echo "输出目录: $OUTPUT_DIR"
else
    echo "❌ 训练失败！"
    exit 1
fi
