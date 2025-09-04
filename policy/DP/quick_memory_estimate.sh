#!/bin/bash

echo "=== 快速内存需求估算 ==="
echo ""

# 训练参数
TASK_NAME="six_tasks"
NUM_GPUS=3
GPU_IDS="0,1,2"
BATCH_SIZE=128
SEQUENCE_LENGTH=1
ACTION_DIM=10

echo "📋 训练配置:"
echo "   任务名称: $TASK_NAME"
echo "   GPU数量: $NUM_GPUS"
echo "   GPU ID: $GPU_IDS"
echo "   Batch Size: $BATCH_SIZE"
echo "   序列长度: $SEQUENCE_LENGTH"
echo "   Action维度: $ACTION_DIM"
echo ""

# 检查数据集
echo "📊 检查数据集..."
if [ -d "data/six-tasks.zarr" ]; then
    DATASET_PATH="data/six-tasks.zarr"
    echo "✅ 找到数据集: $DATASET_PATH"
elif [ -d "data/six_tasks-demo_clean-1200.zarr" ]; then
    DATASET_PATH="data/six_tasks-demo_clean-1200.zarr"
    echo "✅ 找到数据集: $DATASET_PATH"
else
    echo "❌ 未找到数据集，使用默认估算"
    DATASET_SIZE_GB=50
fi

# 如果找到数据集，估算大小
if [ -n "$DATASET_PATH" ]; then
    echo "   正在估算数据集大小..."
    # 使用du命令估算目录大小
    DATASET_SIZE_KB=$(du -sk "$DATASET_PATH" 2>/dev/null | cut -f1)
    if [ -n "$DATASET_SIZE_KB" ]; then
        DATASET_SIZE_GB=$(echo "scale=2; $DATASET_SIZE_KB / 1024 / 1024" | bc 2>/dev/null)
        if [ -z "$DATASET_SIZE_GB" ]; then
            DATASET_SIZE_GB=50
        fi
        echo "   数据集大小: ${DATASET_SIZE_GB} GB"
    else
        DATASET_SIZE_GB=50
        echo "   无法获取大小，使用默认: ${DATASET_SIZE_GB} GB"
    fi
fi

echo ""

# 内存需求估算
echo "🧮 内存需求估算:"

# 基础内存需求（GB）
BASE_MEMORY=9.5
echo "   基础内存: ${BASE_MEMORY} GB"
echo "     - 模型权重: 2.0 GB"
echo "     - 优化器状态: 4.0 GB"
echo "     - 梯度: 2.0 GB"
echo "     - 激活值: 1.0 GB"
echo "     - 系统开销: 0.5 GB"

# 数据内存
BATCH_DATA_GB=$(echo "scale=2; $BATCH_SIZE * $SEQUENCE_LENGTH * $ACTION_DIM * 4 / 1024 / 1024 / 1024" | bc 2>/dev/null)
if [ -z "$BATCH_DATA_GB" ]; then
    BATCH_DATA_GB=0.005
fi
CACHED_DATA_GB=$(echo "scale=2; $DATASET_SIZE_GB * 0.1" | bc 2>/dev/null)
if [ -z "$CACHED_DATA_GB" ]; then
    CACHED_DATA_GB=5.0
fi

echo "   数据内存:"
echo "     - Batch数据: ${BATCH_DATA_GB} GB"
echo "     - 缓存数据: ${CACHED_DATA_GB} GB"

# 多卡训练开销
DDP_OVERHEAD_PER_GPU=1.0
TOTAL_DDP_OVERHEAD=$(echo "scale=2; $DDP_OVERHEAD_PER_GPU * $NUM_GPUS" | bc 2>/dev/null)
if [ -z "$TOTAL_DDP_OVERHEAD" ]; then
    TOTAL_DDP_OVERHEAD=3.0
fi

echo "   多卡开销: ${TOTAL_DDP_OVERHEAD} GB"
echo "     - 梯度同步: 0.5 GB × $NUM_GPUS"
echo "     - 进程通信: 0.3 GB × $NUM_GPUS"
echo "     - 重复存储: 0.2 GB × $NUM_GPUS"

# 计算总内存
TOTAL_MEMORY=$(echo "scale=2; $BASE_MEMORY + $BATCH_DATA_GB + $CACHED_DATA_GB + $TOTAL_DDP_OVERHEAD" | bc 2>/dev/null)
if [ -z "$TOTAL_MEMORY" ]; then
    TOTAL_MEMORY=18.0
fi

PER_GPU_MEMORY=$(echo "scale=2; $TOTAL_MEMORY / $NUM_GPUS" | bc 2>/dev/null)
if [ -z "$PER_GPU_MEMORY" ]; then
    PER_GPU_MEMORY=6.0
fi

echo ""
echo "💾 总内存需求:"
echo "   总内存: ${TOTAL_MEMORY} GB"
echo "   每GPU内存: ${PER_GPU_MEMORY} GB"

echo ""

# 检查GPU状态
echo "🔍 检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU显存使用情况:"
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader | while IFS=, read -r index total used free; do
        # 清理输出
        index=$(echo "$index" | tr -d ' ')
        total=$(echo "$total" | tr -d ' MiB')
        used=$(echo "$used" | tr -d ' MiB')
        free=$(echo "$free" | tr -d ' MiB')
        
        # 转换为GB
        total_gb=$(echo "scale=2; $total / 1024" | bc 2>/dev/null)
        used_gb=$(echo "scale=2; $used / 1024" | bc 2>/dev/null)
        free_gb=$(echo "scale=2; $free / 1024" | bc 2>/dev/null)
        
        if [ -n "$total_gb" ] && [ -n "$free_gb" ]; then
            if [ "$index" = "0" ] || [ "$index" = "1" ] || [ "$index" = "2" ]; then
                if (( $(echo "$free_gb >= $PER_GPU_MEMORY" | bc -l) )); then
                    echo "   GPU $index: ✅ 可用 (${free_gb}GB 可用, 需要${PER_GPU_MEMORY}GB)"
                else
                    echo "   GPU $index: ❌ 显存不足 (${free_gb}GB 可用, 需要${PER_GPU_MEMORY}GB)"
                fi
            else
                echo "   GPU $index: 🔒 被占用 (${used_gb}GB 使用中)"
            fi
        fi
    done
else
    echo "⚠️  nvidia-smi 不可用，无法检查GPU状态"
fi

echo ""
echo "📝 估算说明:"
echo "   - 这是基于经验值的粗略估算"
echo "   - 实际内存使用可能因模型架构而异"
echo "   - 建议监控训练过程中的实际内存使用"
echo "   - 如果显存不足，可以减小batch_size或使用gradient checkpointing"

echo ""
echo "🚀 开始训练命令:"
echo "   bash train_multi_gpu.sh $TASK_NAME $CONFIG_NAME $NUM_EPISODES $SEED $NUM_GPUS \"$GPU_IDS\""
