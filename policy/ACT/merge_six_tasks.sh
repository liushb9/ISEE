#!/bin/bash

# ACT Policy 六任务数据集合并脚本
# 自动合并processed_data下的所有任务数据到six_tasks

echo "============================================================"
echo "ACT Policy 六任务数据集合并脚本"
echo "============================================================"

# 检查Python脚本是否存在
SCRIPT_PATH="./merge_six_tasks.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到合并脚本 $SCRIPT_PATH"
    exit 1
fi

# 检查processed_data目录是否存在
PROCESSED_DATA_DIR="/home/shengbang/RoboTwin/policy/ACT/processed_data"
if [ ! -d "$PROCESSED_DATA_DIR" ]; then
    echo "错误: 找不到processed_data目录 $PROCESSED_DATA_DIR"
    exit 1
fi

# 显示当前的任务数据
echo "当前processed_data目录下的任务:"
ls -la "$PROCESSED_DATA_DIR" | grep "^d"

echo ""
echo "开始合并数据集..."

# 运行Python合并脚本
python3 "$SCRIPT_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "数据集合并成功完成!"
    echo "============================================================"
    echo ""
    echo "合并后的数据集位置:"
    echo "  /home/shengbang/RoboTwin/policy/ACT/processed_data/sim-six_tasks"
    echo ""
    echo "现在可以使用以下命令训练ACT policy:"
    echo "  ./train.sh <ckpt_dir> ACT sim-six_tasks <seed> <num_epochs> <lr> <hidden_dim> <dim_feedforward> <chunk_size> <kl_weight>"
    echo ""
else
    echo ""
    echo "错误: 数据集合并失败!"
    exit 1
fi
