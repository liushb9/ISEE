#!/bin/bash

# 带调试信息的运行脚本
# 设置环境变量来控制输出详细程度

echo "🔧 数据处理脚本 - 调试模式"
echo "================================"

# 设置调试环境变量
export DEBUG_JOINT_DATA=1    # 显示关节数据详细信息
export DEBUG_HDF5=1          # 显示HDF5文件结构详细信息

echo "✅ 已启用调试模式:"
echo "  - DEBUG_JOINT_DATA=1 (显示关节数据详情)"
echo "  - DEBUG_HDF5=1 (显示HDF5结构详情)"
echo ""

# 运行数据处理
echo "🚀 开始运行数据处理..."
bash process_data.sh "$@"

echo ""
echo "✅ 处理完成！"
echo "如需关闭调试信息，请直接运行: bash process_data.sh"
