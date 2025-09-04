# 多卡训练使用说明

## 🚀 概述

本项目已支持多GPU训练，使用PyTorch Lightning Fabric实现分布式训练。多卡训练可以显著加速训练过程，特别是在处理大规模数据集时。

## 📋 系统要求

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning Fabric
- 多张NVIDIA GPU (推荐2-8张)
- CUDA 11.8+

## 🔧 安装依赖

```bash
pip install lightning
pip install torch torchvision torchaudio
```

## 📁 文件结构

```
policy/DP/
├── train.py                    # 主训练脚本 (已支持多卡)
├── train_multi_gpu.sh         # 多卡训练启动脚本
├── train_six_tasks.sh         # six_tasks专用训练脚本
├── diffusion_policy/
│   ├── config/
│   │   ├── robot_dp_10.yaml   # 10维endpose配置
│   │   ├── robot_dp_14.yaml   # 14维关节配置
│   │   └── robot_dp_16.yaml   # 16维关节配置
│   └── workspace/
│       └── robotworkspace.py  # 训练工作空间 (已支持多卡)
└── data/
    └── six_tasks.zarr         # 合并后的数据集
```

## 🎯 使用方法

### 1. 单卡训练 (原有方式)

```bash
# 使用原有脚本
bash train.sh six_tasks demo_clean 1200 0 14 0

# 或使用专用脚本
bash train_six_tasks.sh demo_clean 1200 0 0
```

### 2. 多卡训练 (新功能)

```bash
# 使用2张GPU训练
bash train_multi_gpu.sh six_tasks demo_clean 1200 0 2

# 使用4张GPU训练
bash train_multi_gpu.sh six_tasks demo_clean 1200 0 4

# 使用所有可用GPU
bash train_multi_gpu.sh six_tasks demo_clean 1200 0 8
```

### 3. 直接调用Python脚本

```bash
# 多卡训练 (自动检测GPU数量)
python train.py --config-name=robot_dp_10.yaml \
                task.name=six_tasks \
                task.dataset.zarr_path="data/six_tasks.zarr" \
                training.seed=0 \
                training.device="cuda:0"
```

## ⚙️ 配置说明

### 自动配置检测

- **six_tasks任务**: 自动检测数据集维度，选择正确的配置文件
  - 10维 → `robot_dp_10.yaml`
  - 14维 → `robot_dp_14.yaml`  
  - 16维 → `robot_dp_16.yaml`

### 环境变量

```bash
export MASTER_ADDR="localhost"      # 主节点地址
export MASTER_PORT="12345"          # 主节点端口
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 指定使用的GPU
```

## 🔍 多卡训练特性

### 1. 分布式数据采样
- 使用 `DistributedSampler` 确保不同GPU处理不同数据
- 自动设置epoch以保持数据一致性

### 2. 梯度同步
- 使用 `fabric.backward()` 自动处理梯度同步
- 支持梯度累积

### 3. 模型并行
- 自动将模型包装为 `DistributedDataParallel`
- 通过 `fabric.setup()` 设置模型和优化器

### 4. 日志和检查点
- 只在rank 0进程上记录日志和保存检查点
- 避免重复输出和文件冲突

## 📊 性能优化建议

### 1. 批次大小调整
```yaml
# 在配置文件中调整
dataloader:
  batch_size: 64  # 多卡时可以减少单卡批次大小
```

### 2. 学习率调整
```yaml
# 多卡训练时通常需要调整学习率
optimizer:
  lr: 2.0e-4  # 单卡1.0e-4，多卡可以适当增加
```

### 3. 数据加载优化
```yaml
dataloader:
  num_workers: 4      # 根据CPU核心数调整
  pin_memory: True    # 启用内存固定
  persistent_workers: True  # 保持worker进程
```

## 🐛 常见问题

### 1. CUDA内存不足
```bash
# 减少批次大小
export CUDA_VISIBLE_DEVICES="0,1"  # 只使用部分GPU
```

### 2. 端口冲突
```bash
# 修改端口号
export MASTER_PORT="12346"
```

### 3. 数据加载慢
```bash
# 增加worker数量
# 检查数据存储位置 (SSD vs HDD)
```

## 📈 性能对比

| GPU数量 | 训练速度 | 内存使用 | 推荐场景 |
|---------|----------|----------|----------|
| 1       | 1x       | 100%     | 调试、小数据集 |
| 2       | 1.8x     | 200%     | 中等数据集 |
| 4       | 3.5x     | 400%     | 大数据集 |
| 8       | 6.5x     | 800%     | 超大数据集 |

## 🎉 开始使用

1. **准备数据**: 运行 `python merge_zarr.py` 合并数据集
2. **选择配置**: 根据数据维度选择配置文件
3. **启动训练**: 使用多卡训练脚本
4. **监控进度**: 查看rank 0进程的输出

```bash
# 完整流程示例
cd /home/shengbang/RoboTwin/policy/DP

# 1. 合并数据
python merge_zarr.py

# 2. 多卡训练
bash train_multi_gpu.sh six_tasks demo_clean 1200 0 4

# 3. 查看输出
tail -f data/outputs/six_tasks-robot_dp-train_seed0/logs.json.txt
```

## 📞 技术支持

如果遇到问题，请检查：
1. GPU驱动和CUDA版本
2. PyTorch和Lightning版本兼容性
3. 数据集路径和权限
4. 系统内存和GPU内存

---

**注意**: 多卡训练需要确保所有GPU具有相同的计算能力，建议使用相同型号的GPU。
