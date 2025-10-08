# 多本体ACT训练指南

## 概述

本指南介绍如何使用修改后的ACT（Action Chunking Transformer）模型进行多本体训练，实现本体泛化。修改后的ACT支持同时训练14维和16维不同本体的数据，实现跨本体泛化。

## 数据集要求

- 支持混合本体维度的数据集
- 当前支持14维和16维本体
- 数据集应包含多个任务，每个任务可能包含不同维度的本体数据
- 示例数据集：`/media/liushengbang/isee/policy/ACT/processed_data/sim-six_tasks/integrated_clean-1200`

### 数据集结构示例
```
sim-six_tasks/integrated_clean-1200/
├── episode_0.hdf5 (14维本体)
├── episode_1.hdf5 (16维本体)
├── episode_2.hdf5 (14维本体)
├── ...
└── episode_1199.hdf5 (16维本体)
```

## 核心修改

### 1. 动态本体维度检测
- 移除了硬编码的`state_dim = 14`
- 自动检测数据集中的本体维度
- 支持混合维度的训练数据

### 2. 数据加载增强
- `utils.py`中的`get_norm_stats()`函数自动检测本体维度
- `load_data()`函数返回检测到的本体维度信息
- 支持不同episode的本体维度差异

### 3. 网络架构适配
- `detr/models/detr_vae.py`中移除了硬编码的本体维度
- 使用`getattr(args, 'state_dim', 14)`实现动态维度适配
- 网络层自动适应输入的本体维度

## 使用方法

### 基本训练命令

```bash
# 使用修改后的训练脚本（推荐）
./train_multibody.sh six_tasks integrated_clean 1200 0 0

# 手动指定参数
python3 imitate_episodes.py \
    --task_name sim-six_tasks \
    --ckpt_dir ./act_ckpt/act-multibody-test \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --seed 0
```

### 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--task_name` | 数据集名称 | sim-six_tasks |
| `--ckpt_dir` | 检查点保存目录 | ./act_ckpt/act-multibody |
| `--policy_class` | 策略类型 | ACT |
| `--kl_weight` | KL散度权重 | 10 |
| `--chunk_size` | 块大小 | 50 |
| `--hidden_dim` | 隐藏层维度 | 512 |
| `--batch_size` | 批次大小 | 8 |
| `--dim_feedforward` | 前馈网络维度 | 3200 |
| `--num_epochs` | 训练轮数 | 6000 |
| `--lr` | 学习率 | 1e-5 |

## 训练过程

### 1. 数据预处理
- 自动检测数据集中的本体维度
- 计算归一化统计信息（均值、方差）
- 分割训练集和验证集（8:2比例）

### 2. 模型初始化
- 根据检测到的本体维度初始化网络
- 支持14维和16维本体的混合训练
- 自动调整网络层以适应不同维度

### 3. 训练循环
- 每个批次可能包含不同维度的本体数据
- 网络自动处理维度差异
- 支持时间聚合和非时间聚合模式

## 本体泛化能力

修改后的ACT具备以下泛化能力：

1. **跨本体维度泛化**：能够在14维和16维本体之间泛化
2. **任务内泛化**：同一任务内的不同本体配置
3. **跨任务泛化**：不同任务间的本体配置差异

## 评估和部署

### 训练评估
- 在训练过程中自动评估验证集性能
- 支持多本体混合验证
- 保存最佳模型检查点

### 模型部署
- 部署时自动检测目标本体的维度
- 支持14维和16维本体的推理
- 保持与原ACT部署接口的兼容性

## 注意事项

1. **数据一致性**：确保不同维度本体的数据格式一致（关节名称、顺序等）
2. **归一化**：系统会为每种维度单独计算归一化统计信息
3. **内存使用**：混合维度训练可能需要更多GPU内存
4. **收敛性**：多本体训练可能需要更长的训练时间达到收敛

## 故障排除

### 常见问题

1. **维度检测失败**
   - 检查HDF5文件格式是否正确
   - 确保observations/qpos和action字段存在且格式正确

2. **内存不足**
   - 减小batch_size
   - 使用更小的chunk_size

3. **训练不收敛**
   - 调整学习率
   - 增加kl_weight
   - 检查数据质量和多样性

## 示例输出

```
Detecting state dimension from dataset...
Detected state dimension: 14
Found multiple state dimensions in dataset: [14, 16]
Training with mixed state dimensions: [14, 16]
...
Epoch 1000: train_loss=0.123, val_loss=0.145
Best model saved at epoch 950
```

## 扩展功能

未来可以扩展支持更多本体维度（如7维、8维等），只需修改维度检测逻辑即可。
