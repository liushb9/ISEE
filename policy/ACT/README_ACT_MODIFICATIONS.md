# ACT Policy 修改说明

## 概述
本次修改为ACT policy添加了CLIP文本编码功能，使模型能够理解任务描述并生成相应的动作。

## 主要修改内容

### 1. process_data.py
- 添加了CLIP模型导入和`text2feats`函数
- 在数据处理过程中，将任务名转换为text_feat
- 将text_feat保存到HDF5文件的observations组中
- **图像尺寸**: 将所有相机图像resize为256x256像素

### 2. utils.py
- 修改了`EpisodicDataset`类，在`__getitem__`方法中加载text_feat
- 修改了`get_norm_stats`函数，不归一化text_feat
- 数据加载器现在返回5个元素：image_data, qpos_data, action_data, is_pad, text_feat_data
- **图像尺寸**: 确保加载的图像都是256x256像素

### 3. detr_vae.py
- 在`DETRVAE`类中添加了`text_feat_proj` MLP层
- 修改了forward方法，将text_feat处理后与proprio_input拼接
- 在`CNNMLP`类中也添加了类似的text_feat处理
- **动态适配**: 支持1024维和512维的text_feat输入

### 4. act_policy.py
- 修改了`ACTPolicy`和`CNNMLPPolicy`类的`__call__`方法
- 添加了text_feat参数

### 5. imitate_episodes.py
- 添加了CLIP模型导入和初始化
- 修改了`forward_pass`函数，处理text_feat
- 在评估循环中，从任务名生成text_feat并传递给policy
- **图像尺寸**: 确保评估时的图像也是256x256像素

### 6. conda_env.yaml
- 添加了CLIP相关的依赖包

## 使用方法

### 1. 环境设置
```bash
cd policy/ACT
conda env create -f conda_env.yaml
conda activate aloha
```

### 2. 数据处理
```bash
# 处理数据，生成包含text_feat的HDF5文件
./process_data.sh <task_name> <task_config> <expert_data_num>

# 例如：
./process_data.sh beat_block_hammer config1 50
```

### 3. 训练
```bash
# 训练模型
./train.sh <ckpt_dir> <policy_class> <task_name> <seed> <num_epochs> <lr> <hidden_dim> <dim_feedforward> <chunk_size> <kl_weight>
```

### 4. 评估
```bash
# 评估模型
./eval.sh <ckpt_dir> <policy_class> <task_name> <seed> <num_epochs> <lr> <hidden_dim> <dim_feedforward> <chunk_size> <kl_weight>
```

## 技术细节

### text_feat处理流程
1. **任务名预处理**: 将下划线替换为空格（如：`beat_block_hammer` → `beat block hammer`）
2. **CLIP编码**: 使用CLIP RN50模型将文本转换为512维特征向量
3. **MLP处理**: 通过MLP层将512维特征映射到模型的hidden_dim
4. **特征融合**: 将处理后的text_feat与proprio_input相加，实现特征融合

### 数据流
```
任务名 → CLIP编码 → text_feat → MLP处理 → 与视觉特征拼接 → 输出动作
```

## 注意事项

1. **CLIP模型**: 使用RN50作为文本编码器，输出维度为512
2. **text_feat归一化**: text_feat不进行归一化处理，保持原始特征
3. **内存使用**: 添加text_feat会增加一定的内存使用量
4. **兼容性**: 修改后的代码保持向后兼容，text_feat参数为可选参数

## 故障排除

### 常见问题
1. **CLIP模型加载失败**: 确保安装了正确的CLIP版本
2. **内存不足**: 减少batch_size或使用更小的模型
3. **text_feat维度不匹配**: 检查CLIP模型输出维度是否为512

### 调试建议
- 在process_data.py中添加print语句，检查text_feat的生成
- 在训练过程中监控text_feat的梯度流动
- 验证text_feat是否正确传递到模型的各个组件
