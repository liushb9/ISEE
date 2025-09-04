# 六任务数据集合并说明

## 概述
本脚本用于将`/home/shengbang/RoboTwin/policy/ACT/processed_data`下的所有任务数据合并到一个名为`six_tasks`的数据集中，方便训练多任务ACT policy。

## 当前任务数据
根据processed_data目录，当前包含以下6个任务：
- `sim-place_cans_plasticbox` - 放置易拉罐到塑料盒
- `sim-hanging_mug` - 悬挂杯子
- `sim-blocks_ranking_rgb` - 按RGB颜色排序积木
- `sim-blocks_ranking_size` - 按大小排序积木
- `sim-stack_bowls_three` - 堆叠三个碗
- `sim-stack_blocks_three` - 堆叠三个积木

## 使用方法

### 方法1: 使用Shell脚本（推荐）
```bash
cd /home/shengbang/RoboTwin/policy/ACT
./merge_six_tasks.sh
```

### 方法2: 直接运行Python脚本
```bash
cd /home/shengbang/RoboTwin/policy/ACT
python3 merge_six_tasks.py
```

## 脚本功能

### 1. 自动检测任务
- 扫描processed_data目录下的所有子目录
- 自动识别任务名称和配置信息
- 统计每个任务的episode数量

### 2. 数据合并
- 将所有episode文件复制到`sim-six_tasks`目录
- 重新编号episode（从0开始连续编号）
- 保持原始数据完整性

### 3. 配置更新
- 自动更新`SIM_TASK_CONFIGS.json`文件
- 添加`sim-six_tasks`配置项
- 包含总episode数量和相机配置

## 输出结果

### 合并后的数据集
- **位置**: `/home/shengbang/RoboTwin/policy/ACT/processed_data/sim-six_tasks`
- **格式**: 每个episode保存为`episode_X.hdf5`文件
- **内容**: 包含所有6个任务的episode数据

### 配置文件更新
- **文件**: `SIM_TASK_CONFIGS.json`
- **新增配置**: `sim-six_tasks`
- **episode数量**: 所有任务episode的总和

## 训练使用

合并完成后，可以使用以下命令训练ACT policy：

```bash
./train.sh <ckpt_dir> ACT sim-six_tasks <seed> <num_epochs> <lr> <hidden_dim> <dim_feedforward> <chunk_size> <kl_weight>
```

### 参数说明
- `ckpt_dir`: 检查点保存目录
- `ACT`: 策略类型
- `sim-six_tasks`: 任务名称（合并后的数据集）
- `seed`: 随机种子
- `num_epochs`: 训练轮数
- `lr`: 学习率
- `hidden_dim`: 隐藏层维度
- `dim_feedforward`: 前馈网络维度
- `chunk_size`: 块大小
- `kl_weight`: KL散度权重

## 注意事项

1. **数据完整性**: 脚本会复制所有episode文件，确保数据不丢失
2. **目录清理**: 如果`sim-six_tasks`目录已存在，会先删除再重新创建
3. **配置备份**: 建议在运行前备份`SIM_TASK_CONFIGS.json`文件
4. **存储空间**: 确保有足够的磁盘空间存储合并后的数据集

## 故障排除

### 常见问题
1. **权限错误**: 确保脚本有执行权限 `chmod +x merge_six_tasks.sh`
2. **Python依赖**: 确保安装了必要的Python包
3. **路径错误**: 检查processed_data目录路径是否正确

### 调试建议
- 查看脚本输出的详细信息
- 检查生成的日志和错误信息
- 验证合并后的数据集结构

## 示例输出

```
============================================================
ACT Policy 六任务数据集合并脚本
============================================================
找到 6 个任务:
  - sim-place_cans_plasticbox: 50 episodes
  - sim-hanging_mug: 50 episodes
  - sim-blocks_ranking_rgb: 50 episodes
  - sim-blocks_ranking_size: 50 episodes
  - sim-stack_bowls_three: 50 episodes
  - sim-stack_blocks_three: 50 episodes

将合并到: /home/shengbang/RoboTwin/policy/ACT/processed_data/sim-six_tasks
确认继续? (y/N): y

开始合并数据集到: /home/shengbang/RoboTwin/policy/ACT/processed_data/sim-six_tasks

处理任务: sim-place_cans_plasticbox
  - 任务名: place_cans_plasticbox
  - 配置: 50
  - Episode数量: 50
    - 复制 episode_0 -> episode_0
    - 复制 episode_1 -> episode_1
    ...

合并完成!
总episode数量: 300
输出目录: /home/shengbang/RoboTwin/policy/ACT/processed_data/sim-six_tasks

============================================================
数据集合并完成!
============================================================
合并后的数据集包含 300 个episodes
数据集路径: /home/shengbang/RoboTwin/policy/ACT/processed_data/sim-six_tasks

现在可以使用以下命令训练ACT policy:
./train.sh <ckpt_dir> ACT sim-six_tasks <seed> <num_epochs> <lr> <hidden_dim> <dim_feedforward> <chunk_size> <kl_weight>
```
