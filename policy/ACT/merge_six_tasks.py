#!/usr/bin/env python3
"""
合并六个任务的数据集到six_tasks
将/home/shengbang/RoboTwin/policy/ACT/processed_data下的所有数据合并到一起
"""

import os
import shutil
import h5py
import numpy as np
from pathlib import Path
import json

def get_task_configs():
    """获取所有任务配置"""
    processed_data_dir = "/home/shengbang/RoboTwin/policy/ACT/processed_data"
    task_configs = {}
    
    for task_dir in os.listdir(processed_data_dir):
        task_path = os.path.join(processed_data_dir, task_dir)
        if os.path.isdir(task_path):
            # 从目录名中提取任务名
            # 实际格式: sim-task_name (只有2个部分)
            parts = task_dir.split('-')
            if len(parts) >= 2 and parts[0] == 'sim':
                task_name = '-'.join(parts[1:])  # 例如: place_cans_plasticbox
                
                # 查找包含episode文件的子目录
                episode_count = 0
                actual_data_dir = None
                
                # 检查是否有demo_clean-50子目录
                demo_dir = os.path.join(task_path, "demo_clean-50")
                if os.path.exists(demo_dir):
                    actual_data_dir = demo_dir
                    # 统计episode数量
                    for file in os.listdir(demo_dir):
                        if file.startswith('episode_') and file.endswith('.hdf5'):
                            episode_count += 1
                
                # 如果找到episode文件，则添加到配置中
                if episode_count > 0 and actual_data_dir:
                    task_configs[task_dir] = {
                        "dataset_dir": actual_data_dir,  # 使用实际的episode文件目录
                        "num_episodes": episode_count,
                        "episode_len": 1000,  # 默认值，可根据实际情况调整
                        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
                        "task_name": task_name,
                        "config_name": "merged"  # 使用固定配置名
                    }
                    print(f"找到任务: {task_dir} -> {episode_count} episodes (在 {actual_data_dir})")
                else:
                    print(f"警告: {task_dir} 目录中没有找到episode文件")
    
    return task_configs

def merge_datasets(task_configs, output_dir):
    """合并所有数据集"""
    print(f"开始合并数据集到: {output_dir}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    total_episodes = 0
    merged_episode_id = 0
    
    # 遍历每个任务
    for task_key, task_config in task_configs.items():
        print(f"\n处理任务: {task_key}")
        print(f"  - 任务名: {task_config['task_name']}")
        print(f"  - Episode数量: {task_config['num_episodes']}")
        
        task_dir = task_config['dataset_dir']
        
        # 处理每个episode
        for episode_id in range(task_config['num_episodes']):
            source_file = os.path.join(task_dir, f"episode_{episode_id}.hdf5")
            target_file = os.path.join(output_dir, f"episode_{merged_episode_id}.hdf5")
            
            if os.path.exists(source_file):
                # 复制文件
                shutil.copy2(source_file, target_file)
                print(f"    - 复制 episode_{episode_id} -> episode_{merged_episode_id}")
                merged_episode_id += 1
                total_episodes += 1
            else:
                print(f"    - 警告: 文件不存在 {source_file}")
    
    print(f"\n合并完成!")
    print(f"总episode数量: {total_episodes}")
    print(f"输出目录: {output_dir}")
    
    return total_episodes

def create_merged_config(task_configs, total_episodes, output_dir):
    """创建合并后的配置文件"""
    config = {
        "sim-six_tasks": {
            "dataset_dir": output_dir,
            "num_episodes": total_episodes,
            "episode_len": 1000,
            "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        }
    }
    
    # 保存到SIM_TASK_CONFIGS.json
    config_file = "/home/shengbang/RoboTwin/policy/ACT/SIM_TASK_CONFIGS.json"
    
    # 如果文件已存在，读取并更新
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                existing_config = json.load(f)
        except:
            existing_config = {}
    else:
        existing_config = {}
    
    # 更新配置
    existing_config.update(config)
    
    # 保存配置
    with open(config_file, "w") as f:
        json.dump(existing_config, f, indent=4)
    
    print(f"配置文件已更新: {config_file}")
    print("新增配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

def main():
    """主函数"""
    print("=" * 60)
    print("ACT Policy 六任务数据集合并脚本")
    print("=" * 60)
    
    # 获取所有任务配置
    task_configs = get_task_configs()
    
    if not task_configs:
        print("错误: 未找到任何任务数据!")
        return
    
    print(f"找到 {len(task_configs)} 个任务:")
    for task_key, config in task_configs.items():
        print(f"  - {task_key}: {config['num_episodes']} episodes")
    
    # 设置输出目录
    output_dir = "/home/shengbang/RoboTwin/policy/ACT/processed_data/sim-six_tasks"
    
    # 确认操作
    print(f"\n将合并到: {output_dir}")
    confirm = input("确认继续? (y/N): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return
    
    # 合并数据集
    total_episodes = merge_datasets(task_configs, output_dir)
    
    # 创建配置文件
    create_merged_config(task_configs, total_episodes, output_dir)
    
    print("\n" + "=" * 60)
    print("数据集合并完成!")
    print("=" * 60)
    print(f"合并后的数据集包含 {total_episodes} 个episodes")
    print(f"数据集路径: {output_dir}")
    print("\n现在可以使用以下命令训练ACT policy:")
    print(f"./train.sh <ckpt_dir> ACT sim-six_tasks <seed> <num_epochs> <lr> <hidden_dim> <dim_feedforward> <chunk_size> <kl_weight>")

if __name__ == "__main__":
    main()
