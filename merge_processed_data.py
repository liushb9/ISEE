#!/usr/bin/env python3
"""
合并6个任务的processed_data到新的文件夹中
将episode重新编号为0-1199
"""

import os
import shutil
import glob
from pathlib import Path

def merge_processed_data():
    """合并6个任务的processed_data"""
    
    # 源目录和目标目录
    source_base = "/media/liushengbang/RoboTwin_baseline/policy/RDT/processed_data"
    target_dir = "/media/liushengbang/RoboTwin_baseline/policy/RDT/processed_data/rdt_1200-integrated_clean-1200"
    
    # 6个任务目录
    task_dirs = [
        "blocks_ranking_rgb-integrated_clean-200",
        "blocks_ranking_size-integrated_clean-200", 
        "hanging_mug-integrated_clean-200",
        "place_cans_plasticbox-integrated_clean-200",
        "stack_blocks_three-integrated_clean-200",
        "stack_bowls_three-integrated_clean-200"
    ]
    
    print("开始合并processed_data...")
    print(f"目标目录: {target_dir}")
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    episode_counter = 0
    total_episodes = 0
    
    # 处理每个任务
    for task_dir in task_dirs:
        source_task_dir = os.path.join(source_base, task_dir)
        
        if not os.path.exists(source_task_dir):
            print(f"❌ 源目录不存在: {source_task_dir}")
            continue
            
        print(f"\n处理任务: {task_dir}")
        
        # 获取所有episode目录
        episode_dirs = glob.glob(os.path.join(source_task_dir, "episode_*"))
        episode_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        
        print(f"找到 {len(episode_dirs)} 个episode")
        
        # 复制每个episode
        for source_episode_dir in episode_dirs:
            # 新的episode编号
            new_episode_name = f"episode_{episode_counter}"
            target_episode_dir = os.path.join(target_dir, new_episode_name)
            
            # 复制整个episode目录
            shutil.copytree(source_episode_dir, target_episode_dir)
            
            # 重命名HDF5文件
            old_hdf5 = os.path.join(target_episode_dir, f"episode_{episode_counter % 200}.hdf5")
            new_hdf5 = os.path.join(target_episode_dir, f"episode_{episode_counter}.hdf5")
            
            if os.path.exists(old_hdf5):
                os.rename(old_hdf5, new_hdf5)
                print(f"  ✓ 复制 episode_{episode_counter % 200} -> episode_{episode_counter}")
            else:
                print(f"  ⚠️  HDF5文件不存在: {old_hdf5}")
            
            episode_counter += 1
            total_episodes += 1
            
            if episode_counter % 100 == 0:
                print(f"  已处理 {episode_counter} 个episode...")
    
    print(f"\n✅ 合并完成！")
    print(f"总共处理了 {total_episodes} 个episode")
    print(f"episode编号范围: 0 - {episode_counter - 1}")
    print(f"目标目录: {target_dir}")
    
    # 验证结果
    print(f"\n验证结果:")
    target_episodes = glob.glob(os.path.join(target_dir, "episode_*"))
    print(f"目标目录中的episode数量: {len(target_episodes)}")
    
    # 检查episode编号是否连续
    episode_numbers = []
    for ep_dir in target_episodes:
        ep_name = os.path.basename(ep_dir)
        ep_num = int(ep_name.split('_')[1])
        episode_numbers.append(ep_num)
    
    episode_numbers.sort()
    expected_numbers = list(range(len(episode_numbers)))
    
    if episode_numbers == expected_numbers:
        print("✅ Episode编号连续且正确")
    else:
        print("❌ Episode编号不连续")
        print(f"实际编号: {episode_numbers[:10]}...{episode_numbers[-10:]}")
        print(f"期望编号: {expected_numbers[:10]}...{expected_numbers[-10:]}")
    
    return total_episodes

if __name__ == "__main__":
    merge_processed_data()
