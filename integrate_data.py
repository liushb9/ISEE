#!/usr/bin/env python3
"""
数据整合脚本
将4个不同本体的数据整合到统一的data目录下，按任务分类
每个任务包含200条episode（每个本体50条）
"""

import os
import shutil
import glob
from pathlib import Path

def create_directory_structure():
    """创建目标目录结构"""
    base_data_dir = "/media/liushengbang/RoboTwin_baseline/data"
    
    # 6个任务目录
    tasks = [
        "blocks_ranking_rgb",
        "blocks_ranking_size", 
        "hanging_mug",
        "place_cans_plasticbox",
        "stack_blocks_three",
        "stack_bowls_three"
    ]
    
    # 创建主数据目录
    os.makedirs(base_data_dir, exist_ok=True)
    
    # 为每个任务创建目录
    for task in tasks:
        task_dir = os.path.join(base_data_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        print(f"创建任务目录: {task_dir}")
    
    return base_data_dir, tasks

def get_episode_files(source_dir, file_extensions):
    """获取指定目录下的所有episode文件"""
    episode_files = {}
    
    for ext in file_extensions:
        pattern = os.path.join(source_dir, f"episode*.{ext}")
        files = glob.glob(pattern)
        # 按episode编号排序
        files.sort(key=lambda x: int(os.path.basename(x).split('episode')[1].split('.')[0]))
        episode_files[ext] = files
    
    return episode_files

def copy_episode_data(source_task_dir, target_task_dir, episode_offset, max_episodes=50):
    """复制单个本体的episode数据到目标目录"""
    
    # 查找本体目录（包含clean的目录）
    clean_dirs = []
    for item in os.listdir(source_task_dir):
        item_path = os.path.join(source_task_dir, item)
        if os.path.isdir(item_path) and 'clean' in item:
            clean_dirs.append(item_path)
    
    if not clean_dirs:
        print(f"警告: 在 {source_task_dir} 中未找到包含'clean'的目录")
        return episode_offset
    
    # 使用第一个找到的clean目录
    source_clean_dir = clean_dirs[0]
    print(f"处理本体数据: {source_clean_dir}")
    
    # 创建目标clean目录
    target_clean_dir = os.path.join(target_task_dir, "integrated_clean")
    os.makedirs(target_clean_dir, exist_ok=True)
    
    # 复制非episode文件
    for item in os.listdir(source_clean_dir):
        item_path = os.path.join(source_clean_dir, item)
        if os.path.isfile(item_path) and not item.startswith('episode'):
            target_path = os.path.join(target_clean_dir, item)
            shutil.copy2(item_path, target_path)
            print(f"复制文件: {item}")
        elif os.path.isdir(item_path) and item != 'data' and item != 'video' and item != '_traj_data':
            # 复制其他目录（如instructions等）
            target_path = os.path.join(target_clean_dir, item)
            shutil.copytree(item_path, target_path, dirs_exist_ok=True)
            print(f"复制目录: {item}")
    
    # 创建子目录
    data_dir = os.path.join(target_clean_dir, "data")
    video_dir = os.path.join(target_clean_dir, "video")
    traj_data_dir = os.path.join(target_clean_dir, "_traj_data")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(traj_data_dir, exist_ok=True)
    
    # 获取episode文件
    source_data_dir = os.path.join(source_clean_dir, "data")
    source_video_dir = os.path.join(source_clean_dir, "video")
    source_traj_dir = os.path.join(source_clean_dir, "_traj_data")
    
    # 获取所有episode文件
    hdf5_files = get_episode_files(source_data_dir, ['hdf5'])
    mp4_files = get_episode_files(source_video_dir, ['mp4'])
    pkl_files = get_episode_files(source_traj_dir, ['pkl'])
    
    # 复制episode文件（限制数量）
    episodes_copied = 0
    for i in range(min(max_episodes, len(hdf5_files['hdf5']))):
        episode_num = episode_offset + i
        
        # 复制hdf5文件
        if i < len(hdf5_files['hdf5']):
            src_hdf5 = hdf5_files['hdf5'][i]
            dst_hdf5 = os.path.join(data_dir, f"episode{episode_num}.hdf5")
            shutil.copy2(src_hdf5, dst_hdf5)
        
        # 复制mp4文件
        if i < len(mp4_files['mp4']):
            src_mp4 = mp4_files['mp4'][i]
            dst_mp4 = os.path.join(video_dir, f"episode{episode_num}.mp4")
            shutil.copy2(src_mp4, dst_mp4)
        
        # 复制pkl文件
        if i < len(pkl_files['pkl']):
            src_pkl = pkl_files['pkl'][i]
            dst_pkl = os.path.join(traj_data_dir, f"episode{episode_num}.pkl")
            shutil.copy2(src_pkl, dst_pkl)
        
        episodes_copied += 1
    
    print(f"复制了 {episodes_copied} 个episode")
    return episode_offset + episodes_copied

def integrate_task_data(task, base_data_dir):
    """整合单个任务的所有本体数据"""
    print(f"\n开始整合任务: {task}")
    
    # 源数据目录
    source_dirs = [
        f"/media/liushengbang/RoboTwin_baseline/data_aloha-agilex/{task}",
        f"/media/liushengbang/RoboTwin_baseline/data_ARX-X5/{task}",
        f"/media/liushengbang/RoboTwin_baseline/data_franka-panda/{task}",
        f"/media/liushengbang/RoboTwin_baseline/data_ur5-wsg/{task}"
    ]
    
    # 目标目录
    target_task_dir = os.path.join(base_data_dir, task)
    
    episode_offset = 0
    
    # 处理每个本体的数据
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            print(f"处理本体目录: {source_dir}")
            episode_offset = copy_episode_data(source_dir, target_task_dir, episode_offset, max_episodes=50)
        else:
            print(f"警告: 源目录不存在: {source_dir}")
    
    print(f"任务 {task} 整合完成，总共 {episode_offset} 个episode")

def main():
    """主函数"""
    print("开始数据整合...")
    
    # 创建目录结构
    base_data_dir, tasks = create_directory_structure()
    
    # 整合每个任务的数据
    for task in tasks:
        integrate_task_data(task, base_data_dir)
    
    print("\n数据整合完成！")
    print(f"整合后的数据位于: {base_data_dir}")
    
    # 显示每个任务的episode数量
    for task in tasks:
        task_dir = os.path.join(base_data_dir, task, "integrated_clean")
        if os.path.exists(task_dir):
            data_dir = os.path.join(task_dir, "data")
            if os.path.exists(data_dir):
                hdf5_count = len(glob.glob(os.path.join(data_dir, "episode*.hdf5")))
                print(f"{task}: {hdf5_count} 个episode")

if __name__ == "__main__":
    main()
