#!/usr/bin/env python3
"""
六任务数据集合并脚本
用于将ACT项目中的六任务数据合并为多任务数据集
"""

import os
import shutil
import json
import argparse
from pathlib import Path

def merge_six_tasks(base_dir):
    """
    合并六任务数据到sim-six_tasks目录

    Args:
        base_dir: ACT项目根目录
    """
    processed_data_dir = os.path.join(base_dir, "processed_data")
    output_dir = os.path.join(processed_data_dir, "sim-six_tasks")

    # 六任务和对应的数据目录
    task_data_dirs = {
        "sim-blocks_ranking_rgb": "integrated_clean-200",
        "sim-blocks_ranking_size": "integrated_clean-200",
        "sim-hanging_mug": "integrated_clean-200",
        "sim-place_cans_plasticbox": "integrated_clean-200",
        "sim-stack_blocks_three": "integrated_clean-200",
        "sim-stack_bowls_three": "integrated_clean-200"
    }

    print("=" * 60)
    print("ACT Policy 六任务数据集合并脚本")
    print("=" * 60)

    # 检查所有任务目录是否存在
    missing_tasks = []
    available_tasks = []

    for task_name, data_dir in task_data_dirs.items():
        task_dir = os.path.join(processed_data_dir, task_name, data_dir)
        if os.path.exists(task_dir):
            available_tasks.append((task_name, task_dir))
        else:
            missing_tasks.append(f"{task_name}/{data_dir}")

    if missing_tasks:
        print(f"警告: 以下任务数据不存在: {missing_tasks}")
        print("请确保所有任务数据都已处理完成")

    if not available_tasks:
        print("错误: 没有找到任何可用的任务数据")
        return

    print(f"找到 {len(available_tasks)} 个任务:")
    for task, task_dir in available_tasks:
        episode_files = [f for f in os.listdir(task_dir) if f.startswith("episode_") and f.endswith(".hdf5")]
        print(f"  - {task}: {len(episode_files)} episodes")

    # 自动确认继续（非交互模式）
    print(f"\n将合并到: {output_dir}")
    print("自动确认继续...")

    # 创建输出目录
    if os.path.exists(output_dir):
        print(f"删除现有目录: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"\n开始合并数据集到: {output_dir}")
    os.makedirs(output_dir)

    # 合并数据
    total_episodes = 0

    for task_name, task_dir in available_tasks:
        print(f"\n处理任务: {task_name}")

        # 获取episode文件
        episode_files = [f for f in os.listdir(task_dir) if f.startswith("episode_") and f.endswith(".hdf5")]

        for i, episode_file in enumerate(episode_files):
            src_path = os.path.join(task_dir, episode_file)
            dst_path = os.path.join(output_dir, f"episode_{total_episodes}.hdf5")

            print(f"    - 复制 {episode_file} -> episode_{total_episodes}")
            shutil.copy2(src_path, dst_path)
            total_episodes += 1

    print("\n合并完成!")
    print(f"总episode数量: {total_episodes}")
    print(f"输出目录: {output_dir}")

    # 更新SIM_TASK_CONFIGS.json
    config_file = os.path.join(base_dir, "SIM_TASK_CONFIGS.json")

    try:
        with open(config_file, "r") as f:
            configs = json.load(f)
    except Exception as e:
        print(f"警告: 无法读取配置文件 {config_file}: {e}")
        configs = {}

    # 添加六任务配置
    configs["sim-six_tasks"] = {
        "dataset_dir": "./processed_data/sim-six_tasks",
        "num_episodes": total_episodes,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"]
    }

    # 保存配置文件
    with open(config_file, "w") as f:
        json.dump(configs, f, indent=4)

    print(f"\n已更新配置文件: {config_file}")
    print(f"新增配置项: sim-six_tasks (共 {total_episodes} 个episodes)")

    print("\n" + "=" * 60)
    print("数据集合并完成!")
    print("=" * 60)
    print(f"合并后的数据集包含 {total_episodes} 个episodes")
    print(f"数据集路径: {output_dir}")
    print(f"\n现在可以使用以下命令训练ACT policy:")
    print(f"./train.sh <ckpt_dir> ACT sim-six_tasks <seed> <num_epochs> <lr> <hidden_dim> <dim_feedforward> <chunk_size> <kl_weight>")

def main():
    parser = argparse.ArgumentParser(description="合并六任务数据集")
    parser.add_argument("--base_dir", default="/media/liushengbang/isee/policy/ACT",
                       help="ACT项目根目录")

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"错误: 目录不存在: {args.base_dir}")
        return

    merge_six_tasks(args.base_dir)

if __name__ == "__main__":
    main()
