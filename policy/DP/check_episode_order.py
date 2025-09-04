#!/usr/bin/env python3
"""
检查每个任务的数据文件夹下的episode顺序和本体对应关系
验证我们的分组策略是否正确
"""

import os
import h5py
import yaml
import argparse
from pathlib import Path
import numpy as np


def load_embodiment_config(embodiment_type):
    """加载embodiment配置"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../../task_config/_embodiment_config.yml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Embodiment config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        embodiment_configs = yaml.safe_load(f)
    
    if embodiment_type not in embodiment_configs:
        raise ValueError(f"Unknown embodiment type: {embodiment_type}")
    
    robot_file = embodiment_configs[embodiment_type]["file_path"]
    if not os.path.isabs(robot_file):
        robot_file = os.path.normpath(os.path.join(current_dir, "../../", robot_file))
    
    robot_config_path = os.path.join(robot_file, "config.yml")
    if not os.path.exists(robot_config_path):
        raise FileNotFoundError(f"Robot config file not found: {robot_config_path}")
    
    with open(robot_config_path, "r", encoding="utf-8") as f:
        robot_config = yaml.safe_load(f)
    
    return robot_config


def smart_detect_embodiment(hdf5_path):
    """基于实际数据特征的智能embodiment检测"""
    try:
        with h5py.File(hdf5_path, "r") as root:
            if "/joint_action" not in root:
                return "unknown"
            
            joint_action = root["/joint_action"]
            
            # 检查基本结构
            if "left_arm" not in joint_action or "right_arm" not in joint_action:
                return "unknown"
            
            left_arm = joint_action["left_arm"]
            right_arm = joint_action["right_arm"]
            
            # 获取数据样本进行分析
            try:
                # 取前100个时间步进行分析，避免内存问题
                sample_size = min(100, left_arm.shape[0])
                left_sample = left_arm[:sample_size]
                right_sample = right_arm[:sample_size]
                
                # 计算统计特征
                left_mean = np.mean(left_sample, axis=0)
                right_mean = np.mean(right_sample, axis=0)
                left_std = np.std(left_sample, axis=0)
                right_std = np.std(right_sample, axis=0)
                left_range = np.ptp(left_sample, axis=0)
                right_range = np.ptp(right_sample, axis=0)
                
                # 特征向量
                features = np.concatenate([
                    left_mean, right_mean,  # 均值特征
                    left_std, right_std,    # 标准差特征
                    left_range, right_range # 范围特征
                ])
                
                # 基于特征进行判断
                # 这些阈值需要根据实际数据调整
                
                # 检查是否有夹爪
                has_gripper = ("left_gripper" in joint_action and "right_gripper" in joint_action)
                
                # 基于关节数量判断
                left_dim = left_arm.shape[1] if len(left_arm.shape) > 1 else 1
                right_dim = right_arm.shape[1] if len(right_arm.shape) > 1 else 1
                
                if left_dim == 7 and right_dim == 7:
                    return "franka-panda"  # 7关节双臂
                elif left_dim == 6 and right_dim == 6:
                    if has_gripper:
                        # 有夹爪的6关节双臂，进一步区分
                        # 基于关节运动特征
                        left_avg_range = np.mean(left_range)
                        right_avg_range = np.mean(right_range)
                        
                        if left_avg_range > 2.5 and right_avg_range > 2.5:
                            return "aloha-agilex"  # 大范围运动
                        elif left_avg_range > 1.5 and right_avg_range > 1.5:
                            return "ARX-X5"  # 中等范围运动
                        else:
                            return "ur5-wsg"  # 小范围运动
                    else:
                        # 无夹爪的6关节双臂
                        return "ARX-X5"
                else:
                    return "unknown"
                    
            except Exception as e:
                print(f"        警告: 智能检测失败，使用基础检测: {e}")
                # 回退到基础检测
                return detect_embodiment_from_data(hdf5_path)
                
    except Exception as e:
        print(f"    错误: 智能检测无法读取文件: {e}")
        return "error"


def detect_embodiment_from_data(hdf5_path):
    """从HDF5数据中检测embodiment类型（改进版）"""
    try:
        with h5py.File(hdf5_path, "r") as root:
            # 检查是否有特定的标识符
            if "embodiment_type" in root.attrs:
                return root.attrs["embodiment_type"]
            
            # 检查joint_action结构
            if "/joint_action" in root:
                joint_action = root["/joint_action"]
                
                # 检查是否有特定的关节名称
                if "left_arm" in joint_action and "right_arm" in joint_action:
                    left_arm = joint_action["left_arm"]
                    right_arm = joint_action["right_arm"]
                    
                    # 根据关节数量判断
                    left_dim = left_arm.shape[1] if len(left_arm.shape) > 1 else 1
                    right_dim = right_arm.shape[1] if len(right_arm.shape) > 1 else 1
                    
                    # 检查夹爪结构
                    has_left_gripper = "left_gripper" in joint_action
                    has_right_gripper = "right_gripper" in joint_action
                    
                    # 基于关节数量和夹爪结构判断
                    if left_dim == 6 and right_dim == 6:
                        # 都是6关节，需要进一步区分
                        if has_left_gripper and has_right_gripper:
                            # 检查关节数值范围来区分
                            try:
                                left_arm_data = left_arm[:]  # 读取数据
                                right_arm_data = right_arm[:]
                                
                                # 计算关节角度范围
                                left_range = np.ptp(left_arm_data, axis=0)  # peak to peak
                                right_range = np.ptp(right_arm_data, axis=0)
                                
                                # 基于关节运动范围特征判断
                                left_avg_range = np.mean(left_range)
                                right_avg_range = np.mean(right_range)
                                
                                # 这些特征值需要根据实际数据调整
                                if left_avg_range > 3.0 and right_avg_range > 3.0:
                                    return "aloha-agilex"  # 大范围运动
                                elif left_avg_range > 2.0 and right_avg_range > 2.0:
                                    return "ARX-X5"  # 中等范围运动
                                else:
                                    return "ur5-wsg"  # 小范围运动
                                    
                            except Exception as e:
                                print(f"        警告: 无法分析关节范围，使用默认分类: {e}")
                                # 如果无法分析，使用文件名或其他方法
                                return "aloha-agilex"  # 默认分类
                        else:
                            return "aloha-agilex"  # 无夹爪的6关节双臂
                    elif left_dim == 7 and right_dim == 7:
                        return "franka-panda"  # 7关节双臂
                    elif left_dim == 6 and right_dim == 6 and not (has_left_gripper or has_right_gripper):
                        return "ARX-X5"  # 6关节双臂，无夹爪
                    elif left_dim == 6 and right_dim == 6 and (has_left_gripper or has_right_gripper):
                        return "ur5-wsg"  # 6关节双臂，有夹爪
            
            # 检查其他可能的标识符
            if "robot_type" in root.attrs:
                return root.attrs["robot_type"]
            
            if "robot_name" in root.attrs:
                return root.attrs["robot_name"]
            
            # 尝试从文件名推断
            filename = os.path.basename(hdf5_path)
            if "aloha" in filename.lower() or "agilex" in filename.lower():
                return "aloha-agilex"
            elif "franka" in filename.lower() or "panda" in filename.lower():
                return "franka-panda"
            elif "arx" in filename.lower() or "x5" in filename.lower():
                return "ARX-X5"
            elif "ur5" in filename.lower() or "wsg" in filename.lower():
                return "ur5-wsg"
            
            # 如果都找不到，返回unknown
            return "unknown"
            
    except Exception as e:
        print(f"    错误: 无法读取 {hdf5_path}: {e}")
        return "error"


def analyze_episode_data(hdf5_path):
    """分析单个episode的数据特征"""
    try:
        with h5py.File(hdf5_path, "r") as root:
            info = {}
            
            # 基本信息
            info["file_size"] = os.path.getsize(hdf5_path)
            
            # 检查joint_action结构
            if "/joint_action" in root:
                joint_action = root["/joint_action"]
                info["has_joint_action"] = True
                
                # 记录所有子组
                action_groups = {}
                for key in joint_action.keys():
                    if isinstance(joint_action[key], h5py.Dataset):
                        action_groups[key] = joint_action[key].shape
                    else:
                        action_groups[key] = "group"
                
                info["action_groups"] = action_groups
                
                # 检查是否有双臂结构
                if "left_arm" in action_groups and "right_arm" in action_groups:
                    info["arm_structure"] = "dual_arm"
                    if "left_gripper" in action_groups and "right_gripper" in action_groups:
                        info["gripper_structure"] = "dual_gripper"
                    else:
                        info["gripper_structure"] = "no_gripper"
                else:
                    info["arm_structure"] = "single_arm"
            else:
                info["has_joint_action"] = False
            
            # 检查图像数据
            if "/image_dict" in root:
                image_dict = root["/image_dict"]
                info["has_images"] = True
                info["image_cameras"] = list(image_dict.keys())
            else:
                info["has_images"] = False
            
            # 检查其他数据
            other_keys = [key for key in root.keys() if key not in ["joint_action", "image_dict"]]
            info["other_keys"] = other_keys
            
            return info
            
    except Exception as e:
        print(f"    错误: 无法分析 {hdf5_path}: {e}")
        return None


def check_task_episode_order(task_name, task_config, max_episodes=200):
    """检查特定任务的episode顺序和本体对应关系"""
    print(f"\n{'='*60}")
    print(f"检查任务: {task_name} - {task_config}")
    print(f"{'='*60}")
    
    # 构建数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../../data", task_name, task_config, "data")
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    print(f"📁 数据目录: {data_dir}")
    
    # 获取所有episode文件
    episode_files = []
    for i in range(max_episodes):
        episode_path = os.path.join(data_dir, f"episode{i}.hdf5")
        if os.path.exists(episode_path):
            episode_files.append((i, episode_path))
    
    if not episode_files:
        print(f"❌ 没有找到任何episode文件")
        return
    
    print(f"📊 找到 {len(episode_files)} 个episode文件")
    
    # 分析每个episode
    episode_analysis = []
    embodiment_counts = {}
    
    print(f"\n🔍 分析episode数据...")
    for episode_num, episode_path in episode_files:
        print(f"  Episode {episode_num:3d}: ", end="")
        
        # 使用智能检测embodiment类型
        detected_embodiment = smart_detect_embodiment(episode_path)
        
        # 分析数据特征
        data_info = analyze_episode_data(episode_path)
        
        # 记录分析结果
        episode_info = {
            "episode_num": episode_num,
            "path": episode_path,
            "detected_embodiment": detected_embodiment,
            "data_info": data_info
        }
        episode_analysis.append(episode_info)
        
        # 统计embodiment数量
        if detected_embodiment not in embodiment_counts:
            embodiment_counts[detected_embodiment] = 0
        embodiment_counts[detected_embodiment] += 1
        
        print(f"{detected_embodiment:15s} | ", end="")
        
        if data_info and data_info.get("has_joint_action"):
            if "action_groups" in data_info:
                action_info = data_info["action_groups"]
                if "left_arm" in action_info and "right_arm" in action_info:
                    left_shape = action_info["left_arm"]
                    right_shape = action_info["right_arm"]
                    print(f"左臂{left_shape} 右臂{right_shape}")
                else:
                    print(f"单臂结构")
            else:
                print(f"有joint_action但结构未知")
        else:
            print(f"无joint_action数据")
    
    # 分析结果
    print(f"\n📊 分析结果:")
    print(f"{'='*60}")
    
    # 统计embodiment分布
    print(f"Embodiment分布:")
    for emb_type, count in sorted(embodiment_counts.items()):
        print(f"  {emb_type:15s}: {count:3d} episodes")
    
    # 检查我们的假设是否正确
    print(f"\n🔍 验证我们的分组假设:")
    
    # 我们的假设分组（基于merge_data.sh中的实际顺序）
    assumed_groups = {
        "ur5-wsg": {"start": 0, "end": 50, "episodes": []},
        "franka-panda": {"start": 50, "end": 100, "episodes": []},
        "ARX-X5": {"start": 100, "end": 150, "episodes": []},
        "aloha-agilex": {"start": 150, "end": 200, "episodes": []}
    }
    
    # 根据检测结果重新分组
    actual_groups = {}
    for episode_info in episode_analysis:
        detected = episode_info["detected_embodiment"]
        episode_num = episode_info["episode_num"]
        
        if detected not in actual_groups:
            actual_groups[detected] = []
        actual_groups[detected].append(episode_num)
    
    # 比较假设和实际
    print(f"\n假设的分组 vs 实际检测:")
    for assumed_emb, assumed_range in assumed_groups.items():
        print(f"\n  {assumed_emb}:")
        print(f"    假设范围: {assumed_range['start']:3d} - {assumed_range['end']:3d}")
        
        if assumed_emb in actual_groups:
            actual_episodes = sorted(actual_groups[assumed_emb])
            print(f"    实际检测: {len(actual_episodes)} episodes")
            print(f"    实际范围: {min(actual_episodes):3d} - {max(actual_episodes):3d}")
            
            # 检查是否在假设范围内
            in_range = [ep for ep in actual_episodes if assumed_range['start'] <= ep < assumed_range['end']]
            out_of_range = [ep for ep in actual_episodes if ep < assumed_range['start'] or ep >= assumed_range['end']]
            
            if in_range:
                print(f"    ✅ 在假设范围内: {len(in_range)} episodes")
            if out_of_range:
                print(f"    ⚠️  超出假设范围: {len(out_of_range)} episodes: {out_of_range}")
        else:
            print(f"    ❌ 未检测到任何episode")
    
    # 检查是否有未预期的embodiment
    unexpected = [emb for emb in actual_groups.keys() if emb not in assumed_groups]
    if unexpected:
        print(f"\n⚠️  检测到未预期的embodiment类型:")
        for emb in unexpected:
            episodes = sorted(actual_groups[emb])
            print(f"  {emb}: {len(episodes)} episodes, 范围: {min(episodes)} - {max(episodes)}")
    
    # 提供建议
    print(f"\n💡 建议:")
    if len(actual_groups) == 4 and all(emb in actual_groups for emb in assumed_groups.keys()):
        print(f"  ✅ 检测结果与假设基本一致，可以继续使用当前的分组策略")
    else:
        print(f"  ⚠️  检测结果与假设不一致，建议:")
        print(f"    1. 检查数据收集时的本体分配逻辑")
        print(f"    2. 修改分组策略以匹配实际数据")
        print(f"    3. 或者重新组织数据文件")
    
    return episode_analysis, actual_groups


def main():
    parser = argparse.ArgumentParser(description="检查任务episode顺序和本体对应关系")
    parser.add_argument("--task_name", type=str, default="stack_blocks_three", 
                       help="任务名称")
    parser.add_argument("--task_config", type=str, default="demo_clean",
                       help="任务配置")
    parser.add_argument("--max_episodes", type=int, default=200,
                       help="最大episode数量")
    parser.add_argument("--check_all_tasks", action="store_true",
                       help="检查所有可用任务")
    
    args = parser.parse_args()
    
    if args.check_all_tasks:
        # 检查所有可用任务
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(current_dir, "../../data")
        
        if not os.path.exists(data_root):
            print(f"❌ 数据根目录不存在: {data_root}")
            return
        
        # 获取所有任务
        tasks = []
        for task_dir in os.listdir(data_root):
            task_path = os.path.join(data_root, task_dir)
            if os.path.isdir(task_path):
                # 获取任务配置
                configs = []
                for config_dir in os.listdir(task_path):
                    config_path = os.path.join(task_path, config_dir)
                    if os.path.isdir(config_path):
                        configs.append(config_dir)
                
                if configs:
                    tasks.append((task_dir, configs))
        
        print(f"🔍 发现 {len(tasks)} 个任务:")
        for task_name, configs in tasks:
            print(f"  {task_name}: {configs}")
        
        # 检查每个任务
        for task_name, configs in tasks:
            for task_config in configs:
                try:
                    check_task_episode_order(task_name, task_config, args.max_episodes)
                except Exception as e:
                    print(f"❌ 检查任务 {task_name}-{task_config} 时出错: {e}")
                    continue
    else:
        # 检查指定任务
        check_task_episode_order(args.task_name, args.task_config, args.max_episodes)


if __name__ == "__main__":
    main()
