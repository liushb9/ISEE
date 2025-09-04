#!/usr/bin/env python3
"""
测试脚本：验证所有路径是否正确
"""

import os
import yaml

def test_paths():
    """测试所有相关路径"""
    print("=== 路径测试 ===")
    
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录: {current_dir}")
    
    # 测试task_config路径
    task_config_path = os.path.join(current_dir, "../../task_config/_embodiment_config.yml")
    print(f"Task config路径: {task_config_path}")
    print(f"  存在: {os.path.exists(task_config_path)}")
    
    if os.path.exists(task_config_path):
        try:
            with open(task_config_path, "r", encoding="utf-8") as f:
                embodiment_configs = yaml.safe_load(f)
            print(f"  配置加载成功，包含本体: {list(embodiment_configs.keys())}")
            
            # 测试每个本体的路径
            for embodiment_type, config in embodiment_configs.items():
                print(f"\n--- 测试本体: {embodiment_type} ---")
                robot_file = config["file_path"]
                print(f"  配置中的路径: {robot_file}")
                
                # 转换为绝对路径
                if not os.path.isabs(robot_file):
                    robot_file = os.path.join(current_dir, "../../", robot_file)
                print(f"  绝对路径: {robot_file}")
                print(f"  存在: {os.path.exists(robot_file)}")
                
                # 测试config.yml
                robot_config_path = os.path.join(robot_file, "config.yml")
                print(f"  Config.yml路径: {robot_config_path}")
                print(f"  存在: {os.path.exists(robot_config_path)}")
                
                if os.path.exists(robot_config_path):
                    try:
                        with open(robot_config_path, "r", encoding="utf-8") as f:
                            robot_config = yaml.safe_load(f)
                        print(f"  配置加载成功")
                        if "ee_joints" in robot_config:
                            print(f"  末端执行器关节: {robot_config['ee_joints']}")
                    except Exception as e:
                        print(f"  配置加载失败: {e}")
                
                # 测试URDF文件
                if "urdf_path" in robot_config:
                    urdf_path = os.path.join(robot_file, robot_config["urdf_path"])
                    print(f"  URDF路径: {urdf_path}")
                    print(f"  存在: {os.path.exists(urdf_path)}")
                
        except Exception as e:
            print(f"  配置加载失败: {e}")
    
    # 测试数据目录
    data_dir = os.path.join(current_dir, "../../data")
    print(f"\n数据目录: {data_dir}")
    print(f"  存在: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        # 列出可用的任务
        tasks = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"  可用任务: {tasks}")
        
        # 测试stack_blocks_three
        if "stack_blocks_three" in tasks:
            task_dir = os.path.join(data_dir, "stack_blocks_three")
            configs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
            print(f"  stack_blocks_three配置: {configs}")
            
            if "demo_clean" in configs:
                demo_dir = os.path.join(task_dir, "demo_clean", "data")
                if os.path.exists(demo_dir):
                    episodes = [f for f in os.listdir(demo_dir) if f.endswith('.hdf5')]
                    print(f"  demo_clean episodes: {len(episodes)}")
                    if episodes:
                        print(f"    示例: {episodes[:5]}")

if __name__ == "__main__":
    test_paths()
