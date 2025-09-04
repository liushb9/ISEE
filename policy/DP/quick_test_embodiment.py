#!/usr/bin/env python3
"""
快速测试embodiment检测功能
"""

import os
import h5py
import numpy as np

def quick_test_embodiment_detection():
    """快速测试几个episode的embodiment检测"""
    print("🔍 快速测试embodiment检测...")
    
    # 测试路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../../data/stack_blocks_three/demo_clean/data")
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 测试几个关键episode
    test_episodes = [0, 50, 100, 150, 199]  # 每个范围的边界
    
    for episode_num in test_episodes:
        episode_path = os.path.join(data_dir, f"episode{episode_num}.hdf5")
        
        if not os.path.exists(episode_path):
            print(f"Episode {episode_num}: 文件不存在")
            continue
        
        print(f"\n--- Episode {episode_num} ---")
        
        try:
            with h5py.File(episode_path, "r") as root:
                print(f"  文件大小: {os.path.getsize(episode_path)} bytes")
                
                # 检查基本结构
                print(f"  根组键: {list(root.keys())}")
                
                if "/joint_action" in root:
                    joint_action = root["/joint_action"]
                    print(f"  joint_action键: {list(joint_action.keys())}")
                    
                    if "left_arm" in joint_action and "right_arm" in joint_action:
                        left_arm = joint_action["left_arm"]
                        right_arm = joint_action["right_arm"]
                        
                        print(f"  左臂形状: {left_arm.shape}")
                        print(f"  右臂形状: {right_arm.shape}")
                        
                        # 检查夹爪
                        has_left_gripper = "left_gripper" in joint_action
                        has_right_gripper = "right_gripper" in joint_action
                        print(f"  左夹爪: {has_left_gripper}")
                        print(f"  右夹爪: {has_right_gripper}")
                        
                        # 分析关节数据特征
                        try:
                            # 取前10个时间步分析
                            sample_size = min(10, left_arm.shape[0])
                            left_sample = left_arm[:sample_size]
                            right_sample = right_arm[:sample_size]
                            
                            left_mean = np.mean(left_sample, axis=0)
                            right_mean = np.mean(right_sample, axis=0)
                            left_std = np.std(left_sample, axis=0)
                            right_std = np.std(right_sample, axis=0)
                            left_range = np.ptp(left_sample, axis=0)
                            right_range = np.ptp(right_sample, axis=0)
                            
                            print(f"  左臂均值: {left_mean}")
                            print(f"  右臂均值: {right_mean}")
                            print(f"  左臂标准差: {left_std}")
                            print(f"  右臂标准差: {right_std}")
                            print(f"  左臂范围: {left_range}")
                            print(f"  右臂范围: {right_range}")
                            
                            # 基于特征判断
                            left_avg_range = np.mean(left_range)
                            right_avg_range = np.mean(right_range)
                            
                            print(f"  左臂平均范围: {left_avg_range:.3f}")
                            print(f"  右臂平均范围: {right_avg_range:.3f}")
                            
                            # 判断embodiment
                            if left_arm.shape[1] == 7 and right_arm.shape[1] == 7:
                                detected = "franka-panda"
                            elif left_arm.shape[1] == 6 and right_arm.shape[1] == 6:
                                if has_left_gripper and has_right_gripper:
                                    if left_avg_range > 2.5 and right_avg_range > 2.5:
                                        detected = "aloha-agilex"
                                    elif left_avg_range > 1.5 and right_avg_range > 1.5:
                                        detected = "ARX-X5"
                                    else:
                                        detected = "ur5-wsg"
                                else:
                                    detected = "ARX-X5"
                            else:
                                detected = "unknown"
                            
                            print(f"  检测结果: {detected}")
                            
                        except Exception as e:
                            print(f"  分析关节特征失败: {e}")
                else:
                    print(f"  没有joint_action数据")
                
                # 检查其他属性
                if hasattr(root, 'attrs'):
                    print(f"  根组属性: {dict(root.attrs)}")
                
        except Exception as e:
            print(f"  读取文件失败: {e}")


if __name__ == "__main__":
    quick_test_embodiment_detection()
