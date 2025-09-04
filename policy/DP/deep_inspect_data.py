#!/usr/bin/env python3
"""
深度检查HDF5文件，寻找本体信息的线索
"""

import os
import h5py
import numpy as np
import json

def deep_inspect_hdf5(hdf5_path):
    """深度检查HDF5文件的所有内容"""
    print(f"\n🔍 深度检查: {os.path.basename(hdf5_path)}")
    print("=" * 60)
    
    try:
        with h5py.File(hdf5_path, "r") as root:
            # 1. 检查根组属性
            print("📋 根组属性:")
            if hasattr(root, 'attrs'):
                for key, value in root.attrs.items():
                    print(f"  {key}: {value}")
            else:
                print("  无根组属性")
            
            # 2. 检查根组键
            print(f"\n📁 根组键: {list(root.keys())}")
            
            # 3. 递归检查所有组和数据集
            def inspect_group(group, level=0, path=""):
                indent = "  " * level
                current_path = f"{path}/{group.name}" if path else group.name
                
                print(f"{indent}📂 {current_path}")
                
                # 检查属性
                if hasattr(group, 'attrs'):
                    for key, value in group.attrs.items():
                        print(f"{indent}  🔖 {key}: {value}")
                
                # 检查子项
                for key in group.keys():
                    item = group[key]
                    item_path = f"{current_path}/{key}"
                    
                    if isinstance(item, h5py.Group):
                        inspect_group(item, level + 1, current_path)
                    elif isinstance(item, h5py.Dataset):
                        print(f"{indent}  📊 {key}: {item.shape} {item.dtype}")
                        
                        # 检查数据集属性
                        if hasattr(item, 'attrs'):
                            for attr_key, attr_value in item.attrs.items():
                                print(f"{indent}    🔖 {attr_key}: {attr_value}")
                        
                        # 如果是小数据集，显示一些样本数据
                        if item.size < 100 and item.dtype.kind in 'iuf':
                            try:
                                sample_data = item[:]
                                print(f"{indent}    样本数据: {sample_data}")
                            except:
                                pass
            
            # 开始递归检查
            inspect_group(root)
            
            # 4. 特别检查joint_action结构
            if "/joint_action" in root:
                print(f"\n🤖 详细检查joint_action结构:")
                joint_action = root["/joint_action"]
                
                for key in joint_action.keys():
                    item = joint_action[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {key}: {item.shape} {item.dtype}")
                        
                        # 检查属性
                        if hasattr(item, 'attrs'):
                            for attr_key, attr_value in item.attrs.items():
                                print(f"    🔖 {attr_key}: {attr_value}")
                        
                        # 显示数据样本
                        try:
                            if item.size > 0:
                                sample = item[:min(3, item.shape[0])]
                                print(f"    样本: {sample}")
                        except:
                            pass
            
            # 5. 检查图像数据
            if "/image_dict" in root:
                print(f"\n📷 检查图像数据:")
                image_dict = root["/image_dict"]
                for key in image_dict.keys():
                    item = image_dict[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {key}: {item.shape} {item.dtype}")
            
            # 6. 尝试从数据内容推断本体
            print(f"\n🧠 尝试从数据内容推断本体:")
            try:
                if "/joint_action" in root:
                    joint_action = root["/joint_action"]
                    
                    if "left_arm" in joint_action and "right_arm" in joint_action:
                        left_arm = joint_action["left_arm"]
                        right_arm = joint_action["right_arm"]
                        
                        print(f"  左臂: {left_arm.shape} {left_arm.dtype}")
                        print(f"  右臂: {right_arm.shape} {right_arm.dtype}")
                        
                        # 检查关节数量
                        left_dim = left_arm.shape[1] if len(left_arm.shape) > 1 else 1
                        right_dim = right_arm.shape[1] if len(right_arm.shape) > 1 else 1
                        print(f"  关节维度: 左{left_dim}, 右{right_dim}")
                        
                        # 检查夹爪
                        has_gripper = any("gripper" in k for k in joint_action.keys())
                        print(f"  有夹爪: {has_gripper}")
                        
                        # 分析关节数值特征
                        if left_arm.size > 0 and right_arm.size > 0:
                            try:
                                # 取前几个时间步分析
                                sample_size = min(5, left_arm.shape[0])
                                left_sample = left_arm[:sample_size]
                                right_sample = right_arm[:sample_size]
                                
                                left_mean = np.mean(left_sample, axis=0)
                                right_mean = np.mean(right_sample, axis=0)
                                left_std = np.std(left_sample, axis=0)
                                right_std = np.std(right_sample, axis=0)
                                
                                print(f"  左臂均值: {left_mean}")
                                print(f"  右臂均值: {right_mean}")
                                print(f"  左臂标准差: {left_std}")
                                print(f"  右臂标准差: {right_std}")
                                
                                # 检查关节角度范围
                                left_range = np.ptp(left_sample, axis=0)
                                right_range = np.ptp(right_sample, axis=0)
                                print(f"  左臂范围: {left_range}")
                                print(f"  右臂范围: {right_range}")
                                
                            except Exception as e:
                                print(f"    分析关节特征失败: {e}")
                
            except Exception as e:
                print(f"  检查joint_action失败: {e}")
                
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")


def main():
    """主函数"""
    print("🔍 深度检查HDF5文件，寻找本体信息")
    
    # 检查路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../../data/stack_blocks_three/demo_clean/data")
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 选择几个关键episode进行深度检查
    test_episodes = [0, 50, 100, 150, 199]
    
    for episode_num in test_episodes:
        episode_path = os.path.join(data_dir, f"episode{episode_num}.hdf5")
        
        if os.path.exists(episode_path):
            deep_inspect_hdf5(episode_path)
        else:
            print(f"\n❌ Episode {episode_num} 不存在")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
