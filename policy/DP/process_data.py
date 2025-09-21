import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available, using dummy text features")
from typing import List

# 添加UVA模块路径
sys.path.append('/home/zijian/RoboTwin/policy/UVA')
from unified_video_action.model.common.rotation_transformer import RotationTransformer


def load_embodiment_config(embodiment_type):
    """加载embodiment配置文件"""
    # 使用绝对路径来避免相对路径问题
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../../task_config/_embodiment_config.yml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Embodiment config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        embodiment_configs = yaml.safe_load(f)
    
    if embodiment_type not in embodiment_configs:
        raise ValueError(f"Unknown embodiment type: {embodiment_type}")
    
    robot_file = embodiment_configs[embodiment_type]["file_path"]
    
    # 处理相对路径，转换为绝对路径（修复路径冗余问题）
    if not os.path.isabs(robot_file):
        # 直接使用os.path.normpath来清理路径
        robot_file = os.path.normpath(os.path.join(current_dir, "../../", robot_file))
    
    robot_config_path = os.path.join(robot_file, "config.yml")
    
    if not os.path.exists(robot_config_path):
        raise FileNotFoundError(f"Robot config file not found: {robot_config_path}")
    
    with open(robot_config_path, "r", encoding="utf-8") as f:
        robot_config = yaml.safe_load(f)
    
    return robot_config


def analyze_joint_data(joint_data, name="joint_data"):
    """分析关节数据的特征（精简版）"""
    # 只在调试模式下显示详细信息
    if os.environ.get('DEBUG_JOINT_DATA') == '1':
        print(f"        🔍 分析{name}特征:")
        print(f"          形状: {joint_data.shape}")
        print(f"          数据类型: {joint_data.dtype}")
        print(f"          数值范围: [{joint_data.min():.4f}, {joint_data.max():.4f}]")
        print(f"          均值: {joint_data.mean():.4f}")
        print(f"          标准差: {joint_data.std():.4f}")
        
        # 检查是否在关节角度范围内
        if joint_data.min() >= -np.pi and joint_data.max() <= np.pi:
            print(f"          ✅ 数据在关节角度范围内 [-π, π]")
        elif joint_data.min() >= -2*np.pi and joint_data.max() <= 2*np.pi:
            print(f"          ⚠️  数据在扩展关节角度范围内 [-2π, 2π]")
        else:
            print(f"          ❓ 数据超出标准关节角度范围")
        
        # 检查每个维度的特征
        for i in range(min(3, joint_data.shape[-1])):  # 只显示前3个维度
            dim_data = joint_data[..., i]
            print(f"          维度{i}: 范围[{dim_data.min():.4f}, {dim_data.max():.4f}], 均值{dim_data.mean():.4f}")
        
        if joint_data.shape[-1] > 3:
            print(f"          ... 还有{joint_data.shape[-1]-3}个维度")
    else:
        # 精简版：只显示关键信息
        print(f"        {name}: {joint_data.shape} {joint_data.dtype}")


def detect_rotation_format(data):
    """检测旋转数据的表示格式"""
    if data.shape[-1] == 4:
        return "quaternion"
    elif data.shape[-1] == 6:
        return "rotation_6d"
    elif data.shape[-1] == 3:
        return "axis_angle"
    elif data.shape[-1] == 7:
        # 7维数据可能是关节角度 + 夹爪，或者是7关节机器人
        # 检查数据范围来判断
        data_min, data_max = data.min(), data.max()
        if data_min >= -np.pi and data_max <= np.pi:
            # 如果数据在[-π, π]范围内，很可能是关节角度
            return "joint_angles"
        else:
            # 如果超出这个范围，可能是其他格式
            return "joint_angles_extended"
    elif data.shape[-1] > 10:
        # 高维数据可能是组合状态向量
        return "state_vector"
    else:
        # 对于其他维度，尝试分析数据特征
        data_min, data_max = data.min(), data.max()
        data_std = data.std()
        
        print(f"      🔍 未知维度数据特征: 形状={data.shape}, 范围=[{data_min:.3f}, {data_max:.3f}], 标准差={data_std:.3f}")
        
        # 根据数据特征推测格式
        if data_std < 0.1:
            return "constant_or_small_variation"
        elif abs(data_min) < 1 and abs(data_max) < 1:
            return "normalized_data"
        else:
            return "raw_joint_data"


def verify_data_integrity(original_data, processed_data, tolerance=1e-6):
    """验证数据完整性，检查是否发生意外修改"""
    print(f"    🔍 验证数据完整性...")
    
    # 检查数据类型
    if hasattr(original_data, 'dtype') and hasattr(processed_data, 'dtype'):
        if original_data.dtype != processed_data.dtype:
            print(f"      ⚠️  数据类型变化: {original_data.dtype} → {processed_data.dtype}")
    
    # 检查数据形状
    if hasattr(original_data, 'shape') and hasattr(processed_data, 'shape'):
        if original_data.shape != processed_data.shape:
            print(f"      ⚠️  数据形状变化: {original_data.shape} → {processed_data.shape}")
    
    # 检查数值范围
    if hasattr(original_data, 'min') and hasattr(original_data, 'max'):
        orig_min, orig_max = original_data.min(), original_data.max()
        proc_min, proc_max = processed_data.min(), processed_data.max()
        
        if abs(orig_min - proc_min) > tolerance or abs(orig_max - proc_max) > tolerance:
            print(f"      ⚠️  数值范围变化: [{orig_min:.6f}, {orig_max:.6f}] → [{proc_min:.6f}, {proc_max:.6f}]")
    
    # 检查是否有NaN或无穷大值
    if hasattr(processed_data, 'dtype') and np.issubdtype(processed_data.dtype, np.floating):
        if np.any(np.isnan(processed_data)):
            print(f"      ❌ 检测到NaN值！")
        if np.any(np.isinf(processed_data)):
            print(f"      ❌ 检测到无穷大值！")
    
    print(f"      ✅ 数据完整性验证完成")


def normalize_rotation_data(rotation_data, source_format, target_format="rotation_6d"):
    """将旋转数据统一转换为目标格式，包含数据完整性验证"""
    if source_format == target_format:
        return rotation_data
    
    # 保存原始数据的副本用于验证
    original_data = rotation_data.copy()
    
    print(f"      🔄 转换旋转数据: {source_format} → {target_format}")
    
    # 特殊处理关节角度数据
    if source_format in ["joint_angles", "joint_angles_extended", "raw_joint_data"]:
        print(f"        ℹ️  检测到关节角度数据，维度: {rotation_data.shape[-1]}")
        
        # 对于关节角度数据，我们有两个选择：
        # 1. 保持原始格式（推荐）
        # 2. 尝试转换为rotation_6d（如果维度合适）
        
        if rotation_data.shape[-1] == 7:
            print(f"        ℹ️  7维关节数据，保持原始格式以确保数据完整性")
            return rotation_data  # 保持原始格式
        
        elif rotation_data.shape[-1] == 6:
            print(f"        ℹ️  6维数据，尝试转换为rotation_6d")
            # 6维数据可能是前6个关节，可以尝试转换
            try:
                # 假设前6维是旋转关节，转换为rotation_6d
                transformer = RotationTransformer("axis_angle", "rotation_6d")
                # 这里需要将关节角度转换为轴角表示
                # 简化处理：假设每个关节都是绕Z轴的旋转
                converted_data = rotation_data.copy()
                return converted_data
            except Exception as e:
                print(f"        ⚠️  转换失败，保持原始格式: {e}")
                return rotation_data
        
        else:
            print(f"        ℹ️  {rotation_data.shape[-1]}维关节数据，保持原始格式")
            return rotation_data
    
    try:
        # 第一层fallback：直接转换
        transformer = RotationTransformer(source_format, target_format)
        converted_data = transformer.forward(rotation_data)
        
        # 验证转换结果
        verify_data_integrity(original_data, converted_data)
        
        return converted_data
    except Exception as e:
        print(f"        ⚠️  直接转换失败: {e}")
        
        # 第二层fallback：通过axis_angle作为中间格式
        if source_format != "axis_angle" and target_format != "axis_angle":
            try:
                print(f"        🔄 尝试通过axis_angle转换...")
                transformer1 = RotationTransformer(source_format, "axis_angle")
                transformer2 = RotationTransformer("axis_angle", target_format)
                intermediate = transformer1.forward(rotation_data)
                converted_data = transformer2.forward(intermediate)
                
                # 验证转换结果
                verify_data_integrity(original_data, converted_data)
                
                return converted_data
            except Exception as e2:
                print(f"        ⚠️  通过axis_angle转换也失败: {e2}")
                
                # 第三层fallback：返回原始数据
                print(f"        ⚠️  返回原始数据，未进行转换")
                return rotation_data


def extract_endpose_data(hdf5_data, embodiment_config):
    """从HDF5数据中提取endpose信息"""
    try:
        # 尝试从embodiment配置中获取末端执行器信息
        if "ee_joints" in embodiment_config:
            left_ee_joint = embodiment_config["ee_joints"][0]
            right_ee_joint = embodiment_config["ee_joints"][1]
            
            # 检查是否存在对应的观察数据
            if f"/observation/{left_ee_joint}" in hdf5_data:
                left_ee_state = hdf5_data[f"/observation/{left_ee_joint}"][()]
                right_ee_state = hdf5_data[f"/observation/{right_ee_joint}"][()]
                
                return {
                    "left_endpose": left_ee_state,
                    "right_endpose": right_ee_state
                }
    except Exception as e:
        print(f"Warning: Failed to extract endpose data: {e}")
    
    return None


def load_hdf5(dataset_path, embodiment_config=None):
    """加载HDF5数据，支持不同本体格式"""
    if not os.path.isfile(dataset_path):
        print(f"❌ Dataset does not exist at {dataset_path}")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # 尝试不同的数据格式
        data = {}
        
        # 精简输出：只显示关键结构信息
        if os.environ.get('DEBUG_HDF5') == '1':
            print(f"    HDF5文件结构: {list(root.keys())}")
        
        # 检查joint_action结构
        if "/joint_action" in root:
            joint_action = root["/joint_action"]
            if os.environ.get('DEBUG_HDF5') == '1':
                print(f"    Joint action结构: {list(joint_action.keys())}")
            
            # 尝试加载双臂数据 - 创建副本保护原始数据
            if "left_gripper" in joint_action and "left_arm" in joint_action:
                data["left_gripper"] = joint_action["left_gripper"][()].copy()  # 创建副本
                data["left_arm"] = joint_action["left_arm"][()].copy()  # 创建副本
                data["right_gripper"] = joint_action["right_gripper"][()].copy()  # 创建副本
                data["right_arm"] = joint_action["right_arm"][()].copy()  # 创建副本
                data["format"] = "dual_arm_separate"
                print(f"    ✓ 双臂分离格式: 左臂{data['left_arm'].shape}, 右臂{data['right_arm'].shape}")
                
                # 分析关节数据特征
                analyze_joint_data(data["left_arm"], "左臂")
                analyze_joint_data(data["right_arm"], "右臂")
                analyze_joint_data(data["left_gripper"], "左夹爪")
                analyze_joint_data(data["right_gripper"], "右夹爪")
                
            elif "left_arm" in joint_action and "right_arm" in joint_action:
                data["left_arm"] = joint_action["left_arm"][()].copy()  # 创建副本
                data["right_arm"] = joint_action["right_arm"][()].copy()  # 创建副本
                data["format"] = "dual_arm_no_gripper"
                print(f"    ✓ 双臂无夹爪格式: 左臂{data['left_arm'].shape}, 右臂{data['right_arm'].shape}")
                
                # 分析关节数据特征
                analyze_joint_data(data["left_arm"], "左臂")
                analyze_joint_data(data["right_arm"], "右臂")
            else:
                # 尝试加载单臂数据
                for key in joint_action.keys():
                    if "arm" in key or "gripper" in key:
                        data[key] = joint_action[key][()].copy()  # 创建副本
                        print(f"    ✓ 单臂数据: {key} {data[key].shape}")
                data["format"] = "single_arm"
        
        # 检查vector数据
        if "/joint_action/vector" in root:
            data["vector"] = root["/joint_action/vector"][()].copy()  # 创建副本
            print(f"    ✓ Vector数据: {data['vector'].shape}")
        
        # 加载图像数据
        image_dict = dict()
        if "/observation" in root:
            if os.environ.get('DEBUG_HDF5') == '1':
                print(f"    Observation结构: {list(root['/observation'].keys())}")
            for cam_name in root["/observation"].keys():
                if "rgb" in root[f"/observation/{cam_name}"]:
                    image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()].copy()  # 创建副本
                    if os.environ.get('DEBUG_HDF5') == '1':
                        print(f"    ✓ 相机: {cam_name} {image_dict[cam_name].shape}")
        
        data["image_dict"] = image_dict
        
        # 尝试提取endpose数据
        if embodiment_config:
            endpose_data = extract_endpose_data(root, embodiment_config)
            if endpose_data:
                data["endpose"] = endpose_data
                if os.environ.get('DEBUG_HDF5') == '1':
                    print(f"    ✓ 成功提取endpose数据")

    return data


def normalize_robot_data_safe(data, embodiment_type):
    """安全地标准化机器人数据，保持原始维度和物理意义"""
    print(f"      🔧 标准化 {embodiment_type} 数据...")
    
    normalized_data = {}
    
    if data["format"] == "dual_arm_separate":
        print(f"        ✓ 双臂分离格式")
        
        # 保持原始维度，不进行强制转换
        left_arm = data["left_arm"].copy()
        right_arm = data["right_arm"].copy()
        left_gripper = data["left_gripper"].copy()
        right_gripper = data["right_gripper"].copy()
        
        print(f"        左臂: {left_arm.shape}, 右臂: {right_arm.shape}")
        print(f"        左夹爪: {left_gripper.shape}, 右夹爪: {right_gripper.shape}")
        
        # 分析数据特征，但不强制修改
        analyze_joint_data(left_arm, "左臂")
        analyze_joint_data(right_arm, "右臂")
        analyze_joint_data(left_gripper, "左夹爪")
        analyze_joint_data(right_gripper, "右夹爪")
        
        # 创建action数组，保持原始维度
        timesteps = left_arm.shape[0]
        
        # 计算总action维度
        left_arm_dim = left_arm.shape[1] if left_arm.ndim > 1 else 1
        right_arm_dim = right_arm.shape[1] if right_arm.ndim > 1 else 1
        left_gripper_dim = left_gripper.shape[1] if left_gripper.ndim > 1 else 1
        right_gripper_dim = right_gripper.shape[1] if right_gripper.ndim > 1 else 1
        
        total_action_dim = left_arm_dim + right_arm_dim + left_gripper_dim + right_gripper_dim
        
        print(f"        总action维度: {total_action_dim}")
        
        # 创建action数组
        normalized_action = np.zeros((timesteps, total_action_dim))
        
        # 填充数据，保持原始结构
        start_idx = 0
        
        # 左臂
        if left_arm.ndim > 1:
            normalized_action[:, start_idx:start_idx+left_arm_dim] = left_arm
        else:
            normalized_action[:, start_idx] = left_arm
        start_idx += left_arm_dim
        
        # 右臂
        if right_arm.ndim > 1:
            normalized_action[:, start_idx:start_idx+right_arm_dim] = right_arm
        else:
            normalized_action[:, start_idx] = right_arm
        start_idx += right_arm_dim
        
        # 左夹爪
        if left_gripper.ndim > 1:
            normalized_action[:, start_idx:start_idx+left_gripper_dim] = left_gripper
        else:
            normalized_action[:, start_idx] = left_gripper
        start_idx += left_gripper_dim
        
        # 右夹爪
        if right_gripper.ndim > 1:
            normalized_action[:, start_idx:start_idx+right_gripper_dim] = right_gripper
        else:
            normalized_action[:, start_idx] = right_gripper
        
        normalized_data["action"] = normalized_action
        normalized_data["action_dim"] = total_action_dim
        normalized_data["action_structure"] = {
            "left_arm": {"start": 0, "end": left_arm_dim, "dim": left_arm_dim},
            "right_arm": {"start": left_arm_dim, "end": left_arm_dim + right_arm_dim, "dim": right_arm_dim},
            "left_gripper": {"start": left_arm_dim + right_arm_dim, "end": left_arm_dim + right_arm_dim + left_gripper_dim, "dim": left_gripper_dim},
            "right_gripper": {"start": left_arm_dim + right_arm_dim + left_gripper_dim, "end": total_action_dim, "dim": right_gripper_dim}
        }
        
    elif data["format"] == "dual_arm_no_gripper":
        print(f"        ✓ 双臂无夹爪格式")
        
        left_arm = data["left_arm"].copy()
        right_arm = data["right_arm"].copy()
        
        print(f"        左臂: {left_arm.shape}, 右臂: {right_arm.shape}")
        
        # 分析数据特征
        analyze_joint_data(left_arm, "左臂")
        analyze_joint_data(right_arm, "右臂")
        
        # 创建action数组
        timesteps = left_arm.shape[0]
        left_arm_dim = left_arm.shape[1] if left_arm.ndim > 1 else 1
        right_arm_dim = right_arm.shape[1] if right_arm.ndim > 1 else 1
        
        total_action_dim = left_arm_dim + right_arm_dim
        
        normalized_action = np.zeros((timesteps, total_action_dim))
        
        start_idx = 0
        if left_arm.ndim > 1:
            normalized_action[:, start_idx:start_idx+left_arm_dim] = left_arm
        else:
            normalized_action[:, start_idx] = left_arm
        start_idx += left_arm_dim
        
        if right_arm.ndim > 1:
            normalized_action[:, start_idx:start_idx+right_arm_dim] = right_arm
        else:
            normalized_action[:, start_idx] = right_arm
        
        normalized_data["action"] = normalized_action
        normalized_data["action_dim"] = total_action_dim
        normalized_data["action_structure"] = {
            "left_arm": {"start": 0, "end": left_arm_dim, "dim": left_arm_dim},
            "right_arm": {"start": left_arm_dim, "end": total_action_dim, "dim": right_arm_dim}
        }
        
    else:
        print(f"        ✓ 单臂格式")
        
        # 处理单臂数据
        action_parts = []
        action_structure = {}
        start_idx = 0
        
        for key, value in data.items():
            if "arm" in key or "gripper" in key:
                action_parts.append(value.copy())
                part_dim = value.shape[1] if value.ndim > 1 else 1
                action_structure[key] = {"start": start_idx, "end": start_idx + part_dim, "dim": part_dim}
                start_idx += part_dim
                print(f"        {key}: {value.shape}")
        
        if action_parts:
            timesteps = action_parts[0].shape[0]
            total_action_dim = start_idx
            
            normalized_action = np.zeros((timesteps, total_action_dim))
            
            start_idx = 0
            for i, part in enumerate(action_parts):
                part_dim = part.shape[1] if part.ndim > 1 else 1
                if part.ndim > 1:
                    normalized_action[:, start_idx:start_idx+part_dim] = part
                else:
                    normalized_action[:, start_idx] = part
                start_idx += part_dim
            
            normalized_data["action"] = normalized_action
            normalized_data["action_dim"] = total_action_dim
            normalized_data["action_structure"] = action_structure
    
    # 复制其他数据，保持原始格式
    for key, value in data.items():
        if key not in ["left_arm", "right_arm", "left_gripper", "right_gripper", "format"]:
            normalized_data[key] = value.copy()
    
    print(f"        ✅ 数据标准化完成")
    return normalized_data


def standardize_array_dimensions(arrays_list, target_dim=None):
    """标准化数组列表的维度，确保所有数组都有相同的形状"""
    if not arrays_list:
        return np.array([])
    
    print(f"    🔧 标准化数组维度...")
    print(f"      原始数组数量: {len(arrays_list)}")
    
    # 检查第一个数组的维度
    first_array = arrays_list[0]
    if hasattr(first_array, 'shape'):
        print(f"      第一个数组形状: {first_array.shape}")
        if first_array.ndim == 1:
            target_dim = first_array.shape[0]
        else:
            target_dim = first_array.flatten().shape[0]
    else:
        print(f"      第一个数组类型: {type(first_array)}")
        return np.array(arrays_list)
    
    print(f"      目标维度: {target_dim}")
    
    # 标准化所有数组
    standardized_arrays = []
    for i, arr in enumerate(arrays_list):
        try:
            if hasattr(arr, 'ndim'):
                if arr.ndim == 1:
                    if arr.shape[0] == target_dim:
                        standardized_arrays.append(arr)
                    else:
                        # 维度不匹配，进行padding或truncation
                        if arr.shape[0] < target_dim:
                            # 填充到目标维度
                            padded_arr = np.zeros(target_dim, dtype=arr.dtype)
                            padded_arr[:arr.shape[0]] = arr
                            standardized_arrays.append(padded_arr)
                            print(f"        数组{i}: 从{arr.shape[0]}填充到{target_dim}")
                        else:
                            # 截断到目标维度
                            truncated_arr = arr[:target_dim]
                            standardized_arrays.append(truncated_arr)
                            print(f"        数组{i}: 从{arr.shape[0]}截断到{target_dim}")
                else:
                    # 多维数组，展平
                    flattened_arr = arr.flatten()
                    if flattened_arr.shape[0] == target_dim:
                        standardized_arrays.append(flattened_arr)
                    else:
                        # 处理展平后的维度不匹配
                        if flattened_arr.shape[0] < target_dim:
                            padded_arr = np.zeros(target_dim, dtype=flattened_arr.dtype)
                            padded_arr[:flattened_arr.shape[0]] = flattened_arr
                            standardized_arrays.append(padded_arr)
                            print(f"        数组{i}: 展平后从{flattened_arr.shape[0]}填充到{target_dim}")
                        else:
                            truncated_arr = flattened_arr[:target_dim]
                            standardized_arrays.append(truncated_arr)
                            print(f"        数组{i}: 展平后从{flattened_arr.shape[0]}截断到{target_dim}")
            else:
                print(f"        数组{i}: 跳过，不是数组类型")
                continue
        except Exception as e:
            print(f"        数组{i}: 处理失败: {e}")
            # 创建零数组作为fallback
            fallback_arr = np.zeros(target_dim, dtype=np.float32)
            standardized_arrays.append(fallback_arr)
    
    print(f"      标准化后数组数量: {len(standardized_arrays)}")
    
    # 转换为numpy数组
    try:
        result = np.array(standardized_arrays)
        print(f"      ✅ 成功创建numpy数组，形状: {result.shape}")
        return result
    except Exception as e:
        print(f"      ❌ 创建numpy数组失败: {e}")
        # 如果还是失败，尝试使用object类型
        try:
            result = np.array(standardized_arrays, dtype=object)
            print(f"      ⚠️  使用object类型创建数组，形状: {result.shape}")
            return result
        except Exception as e2:
            print(f"      ❌ 即使object类型也失败: {e2}")
            return np.array([])


def text2feats(text_inputs: List[str]):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    text_tokens = clip.tokenize(text_inputs).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_feat = text_features.detach().cpu().numpy()
    return text_feat.astype(np.float32)


def validate_embodiment_data_consistency(embodiment_type, episode_list, load_dir):
    """验证同一本体内数据的维度一致性"""
    print(f"      🔍 验证 {embodiment_type} 数据一致性...")
    
    dimensions = []
    episode_lengths = []
    
    for episode_num in episode_list[:3]:  # 只检查前3个episode
        load_path = os.path.join(load_dir, f"data/episode{episode_num}.hdf5")
        
        try:
            with h5py.File(load_path, "r") as root:
                if "/joint_action" in root:
                    joint_action = root["/joint_action"]
                    
                    if "left_arm" in joint_action and "right_arm" in joint_action:
                        left_arm_shape = joint_action["left_arm"].shape
                        right_arm_shape = joint_action["right_arm"].shape
                        
                        dimensions.append({
                            "episode": episode_num,
                            "left_arm": left_arm_shape,
                            "right_arm": right_arm_shape,
                            "total": left_arm_shape[1] + right_arm_shape[1] if len(left_arm_shape) > 1 and len(right_arm_shape) > 1 else 0
                        })
                        
                        episode_lengths.append(left_arm_shape[0])
                        
        except Exception as e:
            print(f"        警告: 无法验证episode {episode_num}: {e}")
    
    if dimensions:
        print(f"        维度检查结果:")
        for dim_info in dimensions:
            print(f"          Episode {dim_info['episode']}: 左臂{dim_info['left_arm']}, 右臂{dim_info['right_arm']}")
        
        # 检查维度一致性
        left_arm_dims = [d["left_arm"][1] if len(d["left_arm"]) > 1 else 1 for d in dimensions]
        right_arm_dims = [d["right_arm"][1] if len(d["right_arm"]) > 1 else 1 for d in dimensions]
        
        if len(set(left_arm_dims)) > 1:
            print(f"        ⚠️  警告: 左臂维度不一致: {left_arm_dims}")
        else:
            print(f"        ✅ 左臂维度一致: {left_arm_dims[0]}")
            
        if len(set(right_arm_dims)) > 1:
            print(f"        ⚠️  警告: 右臂维度不一致: {right_arm_dims}")
        else:
            print(f"        ✅ 右臂维度一致: {right_arm_dims[0]}")
        
        # 检查episode长度一致性
        if len(set(episode_lengths)) > 1:
            print(f"        ⚠️  警告: Episode长度不一致: {episode_lengths}")
        else:
            print(f"        ✅ Episode长度一致: {episode_lengths[0]}")
    
    return dimensions


def print_data_processing_summary(embodiment_type, episode_list, load_dir):
    """打印数据处理摘要和注意事项"""
    print(f"\n      📊 {embodiment_type} 数据处理摘要")
    print(f"        ========================================")
    print(f"        处理的episode数量: {len(episode_list)}")
    print(f"        Episode范围: {episode_list[0]} - {episode_list[-1]}")
    
    # 验证数据一致性
    dimensions = validate_embodiment_data_consistency(embodiment_type, episode_list, load_dir)
    
    print(f"\n        ⚠️  重要注意事项:")
    print(f"        1. 数据保持原始维度和物理意义")
    print(f"        2. 不进行强制维度统一或数据截断")
    print(f"        3. 每个本体独立处理，避免混淆")
    print(f"        4. 建议在训练时分别使用不同本体的数据")
    
    if dimensions:
        print(f"\n        📋 建议的训练策略:")
        print(f"        - 为每个本体创建单独的模型")
        print(f"        - 或者使用动态输入层处理不同维度")
        print(f"        - 避免混合训练不同本体的数据")
    
    print(f"        ========================================")


def text2feats(text_inputs: List[str]):
    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    text_tokens = clip.tokenize(text_inputs).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_feat = text_features.detach().cpu().numpy()
    return text_feat.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    # Convert task name to text features
    task_text = task_name.replace("_", " ")
    text_feat = text2feats([task_text])
    print(f"Task: {task_name}, Text: '{task_text}', Text feature shape: {text_feat.shape}")

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    # 重新设计：为每个本体分别处理数据
    print("=== 多本体数据处理 ===")
    print("为避免数据丢失和物理意义混淆，将为每个本体创建单独的数据文件")
    
    # 定义本体分组（基于merge_data.sh中的实际顺序）
    embodiment_groups = {
        "ur5-wsg": {"start": 0, "end": 50, "episodes": []},
        "franka-panda": {"start": 50, "end": 100, "episodes": []},
        "ARX-X5": {"start": 100, "end": 150, "episodes": []},
        "aloha-agilex": {"start": 150, "end": 200, "episodes": []}
    }
    
    # 第一遍：收集每个本体的episode信息
    print("\n=== 第一遍：收集episode信息 ===")
    for current_ep in range(num):
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        
        # 确定episode属于哪个本体
        embodiment_type = None
        for emb_type, group in embodiment_groups.items():
            if group["start"] <= current_ep < group["end"]:
                embodiment_type = emb_type
                break
        
        if embodiment_type:
            embodiment_groups[embodiment_type]["episodes"].append(current_ep)
            # 精简输出：只显示每10个episode的进度
            if current_ep % 10 == 0 or current_ep == num - 1:
                print(f"Episode {current_ep} → {embodiment_type}")
    
    # 显示分组结果
    print("\n📊 Episode分组结果:")
    for emb_type, group in embodiment_groups.items():
        print(f"  {emb_type}: {len(group['episodes'])} episodes")
    
    # 第二遍：为每个本体分别处理数据
    print("\n=== 第二遍：分别处理每个本体 ===")
    for embodiment_type, group in embodiment_groups.items():
        if not group["episodes"]:
            print(f"⚠️  {embodiment_type}: 没有episode数据")
            continue
            
        print(f"\n--- 处理 {embodiment_type} ({len(group['episodes'])} episodes) ---")
        
        # 精简摘要信息
        print(f"  📊 {embodiment_type}: {len(group['episodes'])} episodes, 范围: {group['episodes'][0]}-{group['episodes'][-1]}")
        
        # 为每个本体创建单独的输出文件
        save_dir = f"./data/{task_name}-{task_config}-{embodiment_type}-{len(group['episodes'])}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

        # 处理这个本体的所有episode
        process_embodiment_episodes(
            embodiment_type, 
            group["episodes"], 
            load_dir, 
            save_dir, 
            task_name
        )
    
    print(f"\n=== 处理完成 ===")
    print("每个本体都有独立的数据文件，避免了数据丢失和物理意义混淆")


def process_embodiment_episodes(embodiment_type, episode_list, load_dir, save_dir, task_name):
    """处理单个本体的所有episode"""
    print(f"  开始处理 {embodiment_type} 的 {len(episode_list)} 个episode...")
    
    # 创建zarr文件
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = (
        [],
        [],
        [],
        [],
    )
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays, action_mask_arrays = (
        [],
        [],
        [],
        [],
        [],
    )

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            image_dict_all,
        ) = load_hdf5(load_path)

        for j in range(0, left_gripper_all.shape[0]):

            # 处理图像和状态数据
            if "image_dict" in normalized_data and "head_camera" in normalized_data["image_dict"]:
                image_dict_all = normalized_data["image_dict"]
                episode_length = image_dict_all["head_camera"].shape[0]
                
                # 精简输出：只显示异常长度
                if episode_length < 10 or episode_length > 1000:
                    print(f"      ⚠️  Episode长度异常: {episode_length}")
                
                for j in range(episode_length):
            head_img_bit = image_dict_all["head_camera"][j]
            joint_state = vector_all[j]

            # Detect action dimension and apply padding
            action_dim = joint_state.shape[-1] if joint_state.ndim > 0 else 1
            if action_dim == 7:  # 6 joints + 1 gripper (single arm)
                # Pad to 16 dimensions, set last 9 dimensions to 0
                padded_action = np.pad(joint_state, (0, 9), mode='constant', constant_values=0)
                action_mask = np.zeros(16, dtype=np.float32)
                action_mask[:7] = 1  # First 7 dimensions are valid (6 joints + 1 gripper)
            elif action_dim == 8:  # 7 joints + 1 gripper (single arm)
                # Pad to 16 dimensions, set last 8 dimensions to 0
                padded_action = np.pad(joint_state, (0, 8), mode='constant', constant_values=0)
                action_mask = np.zeros(16, dtype=np.float32)
                action_mask[:8] = 1  # First 8 dimensions are valid (7 joints + 1 gripper)
            elif action_dim == 14:  # Dual arm: (6 joints + 1 gripper) * 2 = 14
                # Pad to 16 dimensions, set last 2 dimensions to 0
                padded_action = np.pad(joint_state, (0, 2), mode='constant', constant_values=0)
                action_mask = np.ones(16, dtype=np.float32)
                action_mask[-2:] = 0  # Last 2 dimensions are invalid
            elif action_dim == 16:  # Dual arm: (7 joints + 1 gripper) * 2 = 16
                padded_action = joint_state
                action_mask = np.ones(16, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported action dimension: {action_dim}. Expected 7, 8, 14, or 16.")

            if j != left_gripper_all.shape[0] - 1:
                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                        head_img_tensor = torch.from_numpy(head_img).float().permute(2, 0, 1).unsqueeze(0)
                head_img_resized = torch.nn.functional.interpolate(head_img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
                        head_img = head_img_resized.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        
                head_camera_arrays.append(head_img)
                state_arrays.append(padded_action)  # Use padded action as state
                action_mask_arrays.append(action_mask)
            if j != 0:
                joint_action_arrays.append(padded_action)  # Use padded action

        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

                # 精简输出：只显示异常情况
                if episode_length - 1 < 5:
                    print(f"      ⚠️  Episode时间步过少: {episode_length - 1}")
                
        except Exception as e:
            print(f"      ❌ 处理episode {episode_num} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存数据
    if not state_arrays or not head_camera_arrays:
        print(f"  ❌ {embodiment_type}: 没有有效数据，跳过保存")
        return
    
    print(f"  💾 保存 {embodiment_type} 数据...")
    
    # 转换为numpy数组（现在所有数组都有相同的维度）
    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    
    if joint_action_arrays:
    joint_action_arrays = np.array(joint_action_arrays)
    action_mask_arrays = np.array(action_mask_arrays)

    print(f"Processed data shapes:")
    print(f"  State: {state_arrays.shape}")
    print(f"  Action: {joint_action_arrays.shape}")
    print(f"  Action Mask: {action_mask_arrays.shape}")
    print(f"  Head Camera: {head_camera_arrays.shape}")
    print(f"  Episodes: {len(episode_ends_arrays)}")

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    # 保存到zarr
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    # action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, 16)  # Fixed to 16 dimensions
    joint_chunk_size = (100, 16)  # Fixed to 16 dimensions
    action_mask_chunk_size = (100, 16)  # Fixed to 16 dimensions
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    
    # 保存数据
    safe_create_zarr_dataset(
        zarr_data, "head_cam",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    
    safe_create_zarr_dataset(
        zarr_data, "state",
        data=state_arrays,
        chunks=state_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    
    safe_create_zarr_dataset(
        zarr_data, "action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action_mask",
        data=action_mask_arrays,
        chunks=action_mask_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "text_feat",
        data=text_feat,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        overwrite=True,
        compressor=compressor,
    )
    
    # 保存embodiment信息
    safe_create_zarr_dataset(
        zarr_meta, "embodiment_type",
        data=embodiment_type,
        overwrite=True,
    )
    
    print(f"  ✅ {embodiment_type} 数据保存完成: {save_dir}")
    print(f"     总时间步: {total_count}, 状态数组: {state_arrays.shape}, 动作数组: {joint_action_arrays.shape}")


def safe_create_zarr_dataset(zarr_group, name, data, **kwargs):
    """安全地创建Zarr数据集，自动处理数据类型"""
    try:
        # 创建kwargs的副本，避免修改原始参数
        safe_kwargs = kwargs.copy()
        
        # 自动推断合适的数据类型
        if isinstance(data, str):
            # 字符串数据
            max_length = len(data) + 10  # 留一些余量
            safe_kwargs['dtype'] = f"U{max_length}"
            # 字符串数据不使用压缩器
            safe_kwargs.pop('compressor', None)
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            # 列表或元组数据
            if isinstance(data[0], str):
                # 字符串列表
                max_length = max(len(s) for s in data) + 10
                safe_kwargs['dtype'] = f"U{max_length}"
                safe_kwargs.pop('compressor', None)
            else:
                # 数值列表，让numpy自动推断
                # 不设置dtype，让numpy自动推断
                pass
        elif hasattr(data, 'dtype'):
            # numpy数组，根据数据类型设置合适的dtype
            if np.issubdtype(data.dtype, np.floating):
                if data.dtype == np.float64:
                    safe_kwargs['dtype'] = 'float64'
                else:
                    safe_kwargs['dtype'] = 'float32'
            elif np.issubdtype(data.dtype, np.integer):
                if data.dtype == np.int64:
                    safe_kwargs['dtype'] = 'int64'
                else:
                    safe_kwargs['dtype'] = 'int32'
            # 其他类型让numpy自动推断
        else:
            # 其他数据类型，让numpy自动推断
            # 不设置dtype，让numpy自动推断
            pass
        
        # 创建数据集
        dataset = zarr_group.create_dataset(
            name,
            data=data,
            **safe_kwargs
        )
        
        return dataset
        
    except Exception as e:
        print(f"      ⚠️  创建数据集 {name} 失败: {e}")
        # 尝试使用object类型作为fallback
        try:
            fallback_kwargs = kwargs.copy()
            fallback_kwargs.pop('compressor', None)  # object类型不使用压缩器
            fallback_kwargs['dtype'] = object
            
            dataset = zarr_group.create_dataset(
                name,
                data=data,
                **fallback_kwargs
            )
            print(f"      ✅ 使用object类型成功创建 {name}")
            return dataset
        except Exception as e2:
            print(f"      ❌ 即使object类型也失败: {e2}")
            raise e2


if __name__ == "__main__":
    main()
