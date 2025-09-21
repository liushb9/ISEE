"""
数据预处理工具 - 支持不同自由度的机械臂数据处理

该模块提供按需填充机制的数据预处理功能，能够处理：
- 6自由度机械臂 + 夹爪 (7维)
- 7自由度机械臂 + 夹爪 (8维)

基于RDT论文的按需填充机制，使用0-1掩码向量区分真实数据和填充值。
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from configs.state_vec import create_dynamic_arm_indices, create_bimanual_indices


class RoboticDataPreprocessor:
    """
    机器人数据预处理器
    支持动态自由度的按需填充机制
    """

    def __init__(self, state_min: Optional[np.ndarray] = None, state_max: Optional[np.ndarray] = None):
        """
        初始化预处理器

        Args:
            state_min: 状态最小值数组
            state_max: 状态最大值数组
        """
        self.state_min = state_min
        self.state_max = state_max

    def detect_arm_dof(self, joint_data: np.ndarray) -> int:
        """
        自动检测机械臂自由度

        Args:
            joint_data: 关节数据，shape为 (..., D)

        Returns:
            int: 机械臂自由度 (6 或 7)
        """
        joint_dim = joint_data.shape[-1]

        # 假设数据格式为 [joint0, joint1, ..., jointN, gripper]
        # 所以自由度 = 总维度 - 1
        arm_dof = joint_dim - 1

        if arm_dof not in [6, 7]:
            raise ValueError(f"检测到不支持的自由度: {arm_dof}，当前支持6或7自由度")

        return arm_dof

    def preprocess_joint_data(self, joint_data: np.ndarray, arm_dof: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理关节数据为统一格式

        Args:
            joint_data: 原始关节数据，shape为 (..., D) where D = arm_dof + 1
            arm_dof: 机械臂自由度，如果为None则自动检测

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - processed_data: 处理后的数据 (..., D)
                - mask: 掩码数组，1表示有效数据，0表示填充数据 (D,)
        """
        if arm_dof is None:
            arm_dof = self.detect_arm_dof(joint_data)

        # 创建动态索引
        dynamic_indices = create_dynamic_arm_indices(arm_dof, "right")

        # 创建掩码
        mask = np.zeros(len(dynamic_indices), dtype=np.float32)
        mask[:len(dynamic_indices)] = 1.0

        # 数据归一化（如果提供了统计信息）
        if self.state_min is not None and self.state_max is not None:
            # 只对有效的维度进行归一化
            processed_data = joint_data.copy()
            for i, idx in enumerate(dynamic_indices):
                if i < joint_data.shape[-1]:
                    processed_data[..., i] = (joint_data[..., i] - self.state_min[idx]) / (self.state_max[idx] - self.state_min[idx]) * 2 - 1
        else:
            processed_data = joint_data

        return processed_data, mask

    def format_to_unified_state(self, joint_data: np.ndarray, arm_dof: Optional[int] = None,
                               state_dim: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        将关节数据格式化为统一的状态向量

        Args:
            joint_data: 关节数据，shape为 (..., D) where D = arm_dof + 1
            arm_dof: 机械臂自由度
            state_dim: 统一状态向量维度 (默认128)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - unified_state: 统一状态向量 (..., 128)
                - state_mask: 状态掩码 (128,)
        """
        if arm_dof is None:
            arm_dof = self.detect_arm_dof(joint_data)

        # 创建动态索引
        dynamic_indices = create_dynamic_arm_indices(arm_dof, "right")

        # 初始化统一状态向量
        original_shape = joint_data.shape[:-1]  # 除去最后一个维度
        unified_state = np.zeros((*original_shape, state_dim), dtype=np.float32)

        # 按需填充数据
        for i, idx in enumerate(dynamic_indices):
            if i < joint_data.shape[-1]:
                unified_state[..., idx] = joint_data[..., i]

        # 创建状态掩码
        state_mask = np.zeros(state_dim, dtype=np.float32)
        state_mask[dynamic_indices] = 1.0

        return unified_state, state_mask

    def postprocess_actions(self, unified_actions: np.ndarray, arm_dof: int,
                           state_min: np.ndarray, state_max: np.ndarray) -> np.ndarray:
        """
        将统一动作向量转换回关节动作

        Args:
            unified_actions: 统一动作向量 (..., 128)
            arm_dof: 机械臂自由度
            state_min: 状态最小值数组
            state_max: 状态最大值数组

        Returns:
            joint_actions: 关节动作 (..., D) where D = arm_dof + 1
        """
        # 创建动态索引
        dynamic_indices = create_dynamic_arm_indices(arm_dof, "right")

        # 提取有效动作
        joint_actions = unified_actions[..., dynamic_indices]

        # 反归一化
        joint_actions = (joint_actions + 1) / 2 * (state_max[dynamic_indices] - state_min[dynamic_indices]) + state_min[dynamic_indices]

        return joint_actions


def create_data_processor_from_stats(stats_dict: Dict[str, Any]) -> RoboticDataPreprocessor:
    """
    从统计信息创建数据处理器

    Args:
        stats_dict: 包含state_min和state_max的统计字典

    Returns:
        RoboticDataPreprocessor: 配置好的数据处理器
    """
    state_min = np.array(stats_dict.get("state_min", []))
    state_max = np.array(stats_dict.get("state_max", []))

    return RoboticDataPreprocessor(state_min=state_min, state_max=state_max)


def validate_data_format(data_path: str) -> Dict[str, Any]:
    """
    验证和分析数据格式

    Args:
        data_path: 数据文件路径

    Returns:
        Dict: 数据格式信息
    """
    import h5py

    info = {
        "datasets": [],
        "joint_dimensions": {},
        "total_episodes": 0
    }

    try:
        with h5py.File(data_path, 'r') as f:
            # 检查数据结构
            for key in f.keys():
                info["datasets"].append(key)

                if hasattr(f[key], 'keys'):
                    # 检查joint_action结构
                    if 'joint_action' in f[key]:
                        joint_action_group = f[key]['joint_action']
                        for subkey in joint_action_group.keys():
                            if subkey in ['left_arm', 'right_arm']:
                                shape = joint_action_group[subkey].shape
                                info["joint_dimensions"][subkey] = shape

        # 分析关节维度
        if 'right_arm' in info["joint_dimensions"]:
            arm_shape = info["joint_dimensions"]["right_arm"]
            if len(arm_shape) >= 2:
                info["detected_dof"] = arm_shape[-1]  # 最后一个维度是关节数

    except Exception as e:
        info["error"] = str(e)

    return info


# 示例用法
if __name__ == "__main__":
    # 示例：处理6关节机械臂数据
    print("=== RDT 按需填充机制示例 ===")

    # 模拟6关节机械臂数据 (6关节 + 1夹爪 = 7维)
    joint_data_6dof = np.random.randn(10, 7).astype(np.float32)  # 10个时间步
    print(f"6DOF关节数据shape: {joint_data_6dof.shape}")

    # 创建预处理器
    processor = RoboticDataPreprocessor()

    # 格式化为统一状态向量
    unified_state, state_mask = processor.format_to_unified_state(joint_data_6dof, arm_dof=6)
    print(f"统一状态向量shape: {unified_state.shape}")
    print(f"状态掩码 (前20维): {state_mask[:20]}")
    print(f"有效数据位置: {np.where(state_mask == 1)[0]}")

    # 模拟7关节机械臂数据 (7关节 + 1夹爪 = 8维)
    joint_data_7dof = np.random.randn(10, 8).astype(np.float32)
    print(f"\n7DOF关节数据shape: {joint_data_7dof.shape}")

    unified_state_7dof, state_mask_7dof = processor.format_to_unified_state(joint_data_7dof, arm_dof=7)
    print(f"7DOF统一状态向量shape: {unified_state_7dof.shape}")
    print(f"7DOF状态掩码 (前20维): {state_mask_7dof[:20]}")
    print(f"7DOF有效数据位置: {np.where(state_mask_7dof == 1)[0]}")

    print("\n=== 按需填充机制验证 ===")
    print("6DOF数据填充位置 (关节0-5 + 夹爪):", [0, 1, 2, 3, 4, 5, 10])
    print("7DOF数据填充位置 (关节0-6 + 夹爪):", [0, 1, 2, 3, 4, 5, 6, 10])
    print("未使用的维度自动填充为0，掩码相应位置为0")
