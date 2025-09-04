"""
Diffusion Policy Package

A robotics policy framework based on diffusion models.
"""

__version__ = "1.0.0"
__author__ = "RoboTwin Team"

# 导入主要模块
try:
    from .workspace.robotworkspace import RobotWorkspace
    from .dataset.robot_image_dataset import RobotImageDataset
    from .policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    from .model.common.normalizer import LinearNormalizer
except ImportError as e:
    # 如果导入失败，记录警告但不阻止包的使用
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")

# 定义包的公共接口
__all__ = [
    "RobotWorkspace",
    "RobotImageDataset", 
    "DiffusionUnetImagePolicy",
    "LinearNormalizer",
]
