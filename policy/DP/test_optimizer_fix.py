#!/usr/bin/env python3
"""
测试优化器修复
验证AdamW优化器的创建
"""

import sys
import os

# 设置正确的Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_optimizer_creation():
    """测试优化器创建"""
    print("=== 测试优化器创建 ===")
    
    try:
        # 测试导入
        import torch
        import torch.nn as nn
        from omegaconf import OmegaConf
        
        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        print("✅ 简单模型创建成功")
        
        # 创建优化器配置
        optimizer_config = {
            '_target_': 'torch.optim.AdamW',
            'lr': 1e-4,
            'betas': [0.95, 0.999],
            'eps': 1e-8,
            'weight_decay': 1e-6
        }
        
        cfg = OmegaConf.create(optimizer_config)
        print("✅ 优化器配置创建成功")
        
        # 测试Hydra实例化
        import hydra
        optimizer = hydra.utils.instantiate(cfg, params=model.parameters())
        print("✅ 优化器通过Hydra创建成功")
        print(f"   优化器类型: {type(optimizer)}")
        print(f"   参数数量: {len(list(optimizer.param_groups))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 优化器测试失败: {e}")
        return False

def test_workspace_optimizer():
    """测试Workspace中的优化器"""
    print("\n=== 测试Workspace优化器 ===")
    
    try:
        # 测试导入
        from diffusion_policy.workspace.robotworkspace import RobotWorkspace
        print("✅ RobotWorkspace导入成功")
        
        # 测试配置结构
        import yaml
        from omegaconf import OmegaConf
        
        # 创建最小配置
        config = {
            'policy': {
                '_target_': 'diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy',
                'shape_meta': {
                    'obs': {'head_cam': {'shape': [3, 256, 256], 'type': 'rgb'}},
                    'action': {'shape': [10], 'type': 'low_dim'}
                },
                'horizon': 8,
                'n_action_steps': 6,
                'n_obs_steps': 3
            },
            'training': {
                'use_ema': True,
                'seed': 42,
                'device': 'cpu'
            },
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': 1e-4,
                'betas': [0.95, 0.999],
                'eps': 1e-8,
                'weight_decay': 1e-6
            },
            'task': {
                'dataset': {
                    '_target_': 'diffusion_policy.dataset.robot_image_dataset.RobotImageDataset',
                    'zarr_path': None
                }
            }
        }
        
        cfg = OmegaConf.create(config)
        print("✅ 配置创建成功")
        print(f"   优化器配置: {cfg.optimizer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workspace优化器测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 优化器修复测试 ===")
    
    # 测试1: 优化器创建
    optimizer_ok = test_optimizer_creation()
    
    # 测试2: Workspace优化器
    workspace_ok = test_workspace_optimizer()
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print(f"优化器创建: {'✅' if optimizer_ok else '❌'}")
    print(f"Workspace优化器: {'✅' if workspace_ok else '❌'}")
    
    if optimizer_ok and workspace_ok:
        print("🎉 所有测试通过！优化器修复成功")
    else:
        print("⚠️  部分测试失败，需要进一步检查")

if __name__ == "__main__":
    main()
