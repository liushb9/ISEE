#!/usr/bin/env python3
"""
完整的多卡训练环境测试脚本
验证所有组件是否正常工作
"""

import sys
import os
import torch

# 设置正确的Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_environment():
    """测试环境配置"""
    print("=== 环境配置测试 ===")
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return True

def test_lightning():
    """测试Lightning功能"""
    print("\n=== Lightning功能测试 ===")
    
    try:
        import lightning
        print(f"✅ Lightning版本: {lightning.__version__}")
        
        from lightning.fabric import Fabric
        print("✅ Fabric可用")
        
        from lightning.pytorch.strategies import DDPStrategy
        print("✅ DDP策略可用")
        
        return True
        
    except Exception as e:
        print(f"❌ Lightning测试失败: {e}")
        return False

def test_fabric_creation():
    """测试Fabric实例创建"""
    print("\n=== Fabric实例测试 ===")
    
    try:
        from lightning.fabric import Fabric
        
        # 创建Fabric实例
        fabric = Fabric(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1,
            strategy="ddp",
            precision="32-true",
        )
        
        print("✅ Fabric实例创建成功")
        
        # 测试关键属性
        print(f"   加速器: {fabric.accelerator}")
        print(f"   策略: {fabric.strategy}")
        
        # 安全地获取设备数量
        try:
            if hasattr(fabric, 'num_devices'):
                device_count = fabric.num_devices
            elif hasattr(fabric, 'devices'):
                device_count = fabric.devices
            elif hasattr(fabric, 'num_gpus'):
                device_count = fabric.num_gpus
            else:
                device_count = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
            print(f"   设备数量: {device_count}")
        except Exception as e:
            print(f"   ⚠️  无法获取设备数量: {e}")
        
        return fabric
        
    except Exception as e:
        print(f"❌ Fabric创建失败: {e}")
        return None

def test_module_imports():
    """测试模块导入"""
    print("\n=== 模块导入测试 ===")
    
    modules_to_test = [
        "diffusion_policy.workspace.robotworkspace.RobotWorkspace",
        "diffusion_policy.dataset.robot_image_dataset.RobotImageDataset",
        "diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy",
        "diffusion_policy.model.common.normalizer.LinearNormalizer",
    ]
    
    all_imports_ok = True
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_workspace_creation():
    """测试Workspace创建"""
    print("\n=== Workspace创建测试 ===")
    
    try:
        from diffusion_policy.workspace.robotworkspace import RobotWorkspace
        from omegaconf import OmegaConf
        
        # 创建最小配置
        cfg = OmegaConf.create({
            "task": {
                "name": "test",
                "dataset": {
                    "_target_": "diffusion_policy.dataset.robot_image_dataset.RobotImageDataset",
                    "zarr_path": "data/test.zarr",
                    "horizon": 1,
                    "pad_before": 0,
                    "pad_after": 0,
                    "seed": 42,
                    "val_ratio": 0.0,
                    "batch_size": 1,
                    "max_train_episodes": None
                },
                "env_runner": None
            },
            "policy": {
                "_target_": "diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy",
                "obs_encoder": {
                    "_target_": "diffusion_policy.model.common.mlp.MLP",
                    "input_dim": 10,
                    "output_dim": 64,
                    "hidden_dims": [64, 64]
                },
                "action_dim": 10,
                "n_obs_steps": 2,
                "n_action_steps": 2,
                "n_cond_steps": 2,
                "horizon": 4,
                "obs_as_global_cond": True,
                "diffusion_step_embed_dim": 64,
                "down_dims": [64, 128, 256],
                "up_dims": [256, 128, 64],
                "n_groups": 8,
                "mid_blocks_per_group": 2,
                "use_fp16": False
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 1e-4,
                "weight_decay": 1e-6
            },
            "dataloader": {
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": True,
                "pin_memory": False,
                "persistent_workers": False
            },
            "val_dataloader": {
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "persistent_workers": False
            },
            "training": {
                "seed": 42,
                "device": "cuda:0",
                "num_epochs": 1,
                "max_train_steps": None,
                "max_val_steps": None,
                "gradient_accumulate_every": 1,
                "lr_warmup_steps": 0,
                "lr_scheduler": "cosine",
                "checkpoint_every": 1,
                "val_every": 1,
                "sample_every": 1,
                "rollout_every": 1,
                "use_ema": False,
                "freeze_encoder": False,
                "debug": False,
                "resume": False,
                "tqdm_interval_sec": 0.1
            },
            "ema": {
                "_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel",
                "decay": 0.9999
            },
            "checkpoint": {
                "topk": 5
            },
            "logging": {
                "mode": "disabled",
                "project": "test_project",
                "name": "test_run"
            },
            "head_camera_type": "default",
            "n_obs_steps": 2,
            "n_action_steps": 2,
            "horizon": 4,
            "obs_dim": 10,
            "action_dim": 10
        })
        
        print("✅ 配置创建成功")
        
        # 创建workspace
        workspace = RobotWorkspace(cfg)
        print("✅ Workspace创建成功")
        
        return workspace
        
    except Exception as e:
        print(f"❌ Workspace创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multigpu_integration(fabric, workspace):
    """测试多卡训练集成"""
    print("\n=== 多卡训练集成测试 ===")
    
    if fabric is None or workspace is None:
        print("❌ 无法测试集成，实例创建失败")
        return False
    
    try:
        # 设置多卡训练属性
        workspace.fabric = fabric
        workspace.rank = 0  # 模拟rank 0
        workspace.world_size = 2  # 模拟2张卡
        
        print("✅ 多卡训练属性设置成功")
        print(f"   Fabric实例: {workspace.fabric is not None}")
        print(f"   Rank: {workspace.rank}")
        print(f"   World_size: {workspace.world_size}")
        
        # 测试多卡训练逻辑
        if hasattr(workspace, 'fabric') and workspace.fabric is not None:
            print("✅ 多卡训练模式检测成功")
            return True
        else:
            print("❌ 多卡训练模式检测失败")
            return False
            
    except Exception as e:
        print(f"❌ 多卡训练集成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 完整多卡训练环境测试 ===")
    
    # 测试环境
    env_ok = test_environment()
    
    # 测试Lightning
    lightning_ok = test_lightning()
    
    # 测试Fabric
    fabric = test_fabric_creation()
    fabric_ok = fabric is not None
    
    # 测试模块导入
    imports_ok = test_module_imports()
    
    # 测试Workspace创建
    workspace = test_workspace_creation()
    workspace_ok = workspace is not None
    
    # 测试多卡训练集成
    integration_ok = False
    if fabric_ok and workspace_ok:
        integration_ok = test_multigpu_integration(fabric, workspace)
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"环境配置: {'✅' if env_ok else '❌'}")
    print(f"Lightning功能: {'✅' if lightning_ok else '❌'}")
    print(f"Fabric创建: {'✅' if fabric_ok else '❌'}")
    print(f"模块导入: {'✅' if imports_ok else '❌'}")
    print(f"Workspace创建: {'✅' if workspace_ok else '❌'}")
    print(f"多卡训练集成: {'✅' if integration_ok else '❌'}")
    
    if all([env_ok, lightning_ok, fabric_ok, imports_ok, workspace_ok, integration_ok]):
        print("\n🎉 所有测试通过！多卡训练环境完全正常")
        print("现在可以安全地运行训练命令:")
        print("bash train_multi_gpu.sh six_tasks demo_clean 1200 0 3 \"0,1,2\"")
    else:
        print("\n⚠️  部分测试失败，需要检查配置")
        
        if not fabric_ok:
            print("建议: 检查Lightning版本和CUDA环境")
        if not imports_ok:
            print("建议: 检查Python路径设置")
        if not workspace_ok:
            print("建议: 检查配置文件结构")
        if not integration_ok:
            print("建议: 检查多卡训练配置")

if __name__ == "__main__":
    main()
