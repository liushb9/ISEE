#!/usr/bin/env python3
"""
简化的多卡训练测试脚本
只测试核心功能，避免复杂的配置问题
"""

import sys
import os
import torch

# 添加路径
sys.path.append('/home/shengbang/RoboTwin/policy/DP')

def test_basic_fabric():
    """测试基本的Fabric功能"""
    print("=== 测试基本Fabric功能 ===")
    
    try:
        from lightning.fabric import Fabric
        
        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        print(f"CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"GPU数量: {device_count}")
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
        
        # 创建Fabric实例
        if cuda_available and device_count > 0:
            fabric = Fabric(
                accelerator="cuda",
                devices=min(2, device_count),  # 最多使用2张卡
                strategy="ddp",
                precision="32-true",
            )
        else:
            fabric = Fabric(
                accelerator="cpu",
                devices=1,
                strategy="ddp",
                precision="32-true",
            )
        
        print(f"✅ Fabric实例创建成功")
        print(f"   加速器: {fabric.accelerator}")
        
        # 修复：使用正确的属性获取设备数量
        try:
            # 尝试不同的属性名
            if hasattr(fabric, 'num_devices'):
                device_count_fabric = fabric.num_devices
            elif hasattr(fabric, 'devices'):
                device_count_fabric = fabric.devices
            elif hasattr(fabric, 'num_gpus'):
                device_count_fabric = fabric.num_gpus
            else:
                # 如果都没有，使用创建时传入的值
                device_count_fabric = min(2, device_count) if cuda_available else 1
                print(f"   ⚠️  无法获取设备数量，使用创建时的值: {device_count_fabric}")
        except Exception as e:
            device_count_fabric = min(2, device_count) if cuda_available else 1
            print(f"   ⚠️  获取设备数量失败，使用创建时的值: {device_count_fabric}")
        
        print(f"   设备数量: {device_count_fabric}")
        print(f"   策略: {fabric.strategy}")
        
        # 修复：安全地获取精度信息
        try:
            if hasattr(fabric, 'precision'):
                precision_info = fabric.precision
            elif hasattr(fabric, '_precision'):
                precision_info = fabric._precision
            elif hasattr(fabric, 'config'):
                precision_info = getattr(fabric.config, 'precision', 'unknown')
            else:
                precision_info = '32-true (创建时设置)'
            print(f"   精度: {precision_info}")
        except Exception as e:
            print(f"   ⚠️  无法获取精度信息: {e}")
            print(f"   精度: 32-true (创建时设置)")
        
        return fabric
        
    except Exception as e:
        print(f"❌ Fabric测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_imports():
    """测试关键模块导入"""
    print("\n=== 测试模块导入 ===")
    
    # 检查Python路径
    print(f"当前Python路径:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # 检查当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查DP目录是否存在
    dp_dir = "/home/shengbang/RoboTwin/policy/DP"
    print(f"DP目录存在: {os.path.exists(dp_dir)}")
    
    # 使用正确的导入方式
    modules_to_test = [
        ("diffusion_policy.workspace", "RobotWorkspace"),
        ("diffusion_policy.dataset", "RobotImageDataset"),
        ("diffusion_policy.policy", "DiffusionUnetImagePolicy"),
        ("diffusion_policy.model.common", "LinearNormalizer"),
    ]
    
    all_imports_ok = True
    
    for module_path, class_name in modules_to_test:
        try:
            # 先导入模块
            module = __import__(module_path, fromlist=[class_name])
            # 然后获取类
            class_obj = getattr(module, class_name)
            print(f"✅ {module_path}.{class_name}")
        except Exception as e:
            print(f"❌ {module_path}.{class_name}: {e}")
            
            # 尝试从DP目录导入
            try:
                sys.path.insert(0, dp_dir)
                module = __import__(module_path, fromlist=[class_name])
                class_obj = getattr(module, class_name)
                print(f"   ✅ 从DP目录导入成功")
                all_imports_ok = True
            except Exception as e2:
                print(f"   ❌ 从DP目录导入也失败: {e2}")
                all_imports_ok = False
            finally:
                # 恢复原始路径
                if dp_dir in sys.path:
                    sys.path.remove(dp_dir)
    
    return all_imports_ok

def test_config_structure():
    """测试配置结构"""
    print("\n=== 测试配置结构 ===")
    
    try:
        from omegaconf import OmegaConf
        
        # 创建最小配置
        cfg = OmegaConf.create({
            "task": {
                "name": "test",
                "dataset": {
                    "zarr_path": "data/test.zarr"
                }
            },
            "training": {
                "seed": 42,
                "device": "cuda:0"
            }
        })
        
        print(f"✅ 配置创建成功")
        print(f"   配置结构: {list(cfg.keys())}")
        
        return cfg
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return None

def test_lightning_version():
    """测试Lightning版本"""
    print("\n=== 测试Lightning版本 ===")
    
    try:
        import lightning
        print(f"✅ Lightning版本: {lightning.__version__}")
        
        # 检查关键组件
        from lightning.fabric import Fabric
        print(f"✅ Fabric可用")
        
        from lightning.pytorch.strategies import DDPStrategy
        print(f"✅ DDP策略可用")
        
        return True
        
    except Exception as e:
        print(f"❌ Lightning测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 简化多卡训练测试 ===")
    
    # 测试Lightning版本
    lightning_ok = test_lightning_version()
    
    # 测试模块导入
    imports_ok = test_imports()
    
    # 测试配置结构
    config_ok = test_config_structure()
    
    # 测试Fabric
    fabric_ok = test_basic_fabric()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"Lightning版本: {'✅' if lightning_ok else '❌'}")
    print(f"模块导入: {'✅' if imports_ok else '❌'}")
    print(f"配置结构: {'✅' if config_ok else '❌'}")
    print(f"Fabric功能: {'✅' if fabric_ok else '❌'}")
    
    if all([lightning_ok, imports_ok, config_ok, fabric_ok]):
        print("\n🎉 所有测试通过！多卡训练环境配置正常")
        print("现在可以运行训练命令:")
        print("bash train_multi_gpu.sh six_tasks demo_clean 1200 0 3 \"0,1,2\"")
    else:
        print("\n⚠️  部分测试失败，需要检查环境配置")
        
        if not lightning_ok:
            print("建议: pip install --upgrade lightning")
        if not imports_ok:
            print("建议: 检查Python路径和模块安装")
        if not fabric_ok:
            print("建议: 检查CUDA环境和Lightning版本")

if __name__ == "__main__":
    main()
