#!/usr/bin/env python3
"""
测试包结构
验证diffusion_policy包是否正确初始化
"""

import sys
import os

# 设置正确的Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_package_import():
    """测试包导入"""
    print("=== 测试包导入 ===")
    
    try:
        # 测试主包导入
        import diffusion_policy
        print(f"✅ 主包导入成功: {diffusion_policy.__version__}")
        
        # 测试子模块导入
        from diffusion_policy.workspace import RobotWorkspace
        print("✅ RobotWorkspace导入成功")
        
        from diffusion_policy.dataset import RobotImageDataset
        print("✅ RobotImageDataset导入成功")
        
        from diffusion_policy.policy import DiffusionUnetImagePolicy
        print("✅ DiffusionUnetImagePolicy导入成功")
        
        from diffusion_policy.model.common import LinearNormalizer
        print("✅ LinearNormalizer导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 包导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_import():
    """测试直接导入"""
    print("\n=== 测试直接导入 ===")
    
    try:
        # 测试直接导入
        from diffusion_policy.workspace.robotworkspace import RobotWorkspace
        print("✅ 直接导入RobotWorkspace成功")
        
        from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset
        print("✅ 直接导入RobotImageDataset成功")
        
        from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
        print("✅ 直接导入DiffusionUnetImagePolicy成功")
        
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        print("✅ 直接导入LinearNormalizer成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 直接导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_info():
    """测试包信息"""
    print("\n=== 测试包信息 ===")
    
    try:
        import diffusion_policy
        
        print(f"包名称: {diffusion_policy.__name__}")
        print(f"包版本: {diffusion_policy.__version__}")
        print(f"包作者: {diffusion_policy.__author__}")
        
        if hasattr(diffusion_policy, '__all__'):
            print(f"公共接口: {diffusion_policy.__all__}")
        else:
            print("⚠️  没有定义__all__")
        
        return True
        
    except Exception as e:
        print(f"❌ 包信息获取失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 包结构测试 ===")
    
    # 测试包导入
    package_ok = test_package_import()
    
    # 测试直接导入
    direct_ok = test_direct_import()
    
    # 测试包信息
    info_ok = test_package_info()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"包导入: {'✅' if package_ok else '❌'}")
    print(f"直接导入: {'✅' if direct_ok else '❌'}")
    print(f"包信息: {'✅' if info_ok else '❌'}")
    
    if all([package_ok, direct_ok, info_ok]):
        print("\n🎉 包结构完全正常！")
        print("现在可以运行多卡训练测试了:")
        print("python test_simple_multigpu.py")
    else:
        print("\n⚠️  包结构有问题，需要进一步检查")
        
        if not package_ok:
            print("建议: 检查__init__.py文件")
        if not direct_ok:
            print("建议: 检查模块文件路径")
        if not info_ok:
            print("建议: 检查包元数据")

if __name__ == "__main__":
    main()
