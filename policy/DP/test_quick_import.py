#!/usr/bin/env python3
"""
快速导入测试
验证修复后的导入逻辑
"""

import sys
import os

# 设置正确的Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试导入"""
    print("=== 快速导入测试 ===")
    
    # 测试1: 直接导入
    print("\n1. 测试直接导入:")
    try:
        from diffusion_policy.workspace import RobotWorkspace
        print("✅ RobotWorkspace导入成功")
    except Exception as e:
        print(f"❌ RobotWorkspace导入失败: {e}")
    
    try:
        from diffusion_policy.dataset import RobotImageDataset
        print("✅ RobotImageDataset导入成功")
    except Exception as e:
        print(f"❌ RobotImageDataset导入失败: {e}")
    
    try:
        from diffusion_policy.policy import DiffusionUnetImagePolicy
        print("✅ DiffusionUnetImagePolicy导入成功")
    except Exception as e:
        print(f"❌ DiffusionUnetImagePolicy导入失败: {e}")
    
    try:
        from diffusion_policy.model.common import LinearNormalizer
        print("✅ LinearNormalizer导入成功")
    except Exception as e:
        print(f"❌ LinearNormalizer导入失败: {e}")
    
    # 测试2: 使用__import__
    print("\n2. 测试__import__:")
    modules_to_test = [
        ("diffusion_policy.workspace", "RobotWorkspace"),
        ("diffusion_policy.dataset", "RobotImageDataset"),
        ("diffusion_policy.policy", "DiffusionUnetImagePolicy"),
        ("diffusion_policy.model.common", "LinearNormalizer"),
    ]
    
    for module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            print(f"✅ {module_path}.{class_name} 通过__import__成功")
        except Exception as e:
            print(f"❌ {module_path}.{class_name} 通过__import__失败: {e}")
    
    # 测试3: 检查包结构
    print("\n3. 检查包结构:")
    try:
        import diffusion_policy
        print(f"✅ 主包: {diffusion_policy.__name__}")
        print(f"   版本: {diffusion_policy.__version__}")
        print(f"   作者: {diffusion_policy.__author__}")
        
        if hasattr(diffusion_policy, '__all__'):
            print(f"   公共接口: {diffusion_policy.__all__}")
        
    except Exception as e:
        print(f"❌ 主包检查失败: {e}")

def main():
    """主函数"""
    test_imports()
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()
