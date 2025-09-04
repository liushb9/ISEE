#!/usr/bin/env python3
"""
直接导入测试脚本
解决模块导入问题
"""

import sys
import os

def test_direct_import():
    """测试直接导入"""
    print("=== 直接导入测试 ===")
    
    # 获取当前脚本的绝对路径
    current_script = os.path.abspath(__file__)
    print(f"当前脚本: {current_script}")
    
    # 获取DP目录的绝对路径
    dp_dir = os.path.dirname(current_script)
    print(f"DP目录: {dp_dir}")
    
    # 检查DP目录内容
    print(f"\nDP目录内容:")
    try:
        for item in os.listdir(dp_dir):
            if os.path.isdir(os.path.join(dp_dir, item)):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  ❌ 无法列出目录内容: {e}")
    
    # 检查diffusion_policy目录
    diffusion_policy_dir = os.path.join(dp_dir, "diffusion_policy")
    print(f"\ndiffusion_policy目录存在: {os.path.exists(diffusion_policy_dir)}")
    
    if os.path.exists(diffusion_policy_dir):
        print("diffusion_policy目录内容:")
        try:
            for item in os.listdir(diffusion_policy_dir):
                item_path = os.path.join(diffusion_policy_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # 检查子目录
                    try:
                        sub_items = os.listdir(item_path)[:5]  # 只显示前5个
                        for sub_item in sub_items:
                            print(f"    - {sub_item}")
                        if len(os.listdir(item_path)) > 5:
                            print(f"    ... 还有 {len(os.listdir(item_path)) - 5} 个文件")
                    except Exception as e:
                        print(f"    ❌ 无法列出子目录内容: {e}")
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"  ❌ 无法列出diffusion_policy目录内容: {e}")
    
    # 尝试不同的导入方式
    print(f"\n=== 尝试不同导入方式 ===")
    
    # 方式1: 直接添加到sys.path
    print("方式1: 直接添加到sys.path")
    try:
        sys.path.insert(0, dp_dir)
        print(f"  ✅ 已添加 {dp_dir} 到sys.path")
        print(f"  当前sys.path[0]: {sys.path[0]}")
    except Exception as e:
        print(f"  ❌ 添加路径失败: {e}")
    
    # 方式2: 尝试导入
    print("\n方式2: 尝试导入模块")
    modules_to_test = [
        "diffusion_policy.workspace.robotworkspace",
        "diffusion_policy.dataset.robot_image_dataset",
        "diffusion_policy.policy.diffusion_unet_image_policy",
        "diffusion_policy.model.common.normalizer",
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"  ✅ {module_name}")
            
            # 尝试获取具体类
            if module_name.endswith('robotworkspace'):
                try:
                    RobotWorkspace = getattr(module, 'RobotWorkspace')
                    print(f"    ✅ 找到RobotWorkspace类")
                except Exception as e:
                    print(f"    ❌ 未找到RobotWorkspace类: {e}")
                    
        except Exception as e:
            print(f"  ❌ {module_name}: {e}")
    
    # 方式3: 检查__init__.py文件
    print(f"\n方式3: 检查__init__.py文件")
    init_files = []
    for root, dirs, files in os.walk(dp_dir):
        for file in files:
            if file == "__init__.py":
                init_files.append(os.path.relpath(os.path.join(root, file), dp_dir))
    
    print(f"找到的__init__.py文件:")
    for init_file in init_files[:10]:  # 只显示前10个
        print(f"  📄 {init_file}")
    if len(init_files) > 10:
        print(f"  ... 还有 {len(init_files) - 10} 个__init__.py文件")
    
    # 方式4: 尝试直接运行Python文件
    print(f"\n方式4: 尝试直接运行Python文件")
    test_file = os.path.join(dp_dir, "diffusion_policy", "workspace", "robotworkspace.py")
    print(f"测试文件: {test_file}")
    print(f"文件存在: {os.path.exists(test_file)}")
    
    if os.path.exists(test_file):
        try:
            with open(test_file, 'r') as f:
                first_lines = f.readlines()[:10]
                print("文件前10行:")
                for i, line in enumerate(first_lines, 1):
                    print(f"  {i:2d}: {line.rstrip()}")
        except Exception as e:
            print(f"  ❌ 无法读取文件: {e}")
    
    return True

def main():
    """主函数"""
    print("=== 模块导入问题诊断 ===")
    
    success = test_direct_import()
    
    print(f"\n=== 诊断完成 ===")
    if success:
        print("✅ 诊断完成，请查看上面的详细信息")
        print("\n建议:")
        print("1. 检查diffusion_policy目录结构")
        print("2. 确认__init__.py文件存在")
        print("3. 检查Python路径设置")
        print("4. 尝试重新安装依赖")
    else:
        print("❌ 诊断过程中出现错误")

if __name__ == "__main__":
    main()
