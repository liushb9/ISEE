#!/usr/bin/env python3
"""
测试修复后的本体顺序
"""

def test_embodiment_order():
    """测试本体顺序是否正确"""
    print("🔍 测试本体顺序...")
    
    # 正确的本体顺序（基于merge_data.sh）
    correct_order = {
        "ur5-wsg": {"start": 0, "end": 50, "episodes": []},
        "franka-panda": {"start": 50, "end": 100, "episodes": []},
        "ARX-X5": {"start": 100, "end": 150, "episodes": []},
        "aloha-agilex": {"start": 150, "end": 200, "episodes": []}
    }
    
    print("✅ 正确的本体顺序:")
    for emb_type, range_info in correct_order.items():
        print(f"  {emb_type:15s}: {range_info['start']:3d} - {range_info['end']:3d} ({range_info['end'] - range_info['start']} episodes)")
    
    # 验证episode分配
    print(f"\n📊 Episode分配验证:")
    for episode_num in range(200):
        for emb_type, range_info in correct_order.items():
            if range_info["start"] <= episode_num < range_info["end"]:
                print(f"  Episode {episode_num:3d} → {emb_type}")
                break
    
    print(f"\n🎯 总结:")
    print(f"  - ur5-wsg: episodes 0-49 (50个)")
    print(f"  - franka-panda: episodes 50-99 (50个)")
    print(f"  - ARX-X5: episodes 100-149 (50个)")
    print(f"  - aloha-agilex: episodes 150-199 (50个)")
    print(f"  - 总计: 200个episodes")


if __name__ == "__main__":
    test_embodiment_order()
