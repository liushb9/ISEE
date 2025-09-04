import os
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
import numpy as np
import glob
import traceback

# 获取data目录下所有的zarr文件夹
data_dir = "data"
zarr_folders = glob.glob(os.path.join(data_dir, "*-demo_clean-*-50.zarr"))

# 按任务分组
task_groups = {}
for folder in zarr_folders:
    # 从文件夹名提取任务名，例如：stack_blocks_three-demo_clean-aloha-agilex-50.zarr
    folder_name = os.path.basename(folder)
    parts = folder_name.split('-')
    if len(parts) >= 4:
        task_name = parts[0]  # 例如：stack_blocks_three
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(folder)

print("=== 发现的任务和本体 ===")
for task_name, folders in task_groups.items():
    print(f"{task_name}:")
    for folder in folders:
        print(f"  {os.path.basename(folder)}")
    print()

# 合并后保存的路径
save_path = "data/six-tasks.zarr"

# 如果目标已存在，先删除
if os.path.exists(save_path):
    import shutil
    shutil.rmtree(save_path)

def fix_text_feat(zarr_path):
    """修复text_feat的维度问题"""
    try:
        print(f"    🔧 开始修复text_feat...")
        root = zarr.open(zarr_path, mode='a')
        
        print(f"    📊 读取episode_ends...")
        episode_ends = root['meta']['episode_ends'][:]
        print(f"      episode_ends shape: {episode_ends.shape}, 值: {episode_ends}")
        
        print(f"    📊 读取text_feat...")
        text_feat = root['data']['text_feat'][:]
        print(f"      text_feat shape: {text_feat.shape}, dtype: {text_feat.dtype}")
        
        new_len = episode_ends[-1]
        print(f"    📏 计算新长度: {new_len}")
        
        new_text_feat = np.zeros((new_len, text_feat.shape[1]), dtype=text_feat.dtype)
        print(f"    🆕 创建新数组: {new_text_feat.shape}")
        
        start = 0
        for i, end in enumerate(episode_ends):
            new_text_feat[start:end] = text_feat[i]
            start = end
        
        print(f"    🗑️  删除原有text_feat...")
        del root['data']['text_feat']
        
        print(f"    💾 写入新text_feat...")
        root['data'].create_dataset('text_feat', data=new_text_feat, shape=new_text_feat.shape, dtype=new_text_feat.dtype, overwrite=True)
        
        print(f"  ✅ Fixed text_feat for {os.path.basename(zarr_path)}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to fix text_feat for {os.path.basename(zarr_path)}: {e}")
        print(f"    详细错误信息:")
        traceback.print_exc()
        return False

# 创建空buffer
print("🆕 创建空的ReplayBuffer...")
merged_buffer = ReplayBuffer.create_empty_numpy()
print(f"✅ 空buffer创建成功")

# 记录第一个episode的维度作为参考
reference_dimensions = None

# 按任务顺序处理
task_order = [
    "stack_blocks_three",
    "stack_bowls_three", 
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "hanging_mug",
    "place_cans_plasticbox"
]

total_episodes = 0

for task_name in task_order:
    if task_name not in task_groups:
        print(f"⚠️  任务 {task_name} 未找到数据")
        continue
        
    print(f"\n=== 处理任务: {task_name} ===")
    task_folders = task_groups[task_name]
    
    # 按本体顺序处理（与process_data.py中的顺序一致）
    embodiment_order = ["ur5-wsg", "franka-panda", "ARX-X5", "aloha-agilex"]
    
    for embodiment in embodiment_order:
        # 找到对应本体的文件夹
        target_folder = None
        for folder in task_folders:
            if embodiment in folder:
                target_folder = folder
                break
        
        if target_folder is None:
            print(f"  ⚠️  本体 {embodiment} 的数据未找到")
            continue
            
        print(f"  处理本体: {embodiment}")
        print(f"    文件夹: {os.path.basename(target_folder)}")
        
        # 修复text_feat
        fix_success = fix_text_feat(target_folder)
        if not fix_success:
            print(f"    ⚠️  text_feat修复失败，跳过此本体")
            continue
        
        # 加载数据
        try:
            print(f"    📥 开始加载ReplayBuffer...")
            buffer = ReplayBuffer.copy_from_path(
                target_folder,
                keys=["head_cam", "state", "action", "text_feat"],
            )
            
            print(f"    加载了 {buffer.n_episodes} 个episodes")
            print(f"    数据形状:")
            for key in ["head_cam", "state", "action", "text_feat"]:
                if key in buffer:
                    print(f"      {key}: {buffer[key].shape}")
            
            # 检查维度一致性
            if reference_dimensions is None:
                # 第一个episode，记录参考维度
                reference_dimensions = {}
                for key in ["state", "action"]:
                    if key in buffer:
                        reference_dimensions[key] = buffer[key].shape[1:]
                print(f"    📏 设置参考维度: {reference_dimensions}")
            else:
                # 检查维度是否匹配
                dimension_mismatch = False
                for key in ["state", "action"]:
                    if key in buffer and key in reference_dimensions:
                        if buffer[key].shape[1:] != reference_dimensions[key]:
                            print(f"    ❌ 维度不匹配: {key} 期望 {reference_dimensions[key]}, 实际 {buffer[key].shape[1:]}")
                            dimension_mismatch = True
                
                if dimension_mismatch:
                    print(f"    ⚠️  跳过 {embodiment}，维度不匹配")
                    continue
            
            # 添加episodes到合并buffer
            print(f"    🔄 开始添加episodes到合并buffer...")
            added_count = 0
            
            for i in range(buffer.n_episodes):
                try:
                    episode = buffer.get_episode(i, copy=True)
                    # 扩展 text_feat
                    ep_len = episode['action'].shape[0]
                    text_feat = buffer['text_feat'][i]  # shape: (feat_dim,)
                    episode['text_feat'] = np.tile(text_feat, (ep_len, 1))  # shape: (ep_len, feat_dim)
                    
                    merged_buffer.add_episode(episode)
                    added_count += 1
                    
                except Exception as e:
                    print(f"      ❌ 添加episode {i} 失败: {e}")
                    traceback.print_exc()
                    continue
                
            total_episodes += added_count
            print(f"    成功添加 {added_count} 个episodes")
            
        except Exception as e:
            print(f"    ❌ 加载 {target_folder} 失败: {e}")
            print(f"    详细错误信息:")
            traceback.print_exc()

print(f"\n=== 合并完成 ===")
print(f"总共处理了 {total_episodes} 个episodes")

if total_episodes > 0:
    print(f"合并后的数据形状:")
    print(f"  head_cam: {merged_buffer['head_cam'].shape}")
    print(f"  state: {merged_buffer['state'].shape}")
    print(f"  action: {merged_buffer['action'].shape}")
    print(f"  text_feat: {merged_buffer['text_feat'].shape}")

    # 保存到zarr
    print(f"\n保存合并后的数据到 {save_path}...")
    try:
        merged_buffer.save_to_path(save_path)
        print(f"✅ 成功保存到 {save_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        traceback.print_exc()
else:
    print("❌ 没有成功处理任何episodes，无法保存")
