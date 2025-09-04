import os
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
import numpy as np
import glob

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
        root = zarr.open(zarr_path, mode='a')
        episode_ends = root['meta']['episode_ends'][:]
        text_feat = root['data']['text_feat'][:]
        new_len = episode_ends[-1]
        new_text_feat = np.zeros((new_len, text_feat.shape[1]), dtype=text_feat.dtype)
        start = 0
        for i, end in enumerate(episode_ends):
            new_text_feat[start:end] = text_feat[i]
            start = end
        # 删除原有text_feat
        del root['data']['text_feat']
        # 写入新text_feat
        root['data'].create_dataset('text_feat', data=new_text_feat, shape=new_text_feat.shape, dtype=new_text_feat.dtype, overwrite=True)
        print(f"  ✅ Fixed text_feat for {os.path.basename(zarr_path)}")
    except Exception as e:
        print(f"  ❌ Failed to fix text_feat for {os.path.basename(zarr_path)}: {e}")

def extract_endpose_from_state_action(state_data, action_data):
    """
    从state和action数据中提取endpose信息
    假设endpose数据在state和action的末尾部分
    """
    # 定义endpose的维度（位置3维 + 旋转6维 + 夹爪1维 = 10维）
    endpose_dim = 10
    
    # 从state中提取endpose（假设在末尾）
    if state_data.shape[1] >= endpose_dim:
        state_endpose = state_data[:, -endpose_dim:]
    else:
        # 如果state维度不够，用零填充
        padding = np.zeros((state_data.shape[0], endpose_dim - state_data.shape[1]))
        state_endpose = np.concatenate([state_data, padding], axis=1)
    
    # 从action中提取endpose（假设在末尾）
    if action_data.shape[1] >= endpose_dim:
        action_endpose = action_data[:, -endpose_dim:]
    else:
        # 如果action维度不够，用零填充
        padding = np.zeros((action_data.shape[0], endpose_dim - action_data.shape[1]))
        action_endpose = np.concatenate([action_data, padding], axis=1)
    
    return state_endpose, action_endpose

# 创建空buffer
merged_buffer = ReplayBuffer.create_empty_numpy()

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
        
    print(f"=== 处理任务: {task_name} ===")
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
        
        # 修复text_feat
        fix_text_feat(target_folder)
        
        # 加载数据
        try:
            buffer = ReplayBuffer.copy_from_path(
                target_folder,
                keys=["head_cam", "state", "action", "text_feat"],
            )
            
            print(f"    加载了 {buffer.n_episodes} 个episodes")
            print(f"    原始数据形状:")
            for key in ["head_cam", "state", "action", "text_feat"]:
                if key in buffer:
                    print(f"      {key}: {buffer[key].shape}")
            
            # 添加episodes到合并buffer
            added_count = 0
            
            for i in range(buffer.n_episodes):
                try:
                    episode = buffer.get_episode(i, copy=True)
                    
                    # 提取endpose数据，统一维度
                    state_endpose, action_endpose = extract_endpose_from_state_action(
                        episode['state'], episode['action']
                    )
                    
                    # 替换为endpose数据
                    episode['state'] = state_endpose
                    episode['action'] = action_endpose
                    
                    # 扩展 text_feat
                    ep_len = episode['action'].shape[0]
                    text_feat = buffer['text_feat'][i]  # shape: (feat_dim,)
                    episode['text_feat'] = np.tile(text_feat, (ep_len, 1))  # shape: (ep_len, feat_dim)
                    
                    merged_buffer.add_episode(episode)
                    added_count += 1
                    
                except Exception as e:
                    print(f"      ❌ 添加episode {i} 失败: {e}")
                    continue
                
            total_episodes += added_count
            print(f"    成功添加 {added_count} 个episodes")
            print(f"    调整后数据形状:")
            print(f"      state: {state_endpose.shape}, action: {action_endpose.shape}")
            
        except Exception as e:
            print(f"    ❌ 加载 {target_folder} 失败: {e}")
            import traceback
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