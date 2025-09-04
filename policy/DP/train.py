"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from lightning.fabric import Fabric
import torch
import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def cleanup_previous_processes():
    """清理之前的训练进程"""
    try:
        import psutil
        current_pid = os.getpid()
        
        # 查找可能的Python训练进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    # 只清理包含train.py但不是当前进程的进程
                    if 'train.py' in cmdline and proc.pid != current_pid:
                        print(f"发现之前的训练进程: PID {proc.pid}")
                        # 检查进程是否还在运行
                        if proc.is_running():
                            print(f"终止之前的训练进程: PID {proc.pid}")
                            try:
                                proc.terminate()
                                proc.wait(timeout=3)  # 减少等待时间
                            except psutil.TimeoutExpired:
                                print(f"强制终止进程: PID {proc.pid}")
                                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except ImportError:
        print("psutil未安装，跳过进程清理")
    except Exception as e:
        print(f"进程清理出错: {e}")

def find_free_port(start_port=10000, max_attempts=100):
    """查找可用端口"""
    import socket
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError("无法找到可用端口")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # 设置PyTorch默认数据类型为float32，避免double/float类型不匹配
    torch.set_default_dtype(torch.float32)
    
    # 清理之前的训练进程
    # cleanup_previous_processes()
    
    # 设置多卡训练环境变量，使用动态端口避免冲突
    os.environ["MASTER_ADDR"] = "localhost"
    # 查找可用端口
    port = find_free_port()
    os.environ["MASTER_PORT"] = str(port)
    print(f"使用端口: {port}")

    # 创建Fabric实例，支持多卡训练
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="ddp",
        precision="32-true",
    )
    
    fabric.launch()

    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    
    # 对于six_tasks，使用实际数据集中的图像形状(256x256)
    if cfg.task.name == "six_tasks":
        cfg.task.image_shape = [3, 256, 256]
        cfg.task.shape_meta.obs.head_cam.shape = [3, 256, 256]
    else:
        # 对于其他任务，使用相机配置中的原始形状
        cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
        cfg.task.shape_meta.obs.head_cam.shape = [
            3,
            head_camera_cfg["h"],
            head_camera_cfg["w"],
        ]
    
    OmegaConf.resolve(cfg)
    
    # 再次设置，确保配置生效
    if cfg.task.name == "six_tasks":
        cfg.task.image_shape = [3, 256, 256]
        cfg.task.shape_meta.obs.head_cam.shape = [3, 256, 256]
    else:
        cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
        cfg.task.shape_meta.obs.head_cam.shape = [
            3,
            head_camera_cfg["h"],
            head_camera_cfg["w"],
        ]

    # 获取当前进程的rank和world_size
    rank = fabric.global_rank
    world_size = fabric.world_size
    
    # 判断是否为多卡训练
    is_multi_gpu = world_size > 1
    
    if rank == 0:
        if is_multi_gpu:
            print(f"=== 多卡训练配置 ===")
        else:
            print(f"=== 单卡训练配置 ===")
        print(f"总进程数: {world_size}")
        print(f"当前进程rank: {rank}")
        print(f"使用设备: {fabric.device}")
        print(f"策略: {fabric.strategy}")
        
        # 兼容Lightning 2.5.3的精度属性访问
        try:
            precision = fabric.precision
        except AttributeError:
            try:
                precision = fabric._precision
            except AttributeError:
                try:
                    precision = fabric.precision_
                except AttributeError:
                    precision = "未知"
        
        print(f"精度: {precision}")
    
    # 创建workspace实例，传递fabric相关信息
    workspace = hydra.utils.instantiate(cfg.task.workspace, cfg)
    
    # 设置训练属性
    workspace.fabric = fabric
    workspace.rank = rank
    workspace.world_size = world_size
    
    if rank == 0:
        if is_multi_gpu:
            print(f"✅ 多卡训练配置完成")
        else:
            print(f"✅ 单卡训练配置完成")
        print(f"开始训练...")
    
    workspace.run()


if __name__ == "__main__":
    main()
