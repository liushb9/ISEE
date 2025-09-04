#!/usr/bin/env python3
"""
内存监控脚本
在训练过程中监控内存使用情况
"""

import os
import time
import psutil
import GPUtil
import threading
from datetime import datetime

class MemoryMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.thread = None
        
    def get_memory_info(self):
        """获取内存信息"""
        # 系统内存
        memory = psutil.virtual_memory()
        system_info = {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'used': memory.used / (1024**3),  # GB
            'percent': memory.percent
        }
        
        # GPU内存
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,  # MB
                    'memory_used': gpu.memoryUsed,    # MB
                    'memory_free': gpu.memoryFree,    # MB
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                })
        except Exception as e:
            gpu_info = [{'error': str(e)}]
            
        return system_info, gpu_info
    
    def monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                system_info, gpu_info = self.get_memory_info()
                
                # 打印时间戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n=== {timestamp} ===")
                
                # 系统内存
                print(f"系统内存:")
                print(f"  总内存: {system_info['total']:.1f} GB")
                print(f"  已使用: {system_info['used']:.1f} GB ({system_info['percent']:.1f}%)")
                print(f"  可用: {system_info['available']:.1f} GB")
                
                # GPU内存
                print(f"GPU内存:")
                for gpu in gpu_info:
                    if 'error' in gpu:
                        print(f"  GPU {gpu['id']}: {gpu['error']}")
                    else:
                        print(f"  GPU {gpu['id']} ({gpu['name']}):")
                        print(f"    总内存: {gpu['memory_total']} MB")
                        print(f"    已使用: {gpu['memory_used']} MB ({gpu['memory_percent']:.1f}%)")
                        print(f"    可用: {gpu['memory_free']} MB")
                
                # 检查内存警告
                if system_info['percent'] > 90:
                    print(f"⚠️  警告: 系统内存使用率过高 ({system_info['percent']:.1f}%)")
                
                for gpu in gpu_info:
                    if 'memory_percent' in gpu and gpu['memory_percent'] > 90:
                        print(f"⚠️  警告: GPU {gpu['id']} 内存使用率过高 ({gpu['memory_percent']:.1f}%)")
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.thread.start()
            print(f"内存监控已启动，间隔: {self.interval}秒")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        print("内存监控已停止")

def main():
    """主函数"""
    print("=== 内存监控脚本 ===")
    print("按 Ctrl+C 停止监控")
    
    monitor = MemoryMonitor(interval=10)  # 10秒间隔
    
    try:
        monitor.start()
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n收到停止信号")
        monitor.stop()
        print("监控已停止")

if __name__ == "__main__":
    main()
