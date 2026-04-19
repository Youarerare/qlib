"""
监控GPU使用情况
Monitor GPU usage
"""
import subprocess
import time
import os

def check_gpu():
    """检查GPU使用情况"""
    try:
        # 使用nvidia-smi检查GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("\n" + "="*80)
            print("GPU使用情况")
            print("="*80)

            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    print(f"\nGPU {parts[0]}: {parts[1]}")
                    print(f"  GPU利用率: {parts[2]}%")
                    print(f"  显存使用: {parts[3]} MB / {parts[4]} MB")

            return True
    except:
        pass

    return False

def check_pytorch_gpu():
    """检查PyTorch GPU状态"""
    try:
        import torch
        print("\n" + "="*80)
        print("PyTorch GPU状态")
        print("="*80)
        print(f"\nCUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.current_device()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")

            # 显存信息
            print(f"\n显存分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"显存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    except Exception as e:
        print(f"无法获取PyTorch GPU信息: {e}")

def main():
    print("="*80)
    print("GPU监控工具")
    print("="*80)

    # 检查PyTorch
    check_pytorch_gpu()

    # 检查nvidia-smi
    if not check_gpu():
        print("\n⚠️ nvidia-smi不可用，可能需要安装NVIDIA驱动")

    print("\n" + "="*80)
    print("说明:")
    print("  - 如果GPU利用率>0%，说明模型正在使用GPU")
    print("  - RTX 3060有6GB显存，足够运行LSTM和TabNet")
    print("  - 使用GPU后，训练速度会提升5-10倍")
    print("="*80)

if __name__ == '__main__':
    main()
