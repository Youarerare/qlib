"""
完整的训练诊断报告
Complete training diagnostic report
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime

print("="*80)
print(f"完整训练诊断报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# 1. 系统资源
print("\n【系统资源】")
result = subprocess.run(['wmic', 'OS', 'get', 'FreePhysicalMemory,TotalVisibleMemorySize', '/Value'],
                       capture_output=True, text=True)
lines = [l for l in result.stdout.split('\n') if '=' in l]
for line in lines:
    if line.strip():
        print(f"  {line.strip()}")

# 2. Python进程
print("\n【Python进程】")
result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                       capture_output=True, text=True)
lines = result.stdout.split('\n')
python_lines = [l for l in lines if 'python.exe' in l]
print(f"  进程数: {len(python_lines)}")
if python_lines:
    print("  详情:")
    for line in python_lines[:5]:  # 只显示前5个
        parts = line.split()
        if len(parts) >= 5:
            pid = parts[1]
            mem = parts[4]
            print(f"    PID {pid}: {mem} 内存")

# 3. GPU状态
print("\n【GPU状态】")
try:
    import torch
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"  显存已分配: {mem_alloc:.1f} MB")
        print(f"  显存已预留: {mem_reserved:.1f} MB")

        if mem_alloc > 10:
            print("  ✓ GPU正在被使用！")
        else:
            print("  ⚠ GPU未被使用")
except Exception as e:
    print(f"  错误: {e}")

# 4. 实验目录
print("\n【实验记录】")
mlruns = Path(__file__).parent / 'mlruns'
if mlruns.exists():
    exp_count = 0
    run_count = 0
    models = []

    for exp in mlruns.iterdir():
        if exp.is_dir() and exp.name not in ['.trash', '0']:
            exp_count += 1
            for run in exp.iterdir():
                if run.is_dir() and len(run.name) == 32:
                    run_count += 1
                    model_file = run / 'params' / 'model.class'
                    if model_file.exists():
                        model = model_file.read_text().strip()
                        mod_time = datetime.fromtimestamp(run.stat().st_mtime)
                        models.append({
                            'model': model,
                            'time': mod_time
                        })

    print(f"  实验数: {exp_count}")
    print(f"  运行数: {run_count}")

    if models:
        print("\n  最近运行的模型:")
        for m in sorted(models, key=lambda x: x['time'], reverse=True)[:3]:
            elapsed = (datetime.now() - m['time']).total_seconds()
            if elapsed < 60:
                time_str = f"{int(elapsed)}秒前"
            elif elapsed < 3600:
                time_str = f"{int(elapsed/60)}分钟前"
            else:
                time_str = f"{int(elapsed/3600)}小时前"
            print(f"    {m['model']}: {time_str}")

# 5. 建议
print("\n" + "="*80)
print("诊断结果:")
print("="*80)

try:
    torch.cuda.memory_allocated(0)
    gpu_used = torch.cuda.memory_allocated(0) > 10*1024*1024
except:
    gpu_used = False

if gpu_used:
    print("✓ GPU正在工作，训练进行中")
    print("  预计完成时间: 15-30分钟")
elif len(python_lines) > 0:
    print("⚠ Python进程在运行但GPU未被使用")
    print("  可能原因:")
    print("  1. XGBoost等树模型不使用GPU")
    print("  2. 深度学习模型还未开始训练")
    print("  3. 模型配置问题")
    print("\n  建议: 等待几分钟让模型开始训练")
else:
    print("✗ 没有Python进程运行")
    print("  建议: 重新启动训练")
    print("  命令: python run_single_model.py xgboost")

print("\n可用命令:")
print("  - 查看状态: python show_current_status.py")
print("  - 重启训练: python restart_training_gpu.py")
print("  - 查看GPU: python monitor_gpu.py")
print("="*80)
