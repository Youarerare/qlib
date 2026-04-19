"""
简单快速的状态查看
Simple and quick status check
"""
import subprocess
from pathlib import Path
from datetime import datetime

print("\n" + "="*60)
print(f"训练状态 - {datetime.now().strftime('%H:%M:%S')}")
print("="*60)

# 1. Python进程数
result = subprocess.run(['tasklist'], capture_output=True, text=True)
python_count = result.stdout.count('python.exe')
print(f"\n进程数: {python_count}")

# 2. GPU状态
try:
    import torch
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated(0) / 1024**2
        print(f"GPU显存: {mem:.0f} MB")

        # 检查nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                print(f"GPU利用率: {parts[0]}%")
                print(f"GPU显存: {parts[1]} MB")
        except:
            pass

        if mem > 100:
            print("状态: ✅ GPU正在训练")
        else:
            print("状态: ⏳ 初始化中...")
except:
    print("GPU: 无法检查")

# 3. 模型状态
print("\n最近活动:")
mlruns = Path(__file__).parent / 'mlruns'
for exp in mlruns.iterdir():
    if exp.is_dir() and exp.name not in ['.trash', '0']:
        for run in exp.iterdir():
            if run.is_dir() and len(run.name) == 32:
                model_file = run / 'params' / 'model.class'
                if model_file.exists():
                    model = model_file.read_text().strip()
                    mod_time = datetime.fromtimestamp(run.stat().st_mtime)
                    elapsed = (datetime.now() - mod_time).total_seconds()

                    if elapsed < 60:
                        time_str = f"{int(elapsed)}秒前"
                    elif elapsed < 3600:
                        time_str = f"{int(elapsed/60)}分钟前"
                    else:
                        time_str = f"{int(elapsed/3600)}小时前"

                    print(f"  {model}: {time_str}")

print("="*60)
print("\n提示: 每5分钟运行一次查看进度")
print("命令: python simple_status.py")
