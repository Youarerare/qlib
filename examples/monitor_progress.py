"""
实时监控训练进度
"""
import os
import time
from pathlib import Path

print("=" * 60)
print("训练进度监控")
print("=" * 60)

# 检查GPU
print("\n[GPU状态]")
os.system('"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader')

# 检查Python进程
print("\n[进程状态]")
os.system('tasklist | findstr python | find /c "python" > temp_count.txt')
with open('temp_count.txt', 'r') as f:
    count = f.read().strip()
print(f"运行中的Python进程数: {count}")
os.remove('temp_count.txt')

# 检查MLflow runs
print("\n[训练进度]")
mlruns_path = Path('mlruns/0')
if mlruns_path.exists():
    runs = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name != 'meta.yaml']
    print(f"已完成的训练runs: {len(runs)}")

    # 检查最新的run
    if runs:
        latest = sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(f"最新run: {latest.name}")

        # 检查metrics
        metrics_path = latest / 'metrics'
        if metrics_path.exists():
            for metric_file in metrics_path.iterdir():
                with open(metric_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        print(f"  {metric_file.name}: {last_line}")

print("\n" + "=" * 60)
print("提示: 每30秒自动刷新一次，按Ctrl+C退出")
print("=" * 60)

try:
    while True:
        time.sleep(30)
        print("\n" + "=" * 60)
        print(f"[刷新 - {time.strftime('%H:%M:%S')}]")
        print("=" * 60)

        print("\n[GPU状态]")
        os.system('"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader')

        print("\n[进程状态]")
        os.system('tasklist | findstr python | find /c "python" > temp_count.txt')
        with open('temp_count.txt', 'r') as f:
            count = f.read().strip()
        print(f"运行中的Python进程数: {count}")
        os.remove('temp_count.txt')

        print("\n[训练进度]")
        if mlruns_path.exists():
            runs = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name != 'meta.yaml']
            print(f"已完成的训练runs: {len(runs)}")

            if runs:
                latest = sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]

                # 检查是否完成
                tags_path = latest / 'tags'
                if tags_path.exists():
                    status_file = tags_path / 'status'
                    if status_file.exists():
                        with open(status_file, 'r') as f:
                            print(f"  状态: {f.read().strip()}")

                metrics_path = latest / 'metrics'
                if metrics_path.exists():
                    for metric_file in metrics_path.iterdir():
                        with open(metric_file, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()
                                print(f"  {metric_file.name}: {last_line}")
except KeyboardInterrupt:
    print("\n\n监控已停止")
