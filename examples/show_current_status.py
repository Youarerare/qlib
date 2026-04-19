"""
简单的训练进度查看器
Simple training progress viewer
"""
import subprocess
import time
from datetime import datetime
from pathlib import Path

def check_status():
    """检查当前状态"""
    print("\n" + "="*80)
    print(f"训练状态检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 1. Python进程
    result = subprocess.run(['tasklist'], capture_output=True, text=True)
    python_count = result.stdout.count('python.exe')
    print(f"\n1. Python进程数: {python_count}")

    # 2. GPU使用
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1024**2
            print(f"2. GPU显存: {mem:.1f} MB (RTX 3060)")
            if mem > 100:
                print("   ✓ GPU正在被使用")
            else:
                print("   - GPU空闲")
    except:
        print("2. GPU: 无法检查")

    # 3. 检查mlruns
    mlruns = Path(__file__).parent / 'mlruns'
    exp_count = 0
    run_count = 0

    for exp in mlruns.iterdir():
        if exp.is_dir() and exp.name not in ['.trash', '0']:
            exp_count += 1
            for run in exp.iterdir():
                if run.is_dir():
                    run_count += 1

    print(f"\n3. 实验数量: {exp_count}")
    print(f"4. 运行数量: {run_count}")

    # 4. 最近活动
    print(f"\n5. 最近运行的模型:")
    models_found = []

    for exp in mlruns.iterdir():
        if exp.is_dir() and exp.name not in ['.trash', '0']:
            for run in exp.iterdir():
                if run.is_dir():
                    model_file = run / 'params' / 'model.class'
                    if model_file.exists():
                        model = model_file.read_text().strip()
                        mod_time = datetime.fromtimestamp(run.stat().st_mtime)
                        elapsed = (datetime.now() - mod_time).total_seconds()

                        if elapsed < 3600:  # 1小时内
                            models_found.append({
                                'model': model,
                                'elapsed': elapsed,
                                'time_str': f"{int(elapsed/60)}分钟前" if elapsed >= 60 else f"{int(elapsed)}秒前"
                            })

    if models_found:
        for m in sorted(models_found, key=lambda x: x['elapsed']):
            print(f"   - {m['model']}: 最后活动 {m['time_str']}")
    else:
        print("   没有最近的活动")

    # 5. 判断状态
    print("\n" + "="*80)
    if python_count > 2 and run_count > 0:
        print("状态: 🟢 训练正在进行中")
        print("\n预估:")
        print("  - GPU加速已启用")
        print("  - 深度学习模型每个epoch约30秒-1分钟")
        print("  - 总训练时间约15-30分钟")
    else:
        print("状态: ⚪ 训练可能已完成或未开始")
    print("="*80)

    return python_count, run_count

if __name__ == '__main__':
    python_count, run_count = check_status()

    print("\n命令:")
    print("  - 再次检查: python show_current_status.py")
    print("  - 详细监控: python monitor_training_detailed.py")
    print("  - 查看结果: python check_model_status.py")
