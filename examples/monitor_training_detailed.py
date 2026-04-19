"""
实时显示训练日志和进度
Real-time training log viewer
"""
import os
import time
from pathlib import Path
from datetime import datetime
import subprocess

def get_running_processes():
    """获取运行中的Python进程"""
    result = subprocess.run(
        ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
        capture_output=True,
        text=True
    )

    processes = []
    for line in result.stdout.strip().split('\n'):
        if 'python.exe' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                pid = parts[1].strip('"')
                if pid.isdigit():
                    processes.append(pid)
    return processes

def check_mlruns_activity():
    """检查mlruns目录的活动"""
    mlruns_dir = Path(__file__).parent / 'mlruns'

    runs = []
    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.trash', '0']:
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    # 获取最后修改时间
                    last_modified = datetime.fromtimestamp(run_dir.stat().st_mtime)

                    # 读取模型类型
                    model_file = run_dir / 'params' / 'model.class'
                    model_class = model_file.read_text().strip() if model_file.exists() else 'Unknown'

                    # 计算运行时间
                    start_time_file = run_dir / 'tags' / 'mlflow.startTime'
                    if start_time_file.exists():
                        start_time_str = start_time_file.read_text().strip()
                        try:
                            # 尝试解析时间
                            elapsed = (datetime.now() - last_modified).total_seconds()
                        except:
                            elapsed = 0
                    else:
                        elapsed = 0

                    runs.append({
                        'model': model_class,
                        'run_id': run_dir.name[:8],
                        'last_modified': last_modified,
                        'elapsed': elapsed,
                        'status': 'active' if elapsed < 60 else 'stale'
                    })

    return runs

def monitor_gpu():
    """监控GPU使用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_util = torch.cuda.memory_allocated(0) / 1024**2
            return gpu_util
    except:
        pass
    return 0

def main():
    """主监控循环"""
    print("="*80)
    print("实时训练监控")
    print("="*80)
    print("\n按 Ctrl+C 退出\n")

    try:
        iteration = 0
        while True:
            # 清屏
            os.system('cls' if os.name == 'nt' else 'clear')

            print("="*80)
            print(f"训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (刷新间隔: 5秒)")
            print("="*80)

            # 检查Python进程
            processes = get_running_processes()
            print(f"\n💻 Python进程数: {len(processes)}")

            # 检查GPU
            gpu_mb = monitor_gpu()
            print(f"🎮 GPU显存使用: {gpu_mb:.2f} MB")

            # 检查实验运行
            print("\n📊 实验状态:")
            print("-" * 80)

            runs = check_mlruns_activity()
            active_count = sum(1 for r in runs if r['status'] == 'active')

            for run in sorted(runs, key=lambda x: x['last_modified'], reverse=True):
                status_icon = '🟢' if run['status'] == 'active' else '⚪'
                time_ago = (datetime.now() - run['last_modified']).total_seconds()

                if time_ago < 60:
                    time_str = f"{int(time_ago)}秒前"
                elif time_ago < 3600:
                    time_str = f"{int(time_ago/60)}分钟前"
                else:
                    time_str = f"{int(time_ago/3600)}小时前"

                print(f"{status_icon} {run['model']:<15} | 运行ID: {run['run_id']} | 最后活动: {time_str}")

            print("-" * 80)
            print(f"\n活跃实验: {active_count}/{len(runs)}")

            # 预估进度
            if active_count > 0:
                print("\n⏳ 训练进行中...")
                print("   - 如果GPU显存>100MB，说明深度学习模型正在训练")
                print("   - 每个epoch大约需要30秒-1分钟（GPU加速）")
                print("   - LSTM需要200个epochs，TabNet需要100个epochs")
            else:
                print("\n⚠️  没有活跃的训练进程")

            # 提示
            print("\n" + "="*80)
            print("💡 提示:")
            print("  - 查看详细日志: cd mlruns/<exp_id>/<run_id>/artifacts/")
            print("  - 查看GPU: python monitor_gpu.py")
            print("  - 查看结果: python check_model_status.py")
            print("="*80)

            iteration += 1
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n✓ 监控已停止")

if __name__ == '__main__':
    main()
