"""
清理并重新启动GPU加速训练
Clean up and restart training with GPU
"""
import subprocess
import sys
import os
from pathlib import Path

def kill_all_python():
    """杀掉所有Python进程"""
    print("正在停止所有Python进程...")
    os.system('taskkill /F /IM python.exe')
    print("✓ 所有Python进程已停止")

def run_model(name, config):
    """运行单个模型"""
    print(f"\n{'='*80}")
    print(f"启动: {name}")
    print(f"{'='*80}")

    cmd = [sys.executable, '-m', 'qlib.cli.run', config]
    process = subprocess.Popen(cmd, cwd=Path(__file__).parent)

    print(f"进程已启动，PID: {process.pid}")
    return process

def main():
    print("="*80)
    print("重新启动GPU加速训练")
    print("="*80)
    print("\n您的GPU: NVIDIA RTX 3060")
    print("CUDA已启用: ✓")

    choice = input("\n是否要停止所有正在运行的Python进程并重新开始？(y/n): ")

    if choice.lower() == 'y':
        # 停止所有进程
        kill_all_python()
        import time
        time.sleep(2)

        print("\n" + "="*80)
        print("重新启动训练（GPU加速）")
        print("="*80)

        models = [
            ('XGBoost', 'benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml'),
            ('LSTM', 'benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml'),
            ('TabNet', 'benchmarks/TabNet/workflow_config_tabnet_Alpha158.yaml')
        ]

        processes = []
        for name, config in models:
            proc = run_model(name, config)
            processes.append((name, proc))
            import time
            time.sleep(2)  # 稍微延迟一下

        print("\n" + "="*80)
        print("所有模型已启动！")
        print("="*80)
        print("\n预计训练时间:")
        print("  - XGBoost: ~5-10分钟")
        print("  - LSTM (GPU): ~15-20分钟 ⚡")
        print("  - TabNet (GPU): ~5-10分钟 ⚡")

        print("\n监控命令:")
        print("  python show_current_status.py  # 查看状态")
        print("  python monitor_gpu.py          # 查看GPU")
    else:
        print("\n取消操作")

if __name__ == '__main__':
    main()
