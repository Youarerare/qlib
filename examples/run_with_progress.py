"""
重新运行模型并实时显示训练进度
Run models with real-time progress display
"""
import subprocess
import sys
import time
from pathlib import Path

def run_with_output(name, config):
    """运行模型并显示输出"""
    print(f"\n{'='*80}")
    print(f"开始训练: {name}")
    print(f"{'='*80}\n")

    cmd = [sys.executable, '-m', 'qlib.cli.run', config]

    # 实时显示输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent,
        bufsize=1
    )

    # 实时打印每一行
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end='')

            # 如果看到训练相关的关键词，特别标注
            if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'ic', 'train', 'valid']):
                print(f"  ⭐ {line.strip()}")

    process.wait()

    if process.returncode == 0:
        print(f"\n✅ {name} 训练完成!")
    else:
        print(f"\n❌ {name} 训练失败!")

    return process.returncode

if __name__ == '__main__':
    print("="*80)
    print("带进度显示的训练")
    print("="*80)
    print("\n将实时显示训练进度（包括epoch、loss等信息）")
    print("\n选择要运行的模型:")
    print("  1. XGBoost (快速，约5-10分钟)")
    print("  2. LSTM (GPU加速，约15-20分钟)")
    print("  3. TabNet (GPU加速，约5-10分钟)")
    print("  4. 所有模型")

    choice = input("\n请输入选项 (1-4): ").strip()

    models = {
        '1': ('XGBoost', 'benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml'),
        '2': ('LSTM', 'benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml'),
        '3': ('TabNet', 'benchmarks/TabNet/workflow_config_tabnet_Alpha158.yaml'),
    }

    if choice == '4':
        for name, config in models.values():
            run_with_output(name, config)
    elif choice in models:
        name, config = models[choice]
        run_with_output(name, config)
    else:
        print("无效选项")
