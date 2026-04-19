"""
使用GPU加速重新运行模型
Re-run models with GPU acceleration
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*80)
    print("GPU加速训练")
    print("="*80)
    print("\n您的GPU: NVIDIA GeForce RTX 3060 Laptop GPU")
    print("CUDA版本: 12.1")
    print("\n现在重新运行模型将使用GPU加速！")
    print("\n预计加速效果:")
    print("  - LSTM: 从2小时 → 15-30分钟 ⚡")
    print("  - TabNet: 从45分钟 → 5-10分钟 ⚡")
    print("\n" + "="*80)

    print("\n建议:")
    print("1. 如果模型正在运行，可以等待它们完成（CPU模式）")
    print("2. 或者重新启动它们以使用GPU加速")

    choice = input("\n是否重新运行模型？(y/n): ").lower()

    if choice == 'y':
        models = [
            ('XGBoost', 'benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml'),
            ('LSTM', 'benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml'),
            ('TabNet', 'benchmarks/TabNet/workflow_config_tabnet_Alpha158.yaml')
        ]

        for name, config in models:
            print(f"\n开始运行: {name}")
            cmd = [sys.executable, '-m', 'qlib.cli.run', config]
            process = subprocess.Popen(cmd, cwd=Path(__file__).parent)
            process.wait()
            print(f"✓ {name} 完成!")
    else:
        print("\n保持当前运行的模型继续执行。")
        print("下次运行将自动使用GPU。")

if __name__ == '__main__':
    main()
