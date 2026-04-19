"""
快速版本对比实验 - 减少数据量，加速训练
Fast comparison with reduced data
"""

import subprocess
import sys
from pathlib import Path
import time

print("="*80)
print("快速训练版本")
print("="*80)
print("\n优化措施:")
print("  1. 数据量减少: 7年 → 4年训练数据")
print("  2. Batch size增大: 800 → 2000/4096")
print("  3. Epoch减少: 200 → 50")
print("  4. Early stop: 10 → 5")
print("\n预计时间:")
print("  - XGBoost: 2-3分钟")
print("  - LSTM: 5-8分钟")
print("  - TabNet: 3-5分钟")
print("  - 总计: ~15分钟")

print("\n" + "="*80)

choice = input("\n是否停止当前训练并运行快速版本？(y/n): ")

if choice.lower() == 'y':
    # 停止当前进程
    print("\n停止所有Python进程...")
    import os
    os.system('taskkill /F /IM python.exe')
    time.sleep(2)

    print("\n" + "="*80)
    print("开始快速训练")
    print("="*80)

    models = [
        ('XGBoost (快速版)', 'benchmarks/XGBoost/workflow_config_xgboost_Alpha158_fast.yaml'),
        ('LSTM (快速版)', 'benchmarks/LSTM/workflow_config_lstm_Alpha158_fast.yaml'),
        ('TabNet (快速版)', 'benchmarks/TabNet/workflow_config_tabnet_Alpha158_fast.yaml')
    ]

    results = []
    for name, config in models:
        print(f"\n{'='*80}")
        print(f"训练: {name}")
        print(f"{'='*80}")

        start_time = time.time()

        cmd = [sys.executable, '-m', 'qlib.cli.run', config]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(__file__).parent
        )

        # 显示输出
        for line in process.stdout:
            print(line, end='')

        process.wait()
        elapsed = time.time() - start_time

        results.append({
            'name': name,
            'time': elapsed,
            'status': '完成' if process.returncode == 0 else '失败'
        })

        if process.returncode == 0:
            print(f"\n✓ {name} 完成! 用时: {elapsed/60:.1f}分钟")
        else:
            print(f"\n✗ {name} 失败!")

    # 总结
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)

    print(f"\n{'模型':<20} {'状态':<10} {'用时(分钟)':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['name']:<20} {r['status']:<10} {r['time']/60:<15.1f}")

    print("\n✓ 所有模型训练完成!")
    print("查看结果: mlflow ui")

else:
    print("\n取消操作")
