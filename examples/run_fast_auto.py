"""
自动运行快速版本对比实验
Fast comparison - auto run version
"""

import subprocess
import sys
from pathlib import Path
import time

print("="*80)
print("快速训练版本")
print("="*80)
print()
print("优化措施:")
print("  1. 数据量减少: 7年 -> 4年训练数据")
print("  2. Batch size增大: 800 -> 2000/4096")
print("  3. Epoch减少: 200 -> 50")
print("  4. Early stop: 10 -> 5")
print()
print("预计时间:")
print("  - XGBoost: 2-3分钟")
print("  - LSTM: 5-8分钟")
print("  - TabNet: 3-5分钟")
print("  - 总计: ~15分钟")
print()
print("="*80)

models = [
    ("XGBoost", "benchmarks/XGBoost/workflow_config_xgboost_Alpha158_fast.yaml"),
    ("LSTM", "benchmarks/LSTM/workflow_config_lstm_Alpha158_fast.yaml"),
    ("TabNet", "benchmarks/TabNet/workflow_config_tabnet_Alpha158_fast.yaml")
]

results = []
for name, config in models:
    print()
    print("="*80)
    print(f"训练: {name}")
    print("="*80)

    start_time = time.time()

    cmd = [sys.executable, "-m", "qlib.cli.run", config]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path(__file__).parent
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()
    elapsed = time.time() - start_time

    results.append({
        "name": name,
        "time": elapsed,
        "status": "完成" if process.returncode == 0 else "失败"
    })

    if process.returncode == 0:
        print(f"\n[OK] {name} 完成! 用时: {elapsed/60:.1f}分钟")
    else:
        print(f"\n[FAIL] {name} 失败!")

# 总结
print()
print("="*80)
print("训练总结")
print("="*80)
print()
print(f"{'模型':<15} {'状态':<10} {'用时(分钟)':<15}")
print("-" * 40)
for r in results:
    print(f"{r['name']:<15} {r['status']:<10} {r['time']/60:.1f}")
print()
print("[OK] 所有模型训练完成!")
print("查看结果: mlflow ui")
