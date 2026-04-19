"""重新训练LSTM和TabNet (已优化显存使用)"""
import subprocess
import sys
from pathlib import Path
import time

print("="*80)
print("重新训练 LSTM 和 TabNet")
print("="*80)
print()
print("优化措施:")
print("  - LSTM batch_size: 2000 -> 512 (避免显存溢出)")
print("  - LSTM n_jobs: 20 -> 10")
print()

models = [
    ("LSTM (优化版)", "benchmarks/LSTM/workflow_config_lstm_Alpha158_fast.yaml"),
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

print()
print("="*80)
print("训练总结")
print("="*80)
print(f"\n{'模型':<20} {'状态':<10} {'用时(分钟)':<15}")
print("-"*45)
for r in results:
    print(f"{r['name']:<20} {r['status']:<10} {r['time']/60:.1f}")

print("\n运行 python show_final_results.py 查看结果")
