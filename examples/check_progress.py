"""
实时监控模型训练进度
Real-time monitoring of model training progress
"""
import os
import time
from pathlib import Path
from datetime import datetime

def check_progress():
    """检查训练进度"""
    mlruns_dir = Path(__file__).parent / 'mlruns'

    models = {
        'XGBoost': 'XGBModel',
        'LSTM': 'LSTM',
        'TabNet': 'TabnetModel'
    }

    print("="*80)
    print(f"Qlib 模型训练进度监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    results = {}

    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.trash', '0']:
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    # 读取模型类型
                    model_file = run_dir / 'params' / 'model.class'
                    if model_file.exists():
                        model_class = model_file.read_text().strip()

                        # 匹配模型
                        for name, class_name in models.items():
                            if class_name in model_class:
                                # 检查文件
                                artifacts = list((run_dir / 'artifacts').glob('*')) if (run_dir / 'artifacts').exists() else []
                                metrics = list((run_dir / 'metrics').glob('*')) if (run_dir / 'metrics').exists() else []

                                results[name] = {
                                    'run_id': run_dir.name,
                                    'artifacts_count': len(artifacts),
                                    'metrics_count': len(metrics),
                                    'has_task': (run_dir / 'artifacts' / 'task').exists(),
                                    'status': '完成' if metrics else ('运行中' if artifacts else '初始化')
                                }

    # 打印结果
    print(f"\n{'模型':<15} {'状态':<10} {'文件数':<10} {'指标数':<10} {'运行ID'}")
    print("-" * 80)

    for name in models.keys():
        if name in results:
            r = results[name]
            print(f"{name:<15} {r['status']:<10} {r['artifacts_count']:<10} {r['metrics_count']:<10} {r['run_id'][:8]}")
        else:
            print(f"{name:<15} {'未开始':<10} {'-':<10} {'-':<10} {'-'}")

    print("\n" + "="*80)

    # 统计
    completed = sum(1 for r in results.values() if r['status'] == '完成')
    running = sum(1 for r in results.values() if r['status'] == '运行中')

    print(f"\n统计:")
    print(f"  已完成: {completed}/3")
    print(f"  运行中: {running}/3")
    print(f"  未开始: {3 - completed - running}/3")

    if completed == 3:
        print("\n✓ 所有模型训练完成!")
        print("  运行 'mlflow ui' 查看详细结果")
    elif running > 0:
        print("\n⏳ 有模型正在运行中，请稍候...")
        print("  可以继续运行此脚本查看进度")

    return results

if __name__ == '__main__':
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        results = check_progress()

        # 如果所有模型都完成了，退出
        completed = sum(1 for r in results.values() if r['status'] == '完成')
        if completed == 3:
            break

        print("\n" + "="*80)
        print("30秒后自动刷新... (按Ctrl+C退出)")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n用户中断")
            break
