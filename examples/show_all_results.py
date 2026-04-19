"""显示所有训练结果"""
from pathlib import Path
import os

mlruns = Path('mlruns/746607912206639406')
if mlruns.exists():
    runs = [d for d in mlruns.iterdir() if d.is_dir() and d.name != 'meta.yaml']
    runs = sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)

    print(f'找到 {len(runs)} 个训练runs\n')
    print('='*80)

    results = []
    for run in runs:
        # 获取模型信息
        model_class_file = run / 'params' / 'model.class'
        model_name = 'Unknown'
        if model_class_file.exists():
            with open(model_class_file, 'r') as f:
                model_name = f.read().strip()

        # 获取metrics
        metrics_dir = run / 'metrics'
        metrics = {}
        if metrics_dir.exists():
            for m in metrics_dir.iterdir():
                with open(m, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last = lines[-1].strip().split()
                        if len(last) >= 2:
                            metrics[m.name] = float(last[1])

        # 获取运行名称
        run_name_file = run / 'tags' / 'mlflow.runName'
        run_name = ''
        if run_name_file.exists():
            with open(run_name_file, 'r') as f:
                run_name = f.read().strip()

        results.append({
            'name': run_name or run.name[:8],
            'model': model_name,
            'metrics': metrics
        })

        print(f"\n[{run_name or run.name[:8]}] - {model_name}")
        print('-'*40)

        # 显示关键指标
        key_metrics = ['ic', 'icir', 'rank_ic', 'rank_icir', 'train_loss', 'val_loss']
        for km in key_metrics:
            if km in metrics:
                print(f"  {km}: {metrics[km]:.6f}")

        # 显示其他指标
        other_metrics = {k: v for k, v in metrics.items() if k not in key_metrics}
        if other_metrics:
            print("  其他指标:")
            for k, v in sorted(other_metrics.items()):
                print(f"    {k}: {v:.6f}")

    # 总结比较
    print('\n' + '='*80)
    print('模型比较总结')
    print('='*80)
    print(f"\n{'模型':<15} {'IC':<12} {'Rank IC':<12} {'ICIR':<12}")
    print('-'*50)
    for r in results:
        ic = r['metrics'].get('ic', r['metrics'].get('test_ic', 0))
        rank_ic = r['metrics'].get('rank_ic', r['metrics'].get('test_rank_ic', 0))
        icir = r['metrics'].get('icir', r['metrics'].get('test_icir', 0))
        print(f"{r['model']:<15} {ic:<12.4f} {rank_ic:<12.4f} {icir:<12.4f}")

else:
    print('mlruns目录不存在')

# 也检查mlruns/0目录
print('\n' + '='*80)
print('检查 mlruns/0 目录')
print('='*80)
mlruns0 = Path('mlruns/0')
if mlruns0.exists():
    runs0 = [d for d in mlruns0.iterdir() if d.is_dir() and d.name != 'meta.yaml']
    print(f'mlruns/0 中的runs数: {len(runs0)}')
