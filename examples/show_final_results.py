"""显示所有训练结果 - 完整版"""
from pathlib import Path
import os

mlruns = Path('mlruns/746607912206639406')
if not mlruns.exists():
    print('mlruns目录不存在')
    exit()

runs = [d for d in mlruns.iterdir() if d.is_dir() and d.name != 'meta.yaml']
runs = sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)

print(f'找到 {len(runs)} 个训练runs')
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

    def read_metric(name):
        f = metrics_dir / name
        if f.exists():
            with open(f, 'r') as fp:
                content = fp.read().strip()
                parts = content.split()
                if len(parts) >= 2:
                    return float(parts[1])
        return None

    ic = read_metric('IC')
    rank_ic = read_metric('Rank IC')
    icir = read_metric('ICIR')
    rank_icir = read_metric('Rank ICIR')
    annualized_return = read_metric('1day.excess_return_without_cost.annualized_return')
    information_ratio = read_metric('1day.excess_return_without_cost.information_ratio')
    max_drawdown = read_metric('1day.excess_return_without_cost.max_drawdown')

    # 获取运行时间
    import time
    mtime = run.stat().st_mtime
    mtime_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))

    results.append({
        'model': model_name,
        'ic': ic,
        'rank_ic': rank_ic,
        'icir': icir,
        'rank_icir': rank_icir,
        'annualized_return': annualized_return,
        'information_ratio': information_ratio,
        'max_drawdown': max_drawdown,
        'time': mtime_str
    })

# 打印详细结果
print(f"\n{'模型':<15} {'IC':<10} {'RankIC':<10} {'ICIR':<10} {'年化收益':<12} {'信息比率':<10} {'时间':<20}")
print('-'*90)

for r in results:
    ic_str = f"{r['ic']:.4f}" if r['ic'] else 'N/A'
    rank_ic_str = f"{r['rank_ic']:.4f}" if r['rank_ic'] else 'N/A'
    icir_str = f"{r['icir']:.4f}" if r['icir'] else 'N/A'
    ret_str = f"{r['annualized_return']:.4f}" if r['annualized_return'] else 'N/A'
    ir_str = f"{r['information_ratio']:.4f}" if r['information_ratio'] else 'N/A'
    print(f"{r['model']:<15} {ic_str:<10} {rank_ic_str:<10} {icir_str:<10} {ret_str:<12} {ir_str:<10} {r['time']:<20}")

# 按模型类型汇总
print('\n' + '='*80)
print('按模型类型汇总 (最新结果)')
print('='*80)

model_results = {}
for r in results:
    model = r['model']
    if model not in model_results or r['time'] > model_results[model]['time']:
        model_results[model] = r

print(f"\n{'模型':<15} {'IC':<12} {'Rank IC':<12} {'ICIR':<12} {'年化收益':<12} {'信息比率':<12}")
print('-'*75)
for model, r in sorted(model_results.items(), key=lambda x: x[1]['ic'] if x[1]['ic'] else 0, reverse=True):
    ic_str = f"{r['ic']:.4f}" if r['ic'] else 'N/A'
    rank_ic_str = f"{r['rank_ic']:.4f}" if r['rank_ic'] else 'N/A'
    icir_str = f"{r['icir']:.4f}" if r['icir'] else 'N/A'
    ret_str = f"{r['annualized_return']:.4f}" if r['annualized_return'] else 'N/A'
    ir_str = f"{r['information_ratio']:.4f}" if r['information_ratio'] else 'N/A'
    print(f"{model:<15} {ic_str:<12} {rank_ic_str:<12} {icir_str:<12} {ret_str:<12} {ir_str:<12}")
