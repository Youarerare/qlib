"""检查训练结果"""
from pathlib import Path
import os

mlruns = Path('mlruns/0')
if mlruns.exists():
    runs = [d for d in mlruns.iterdir() if d.is_dir() and d.name != 'meta.yaml']
    print(f'MLflow runs: {len(runs)}')
    
    for run in sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True):
        print(f'\n{"="*60}')
        print(f'Run: {run.name}')
        print('='*60)
        
        # 检查params
        params_dir = run / 'params'
        if params_dir.exists():
            print('\n[参数]')
            for p in sorted(params_dir.iterdir()):
                with open(p, 'r') as f:
                    print(f'  {p.name}: {f.read().strip()}')
        
        # 检查metrics
        metrics_dir = run / 'metrics'
        if metrics_dir.exists():
            print('\n[指标]')
            for m in sorted(metrics_dir.iterdir()):
                with open(m, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last = lines[-1].strip().split()
                        print(f'  {m.name}: {last[1] if len(last) > 1 else last}')
        
        # 检查artifacts
        artifacts_dir = run / 'artifacts'
        if artifacts_dir.exists():
            print('\n[输出文件]')
            for a in artifacts_dir.rglob('*'):
                if a.is_file():
                    print(f'  {a.relative_to(artifacts_dir)}')
else:
    print('mlruns目录不存在')

# 检查其他可能的结果目录
print('\n' + '='*60)
print('检查其他输出目录')
print('='*60)

# 检查mlruns总目录
mlruns_root = Path('mlruns')
if mlruns_root.exists():
    for exp in mlruns_root.iterdir():
        if exp.is_dir():
            runs_count = len([d for d in exp.iterdir() if d.is_dir()])
            print(f'  {exp.name}: {runs_count} runs')
