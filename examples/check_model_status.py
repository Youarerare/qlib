"""
查看模型训练结果
Check model training results
"""
import os
from pathlib import Path

def check_model_results():
    """检查模型结果"""
    mlruns_dir = Path(__file__).parent / 'mlruns'
    
    print("="*80)
    print("Qlib 模型训练状态检查")
    print("="*80)
    
    # 查找所有实验
    exps = []
    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.trash', '0']:
            # 查找run
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    exps.append(run_dir)
    
    if not exps:
        print("\n没有找到实验结果")
        return
    
    print(f"\n找到 {len(exps)} 个实验运行记录\n")
    
    for run_dir in exps:
        print(f"\n实验ID: {run_dir.parent.name}")
        print(f"运行ID: {run_dir.name}")
        print("-" * 80)
        
        # 读取参数 - 模型类型
        model_class_file = run_dir / 'params' / 'model.class'
        if model_class_file.exists():
            model_class = model_class_file.read_text().strip()
            print(f"模型: {model_class}")
        
        # 检查是否有输出文件
        artifacts_dir = run_dir / 'artifacts'
        if artifacts_dir.exists():
            print("\n生成的文件:")
            for artifact in artifacts_dir.rglob('*'):
                if artifact.is_file():
                    size = artifact.stat().st_size
                    print(f"  - {artifact.name} ({size} bytes)")
        
        # 检查是否有metrics
        metrics_dir = run_dir / 'metrics'
        if metrics_dir.exists():
            print("\n指标:")
            for metric_file in metrics_dir.iterdir():
                if metric_file.is_file():
                    value = metric_file.read_text().strip()
                    print(f"  {metric_file.name}: {value}")
        
        # 检查record文件
        record_file = run_dir / 'artifacts' / 'record'
        if record_file.exists():
            print(f"\n记录文件存在: {record_file}")
        
        print()

if __name__ == '__main__':
    check_model_results()
