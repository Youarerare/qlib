"""
快速运行单个模型并输出结果
Quick run single model and show results
"""
import subprocess
import sys
from pathlib import Path

def run_model(model_name, config_file):
    """运行模型"""
    print(f"\n{'='*80}")
    print(f"开始运行: {model_name}")
    print(f"{'='*80}\n")
    
    cmd = [sys.executable, '-m', 'qlib.cli.run', config_file]
    
    print(f"命令: {' '.join(cmd)}\n")
    
    # 实时输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode == 0:
        print(f"\n✓ {model_name} 完成!")
    else:
        print(f"\n✗ {model_name} 失败! 返回码: {process.returncode}")
    
    return process.returncode

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model = sys.argv[1].lower()
    else:
        print("用法: python run_single_model.py [xgboost|lstm|tabnet]")
        print("\n可用的模型:")
        print("  xgboost - XGBoost模型 (最快)")
        print("  lstm    - LSTM模型 (需要GPU)")
        print("  tabnet  - TabNet模型 (需要GPU)")
        sys.exit(1)
    
    models = {
        'xgboost': ('XGBoost', 'benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml'),
        'lstm': ('LSTM', 'benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml'),
        'tabnet': ('TabNet', 'benchmarks/TabNet/workflow_config_tabnet_Alpha158.yaml')
    }
    
    if model not in models:
        print(f"错误: 未知的模型 '{model}'")
        print(f"可用模型: {', '.join(models.keys())}")
        sys.exit(1)
    
    name, config = models[model]
    run_model(name, config)
