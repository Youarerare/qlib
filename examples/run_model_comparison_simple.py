"""
完整的模型对比运行脚本
Complete model comparison script
"""

import subprocess
import sys
import time
from pathlib import Path
import json

MODELS = [
    {
        'name': 'XGBoost',
        'config': 'benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml',
        'description': 'XGBoost梯度提升树模型 - 快速且效果好'
    },
    {
        'name': 'LSTM',
        'config': 'benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml',
        'description': 'LSTM长短期记忆网络 - 适合序列数据'
    },
    {
        'name': 'TabNet',
        'config': 'benchmarks/TabNet/workflow_config_tabnet_Alpha158.yaml',
        'description': 'TabNet注意力网络 - 可解释性强'
    }
]

def run_model(model_info):
    """运行单个模型"""
    print(f"\n{'='*80}")
    print(f"模型: {model_info['name']}")
    print(f"描述: {model_info['description']}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [sys.executable, '-m', 'qlib.cli.run', model_info['config']]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent
    )
    
    # 实时输出
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    elapsed = time.time() - start_time
    
    result = {
        'model': model_info['name'],
        'returncode': process.returncode,
        'elapsed_time': elapsed,
        'status': '成功' if process.returncode == 0 else '失败'
    }
    
    if process.returncode == 0:
        print(f"\n✓ {model_info['name']} 完成! 耗时: {elapsed:.1f}秒")
    else:
        print(f"\n✗ {model_info['name']} 失败! 返回码: {process.returncode}")
    
    return result

def main():
    """主函数"""
    print("="*80)
    print("Qlib 三模型对比实验")
    print("="*80)
    print("\n将依次运行以下模型:")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model['name']}: {model['description']}")
    
    input("\n按回车键开始运行...")
    
    results = []
    for model_info in MODELS:
        result = run_model(model_info)
        results.append(result)
    
    # 打印总结
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    print(f"\n{'模型':<15} {'状态':<10} {'耗时(秒)':<15}")
    print("-" * 40)
    for r in results:
        print(f"{r['model']:<15} {r['status']:<10} {r['elapsed_time']:<15.1f}")
    
    print("\n结果位置: mlruns/")
    print("查看方法: mlflow ui  (然后在浏览器打开 http://localhost:5000)")

if __name__ == '__main__':
    main()
