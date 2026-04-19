"""
运行TabNet、XGBoost和LSTM三个模型的对比实验
Run comparison experiments for TabNet, XGBoost, and LSTM models
"""

import os
import sys
import subprocess
from pathlib import Path
import time

# 模型配置
MODELS = {
    'XGBoost': {
        'config': 'benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml',
        'requirements': ['xgboost']
    },
    'LSTM': {
        'config': 'benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml',
        'requirements': ['torch']
    },
    'TabNet': {
        'config': 'benchmarks/TabNet/workflow_config_tabnet_Alpha158.yaml',
        'requirements': ['torch']
    }
}

def install_dependencies(requirements):
    """安装依赖"""
    for req in requirements:
        print(f"检查依赖: {req}")
        try:
            __import__(req)
            print(f"  ✓ {req} 已安装")
        except ImportError:
            print(f"  安装 {req}...")
            subprocess.run(['pip', 'install', req], check=True)

def run_model(model_name, config_path):
    """运行单个模型"""
    print("\n" + "="*80)
    print(f"开始运行模型: {model_name}")
    print("="*80)
    
    start_time = time.time()
    
    # 使用qrun命令运行配置文件
    cmd = ['python', '-m', 'qlib.cli.run', config_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {model_name} 运行成功!")
            print(f"  耗时: {elapsed_time:.2f} 秒")
            print(f"\n输出摘要:")
            # 提取关键信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'IC' in line or 'Rank IC' in line or 'excess_return' in line or 'information_ratio' in line:
                    print(f"  {line}")
        else:
            print(f"\n✗ {model_name} 运行失败!")
            print(f"  错误信息: {result.stderr}")
            
    except Exception as e:
        print(f"\n✗ {model_name} 运行出错: {str(e)}")
        return False
    
    return True

def check_data():
    """检查数据是否存在"""
    data_path = Path.home() / '.qlib' / 'qlib_data' / 'cn_data'
    if not data_path.exists():
        print("错误: 数据不存在，请先下载数据")
        print("运行命令: python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
        return False
    
    print(f"✓ 数据目录存在: {data_path}")
    return True

def main():
    """主函数"""
    print("="*80)
    print("Qlib 模型对比实验")
    print("Model Comparison: TabNet vs XGBoost vs LSTM")
    print("="*80)
    
    # 检查数据
    if not check_data():
        return
    
    # 切换到examples目录
    examples_dir = Path(__file__).parent
    os.chdir(examples_dir)
    print(f"\n工作目录: {os.getcwd()}")
    
    # 安装依赖
    print("\n" + "="*80)
    print("检查并安装依赖")
    print("="*80)
    
    all_requirements = set()
    for model_info in MODELS.values():
        all_requirements.update(model_info['requirements'])
    
    install_dependencies(list(all_requirements))
    
    # 运行模型
    results = {}
    for model_name, model_info in MODELS.items():
        config_path = model_info['config']
        if Path(config_path).exists():
            success = run_model(model_name, config_path)
            results[model_name] = success
        else:
            print(f"\n✗ 配置文件不存在: {config_path}")
            results[model_name] = False
    
    # 打印总结
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    for model_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {model_name}: {status}")
    
    print("\n提示:")
    print("  - 模型结果会保存在 mlruns/ 目录下")
    print("  - 可以使用 tensorboard --logdir mlruns 查看详细训练过程")
    print("  - 对于LSTM和TabNet，确保有GPU可用（或修改配置使用CPU）")

if __name__ == '__main__':
    main()
