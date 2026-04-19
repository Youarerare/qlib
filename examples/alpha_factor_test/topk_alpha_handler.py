"""
Top50因子特征生成器
从回测结果中读取Top50因子，计算因子值并生成自定义DataHandler

使用方法:
    1. 先预计算因子: python topk_alpha_handler.py --prepare --topk-csv top50_by_rank_icir.csv
    2. 在workflow配置中引用此类作为handler
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# 添加alpha_factor_test目录到路径
ALPHA_DIR = Path(__file__).parent

sys.path.insert(0, str(ALPHA_DIR))

from qlib.data.dataset.handler import DataHandlerLP
from formula_parser import load_alpha101_formulas, load_alpha191_formulas
from alpha_calculator import AlphaCalculatorAuto
from qlib.contrib.data.handler import _DEFAULT_LEARN_PROCESSORS, _DEFAULT_INFER_PROCESSORS, check_transform_proc


class TopkAlphaHandler(DataHandlerLP):
    """
    自定义Handler，使用Top50 Alpha因子作为特征
    
    参数:
        instruments: 股票池
        start_time: 开始时间
        end_time: 结束时间
        topk_data_path: 预计算的Top50因子数据路径(pkl或csv)
    """
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        topk_data_path=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        
        if topk_data_path is None:
            topk_data_path = str(ALPHA_DIR / 'top50_features.pkl')
        
        # 保存配置供setup_data使用
        self.topk_data_path = topk_data_path
        
        # 构建数据加载器
        data_loader = {
            "class": "TopkAlphaLoader",
            "kwargs": {
                "topk_data_path": topk_data_path,
            },
        }
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )


class TopkAlphaLoader:
    """
    自定义DataLoader，从预计算文件加载Top50因子数据
    """
    
    def __init__(self, topk_data_path, **kwargs):
        self.topk_data_path = topk_data_path
    
    def load(self, instruments=None, start_time=None, end_time=None, **kwargs):
        """加载预计算的因子数据"""
        if not Path(self.topk_data_path).exists():
            raise FileNotFoundError(
                f"预计算数据文件不存在: {self.topk_data_path}\n"
                f"请先运行: python topk_alpha_handler.py --prepare"
            )
        
        data = pd.read_pickle(self.topk_data_path)
        
        if start_time:
            data = data.loc[data.index.get_level_values('datetime') >= start_time]
        if end_time:
            data = data.loc[data.index.get_level_values('datetime') <= end_time]
        
        return data


def prepare_top50_features(
    topk_csv_path,
    output_path,
    alpha101_path,
    alpha191_path,
    instruments="csi300",
    start_time="2020-01-01",
    end_time="2023-12-31",
):
    """
    预计算Top50因子值并保存为pickle文件
    """
    import qlib
    from qlib.data import D
    
    print("="*80)
    print("预计算Top50因子特征")
    print("="*80)
    
    # 初始化qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
    
    # 加载Top50因子列表
    top50_df = pd.read_csv(topk_csv_path)
    
    # 处理重复因子名称：给来自不同来源的同名因子加前缀
    from collections import Counter
    name_counter = Counter(top50_df['alpha_name'])
    duplicates = {name for name, count in name_counter.items() if count > 1}
    
    # 构建唯一的因子名称
    alpha_names = []
    alpha_sources = {}
    
    for _, row in top50_df.iterrows():
        original_name = row['alpha_name']
        source = row['source']
        
        if original_name in duplicates:
            # 重复的名称，加前缀区分
            unique_name = f"{source}_{original_name}"
        else:
            unique_name = original_name
        
        alpha_names.append(unique_name)
        alpha_sources[unique_name] = source
    
    # 加载公式
    alpha101_formulas = load_alpha101_formulas(alpha101_path)
    alpha191_formulas = load_alpha191_formulas(alpha191_path)
    
    # 构建公式字典
    formulas = {}
    for alpha_name in alpha_names:
        source = alpha_sources[alpha_name]
        # 去掉前缀获取原始名称
        original_name = alpha_name.replace('alpha101_', '').replace('alpha191_', '')
        
        if source == 'alpha101':
            if original_name in alpha101_formulas:
                formulas[alpha_name] = alpha101_formulas[original_name]
        elif source == 'alpha191':
            if original_name in alpha191_formulas:
                formulas[alpha_name] = alpha191_formulas[original_name]
    
    print(f"Top50因子数量: {len(alpha_names)}")
    print(f"可加载的公式数量: {len(formulas)}")
    
    # 加载股票数据
    print(f"\n加载股票数据...")
    fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
    
    # 获取指定时间范围内的股票池成分股列表
    instruments_config = D.instruments(instruments)
    stock_list = D.list_instruments(instruments_config, start_time=start_time, end_time=end_time, freq='day', as_list=True)
    print(f"  {instruments}在{start_time}~{end_time}期间的成分股数量: {len(stock_list)}")
    print(f"  前5只股票: {stock_list[:5]}")
    
    # 使用具体的股票列表获取数据
    df_data = D.features(stock_list, fields, start_time=start_time, end_time=end_time)
    
    df_data = df_data.rename(columns={
        '$open': 'open', '$high': 'high', '$low': 'low',
        '$close': 'close', '$volume': 'volume', '$factor': 'factor'
    })
    
    print(f"加载了 {len(df_data)} 条数据, {df_data.index.get_level_values(1).nunique()} 只股票")
    
    # 计算因子
    print(f"\n开始计算 {len(formulas)} 个因子...")
    calc = AlphaCalculatorAuto(df_data)
    
    feature_dict = {}
    success_count = 0
    fail_count = 0
    
    for alpha_name, formula in formulas.items():
        try:
            factor_values = calc.calculate_alpha(formula)
            feature_dict[alpha_name] = factor_values
            success_count += 1
            print(f"  [{success_count}] OK: {alpha_name}")
        except Exception as e:
            fail_count += 1
            print(f"  [{fail_count}] FAIL: {alpha_name} - {str(e)[:60]}")
    
    print(f"\n计算完成: 成功 {success_count}, 失败 {fail_count}")
    
    # 构建特征DataFrame
    feature_df = pd.DataFrame(feature_dict)
    
    # 构建标签（未来收益率）
    label = df_data.groupby(level='instrument')['close'].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )
    feature_df['LABEL0'] = label
    
    # 确保索引正确 - 检查并修复索引顺序
    if feature_df.index.names != ['datetime', 'instrument']:
        # 如果索引名称不正确，需要重新设置
        # qlib的D.features返回的MultiIndex顺序是(datetime, instrument)
        # 但有时索引名称可能未设置或错误
        current_names = feature_df.index.names
        print(f"  当前索引名称: {current_names}")
        
        # 检查是否索引顺序颠倒了
        level0_sample = feature_df.index.get_level_values(0)[:5].tolist()
        level1_sample = feature_df.index.get_level_values(1)[:5].tolist()
        
        # 如果level0看起来像股票代码（包含SH/SZ前缀），level1看起来像日期
        if any(isinstance(x, str) and ('SH' in str(x) or 'SZ' in str(x)) for x in level0_sample):
            # level0是instrument，顺序正确但名称错误
            feature_df.index = feature_df.index.rename(['instrument', 'datetime'])
            # 交换索引层级，使datetime在前
            feature_df = feature_df.swaplevel().sort_index()
            print(f"  已修复索引顺序: 从(instrument, datetime)改为(datetime, instrument)")
    
    # 确保最终索引名称正确
    feature_df.index.names = ['datetime', 'instrument']
    feature_df = feature_df.sort_index()
    
    # 打印验证信息
    print(f"\n索引验证:")
    print(f"  索引层级0前5个值: {feature_df.index.get_level_values(0)[:5].tolist()}")
    print(f"  索引层级1前5个值: {feature_df.index.get_level_values(1)[:5].tolist()}")
    print(f"  股票数量: {feature_df.index.get_level_values('instrument').nunique()}")
    print(f"  时间范围: {feature_df.index.get_level_values('datetime').min()} ~ {feature_df.index.get_level_values('datetime').max()}")
    
    # 保存
    feature_df.to_pickle(output_path)
    print(f"\n数据已保存到: {output_path}")
    print(f"数据形状: {feature_df.shape}")
    print(f"特征列: {list(feature_df.columns)}")
    print(f"时间范围: {feature_df.index.get_level_values('datetime').min()} ~ {feature_df.index.get_level_values('datetime').max()}")
    print(f"股票数量: {feature_df.index.get_level_values('instrument').nunique()}")


def main():
    parser = argparse.ArgumentParser(description='Top50因子预计算工具')
    parser.add_argument('--prepare', action='store_true', help='预计算Top50因子')
    parser.add_argument('--topk-csv', default=str(ALPHA_DIR / 'top50_by_rank_icir.csv'))
    parser.add_argument('--output', default=str(ALPHA_DIR / 'top50_features.pkl'))
    parser.add_argument('--alpha101-path', default=r'C:\Users\syk\Desktop\git_repo\auto_alpha\research_formula_candidates.txt')
    parser.add_argument('--alpha191-path', default=r'C:\Users\syk\Desktop\git_repo\auto_alpha\alpha191.txt')
    parser.add_argument('--instruments', default='csi300')
    parser.add_argument('--start-time', default='2020-01-01')
    parser.add_argument('--end-time', default='2023-12-31')
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_top50_features(
            topk_csv_path=args.topk_csv,
            output_path=args.output,
            alpha101_path=args.alpha101_path,
            alpha191_path=args.alpha191_path,
            instruments=args.instruments,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    else:
        print("请使用 --prepare 参数来预计算Top50因子")


if __name__ == "__main__":
    main()
