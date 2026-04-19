"""
公平对比: Top50因子 vs Alpha158
使用相同的股票池(csi300)和相同的时间范围
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr
from datetime import datetime

import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.processor import CSZScoreNorm

TOPK_PKL = r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test\top50_features.pkl'

# 用于存储共同的股票池
common_stocks = None

def get_common_stock_pool():
    """获取共同的股票池"""
    global common_stocks
    if common_stocks is not None:
        return common_stocks
    
    print("\n获取共同股票池...")
    # 获取Top50的股票列表
    top50_data = pd.read_pickle(TOPK_PKL)
    top50_stocks = set(top50_data.index.get_level_values('instrument').unique())
    print(f"  Top50股票数量: {len(top50_stocks)}")
    
    # 获取Alpha158的股票列表
    handler = Alpha158(
        instruments="csi300",
        start_time="2020-01-01",
        end_time="2023-06-01",
    )
    alpha158_data = handler.fetch()
    alpha158_stocks = set(alpha158_data.index.get_level_values('instrument').unique())
    print(f"  Alpha158股票数量: {len(alpha158_stocks)}")
    
    # 取交集
    common_stocks = sorted(list(top50_stocks & alpha158_stocks))
    print(f"  共同股票数量: {len(common_stocks)}")
    
    return common_stocks

def evaluate_ic(pred, y_true, test_index):
    """计算IC指标"""
    result_df = pd.DataFrame({
        'pred': pred,
        'label': y_true.values if hasattr(y_true, 'values') else y_true
    }, index=test_index)
    
    # 按日期计算IC
    ic_list = []
    rank_ic_list = []
    
    for date in result_df.index.get_level_values('datetime').unique():
        try:
            day_data = result_df.loc[date]
        except:
            continue
            
        if len(day_data) > 10:
            try:
                ic, _ = pearsonr(day_data['pred'], day_data['label'])
                rank_ic, _ = spearmanr(day_data['pred'], day_data['label'])
                ic_list.append(ic)
                rank_ic_list.append(rank_ic)
            except:
                pass
    
    if len(ic_list) == 0:
        return None
    
    ic_arr = np.array(ic_list)
    rank_ic_arr = np.array(rank_ic_list)
    
    return {
        'IC_mean': ic_arr.mean(),
        'IC_std': ic_arr.std(),
        'ICIR': ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
        'IC_positive_ratio': (ic_arr > 0).sum() / len(ic_arr),
        'Rank_IC_mean': rank_ic_arr.mean(),
        'Rank_IC_std': rank_ic_arr.std(),
        'Rank_ICIR': rank_ic_arr.mean() / rank_ic_arr.std() if rank_ic_arr.std() > 0 else 0,
        'num_periods': len(ic_arr)
    }


def prepare_top50_data(common_stocks=None):
    """准备Top50因子数据"""
    print("\n加载Top50因子数据...")
    data = pd.read_pickle(TOPK_PKL)
    
    # 使用共同股票池过滤
    if common_stocks:
        data = data.loc[data.index.get_level_values('instrument').isin(common_stocks)]
        print(f"  使用共同股票池过滤后: {data.index.get_level_values('instrument').nunique()} 只股票")
    
    # 确保索引名称正确
    data.index = data.index.rename(['datetime', 'instrument'])
    
    # 分离特征和标签
    X = data.drop(columns=['LABEL0'])
    y = data['LABEL0']
    
    # 按时间划分
    dt_idx = data.index.get_level_values('datetime')
    train_mask = dt_idx < '2022-01-01'
    test_mask = dt_idx >= '2022-06-01'
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # 去除NaN和inf
    for name, df in [('train', X_train), ('test', X_test)]:
        df = df.replace([np.inf, -np.inf], np.nan)
        valid = ~df.isna().any(axis=1)
        if name == 'train':
            X_train = df[valid]
            y_train = y_train[valid]
        else:
            X_test = df[valid]
            y_test = y_test[valid]
    
    # 去除标签中的NaN
    valid_train = ~y_train.isna()
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    
    valid_test = ~y_test.isna()
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]
    
    # CSZScoreNorm标准化
    print("  进行特征标准化（CSZScoreNorm）...")
    csz = CSZScoreNorm()
    X_train = csz(X_train)
    X_test = csz(X_test)
    
    # 删除标准化后产生的NaN
    valid_train_final = ~X_train.isna().any(axis=1)
    X_train = X_train[valid_train_final]
    y_train = y_train.loc[X_train.index]
    
    valid_test_final = ~X_test.isna().any(axis=1)
    X_test = X_test[valid_test_final]
    y_test = y_test.loc[X_test.index]
    
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  股票池: {X_train.index.get_level_values('instrument').nunique()} 只")
    
    return X_train, y_train, X_test, y_test


def prepare_alpha158_data(common_stocks=None):
    """准备Alpha158数据"""
    print("\n加载Alpha158数据...")
    
    # 使用与Top50相同的时间范围和股票池
    handler = Alpha158(
        instruments="csi300",
        start_time="2020-01-01",
        end_time="2023-06-01",
    )
    
    # 获取数据
    data = handler.fetch()
    
    # 使用共同股票池过滤
    if common_stocks:
        data = data.loc[data.index.get_level_values('instrument').isin(common_stocks)]
        print(f"  使用共同股票池过滤后: {data.index.get_level_values('instrument').nunique()} 只股票")
    
    # 分离特征和标签
    X = data.drop(columns=['LABEL0'])
    y = data['LABEL0']
    
    # 按时间划分（与Top50相同）
    dt_idx = data.index.get_level_values('datetime')
    train_mask = dt_idx < '2022-01-01'
    test_mask = dt_idx >= '2022-06-01'
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # 去除NaN和inf
    for name, df in [('train', X_train), ('test', X_test)]:
        df = df.replace([np.inf, -np.inf], np.nan)
        valid = ~df.isna().any(axis=1)
        if name == 'train':
            X_train = df[valid]
            y_train = y_train[valid]
        else:
            X_test = df[valid]
            y_test = y_test[valid]
    
    # 去除标签中的NaN
    valid_train = ~y_train.isna()
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    
    valid_test = ~y_test.isna()
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]
    
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  股票池: {X_train.index.get_level_values('instrument').nunique()} 只")
    
    return X_train, y_train, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    """训练XGBoost并评估"""
    print(f"\n训练 {model_name} XGBoost模型...")
    
    model = XGBRegressor(
        eval_metric='rmse',
        colsample_bytree=0.8879,
        learning_rate=0.0421,
        max_depth=8,
        n_estimators=647,
        subsample=0.8789,
        n_jobs=20,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    metrics = evaluate_ic(pred, y_test, X_test.index)
    
    if metrics:
        print(f"\n{model_name} IC评估结果:")
        print(f"  IC均值: {metrics['IC_mean']:.6f}")
        print(f"  IC标准差: {metrics['IC_std']:.6f}")
        print(f"  ICIR: {metrics['ICIR']:.4f}")
        print(f"  IC>0比例: {metrics['IC_positive_ratio']:.4f}")
        print(f"  RankIC均值: {metrics['Rank_IC_mean']:.6f}")
        print(f"  RankICIR: {metrics['Rank_ICIR']:.4f}")
    
    return metrics


def main():
    print("="*80)
    print("公平对比: Top50因子 vs Alpha158")
    print("="*80)
    print(f"时间: {datetime.now()}")
    print(f"股票池: csi300")
    print(f"训练集: < 2022-01-01")
    print(f"测试集: >= 2022-06-01")
    print(f"特征标准化: CSZScoreNorm")
    
    # 初始化qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
    
    # 0. 获取共同股票池
    common_stocks = get_common_stock_pool()
    
    # 1. Top50因子
    print("\n" + "="*80)
    print("模型1: Top50因子")
    print("="*80)
    
    X_train_top50, y_train_top50, X_test_top50, y_test_top50 = prepare_top50_data(common_stocks)
    metrics_top50 = train_and_evaluate(X_train_top50, y_train_top50, X_test_top50, y_test_top50, "Top50")
    
    # 2. Alpha158
    print("\n" + "="*80)
    print("模型2: Alpha158")
    print("="*80)
    
    X_train_a158, y_train_a158, X_test_a158, y_test_a158 = prepare_alpha158_data(common_stocks)
    metrics_a158 = train_and_evaluate(X_train_a158, y_train_a158, X_test_a158, y_test_a158, "Alpha158")
    
    # 3. 结果对比
    print("\n" + "="*80)
    print("对比结果总结")
    print("="*80)
    
    if metrics_top50 and metrics_a158:
        print(f"\n{'模型':<12} {'IC均值':>10} {'ICIR':>10} {'RankIC均值':>12} {'RankICIR':>10} {'IC>0比例':>10} {'训练样本':>10} {'测试样本':>10}")
        print("-"*90)
        print(f"{'Top50':<12} {metrics_top50['IC_mean']:>10.6f} {metrics_top50['ICIR']:>10.4f} {metrics_top50['Rank_IC_mean']:>12.6f} {metrics_top50['Rank_ICIR']:>10.4f} {metrics_top50['IC_positive_ratio']:>10.4f} {len(X_train_top50):>10} {len(X_test_top50):>10}")
        print(f"{'Alpha158':<12} {metrics_a158['IC_mean']:>10.6f} {metrics_a158['ICIR']:>10.4f} {metrics_a158['Rank_IC_mean']:>12.6f} {metrics_a158['Rank_ICIR']:>10.4f} {metrics_a158['IC_positive_ratio']:>10.4f} {len(X_train_a158):>10} {len(X_test_a158):>10}")
        
        print(f"\n{'-'*90}")
        # 判断谁更好
        if metrics_top50['Rank_ICIR'] > metrics_a158['Rank_ICIR']:
            print(f"✓ Top50因子RankICIR ({metrics_top50['Rank_ICIR']:.4f}) > Alpha158 ({metrics_a158['Rank_ICIR']:.4f})")
        else:
            print(f"✓ Alpha158因子RankICIR ({metrics_a158['Rank_ICIR']:.4f}) > Top50 ({metrics_top50['Rank_ICIR']:.4f})")
        
        if metrics_top50['ICIR'] > metrics_a158['ICIR']:
            print(f"✓ Top50因子ICIR ({metrics_top50['ICIR']:.4f}) > Alpha158 ({metrics_a158['ICIR']:.4f})")
        else:
            print(f"✓ Alpha158因子ICIR ({metrics_a158['ICIR']:.4f}) > Top50 ({metrics_top50['ICIR']:.4f})")
    
    # 4. 保存结果
    if metrics_top50 and metrics_a158:
        results_df = pd.DataFrame({
            'Top50': metrics_top50,
            'Alpha158': metrics_a158
        }).T
        results_df.to_csv(r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test\xgboost_fair_comparison.csv')
        print(f"\n结果已保存到: xgboost_fair_comparison.csv")
    
    print(f"\n完成时间: {datetime.now()}")


if __name__ == "__main__":
    main()
