"""运行修复后的模型对比"""
import sys
sys.path.insert(0, r'C:\Users\syk\Desktop\git_repo\qlib\examples\alpha_factor_test')

if __name__ == '__main__':
    from clean.model_trainer import run_comparison
    from clean.config import OUTPUT_DIR

    results = run_comparison(str(OUTPUT_DIR / "all_features.pkl"))
    results.to_csv(OUTPUT_DIR / "model_comparison_fixed.csv", index=False)

    print("\n" + "=" * 80)
    print("修复后对比结果:")
    print("=" * 80)
    for _, row in results.iterrows():
        print(f"  {row['model']}: ICIR={row['icir']:.4f}, RankICIR={row['rank_icir']:.4f}, "
              f"训练样本={row['train_samples']}, 测试样本={row['test_samples']}, "
              f"特征数={row['n_features']}")
