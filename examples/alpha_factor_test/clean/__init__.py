"""
Alpha因子挖掘与模型对比项目 - 清理重构版

模块结构:
    config.py          - 全局配置（路径、参数）
    formula_parser.py  - Alpha101/191公式解析器
    alpha_engine.py    - 因子计算引擎
    ic_analyzer.py     - IC/ICIR分析器
    data_manager.py    - 数据管理（qlib数据加载、股票池对齐）
    model_trainer.py   - XGBoost模型训练与对比
    ga_search.py       - 基于DEAP的遗传算法因子搜索
    run_pipeline.py    - 一键执行流水线
"""
