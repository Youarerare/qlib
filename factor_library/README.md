# Factor Library — 量化因子工程平台

基于 QLib 的量化因子管理、回测、搜索与入库平台，提供 Streamlit 可视化界面和命令行工具。

## 核心功能

| 模块 | 功能 | 入口 |
|------|------|------|
| 因子库管理 | SQLite 存储、CRUD、标签、排序、筛选 | `app.py` 页面1-3 |
| 在线回测 | 输入表达式即时回测，IC/ICIR/RankICIR + 分组收益 + IC分布 | `app.py` 页面8 |
| 批量搜索 | 逐个回测公式文件并自动入库 | `batch_search.py` |
| GA 遗传搜索 | 从种子公式出发，变异进化搜索新因子 | `batch_search.py --ga` |
| 单因子搜索 | 以一条公式为起点执行 GA 搜索 | `search_one.py` |
| 一键模型训练 | 从因子库选因子训练 XGBoost | `model_trainer.py` |
| 公式整理 | 合并多源公式、兼容性过滤 | `prepare_formulas.py` |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt        # Streamlit 等界面依赖
pip install -e .                       # qlib 本体（需在项目根目录）
```

### 启动可视化界面

```bash
streamlit run factor_library/app.py
```

8 个页面：因子列表 / 因子详情 / 因子对比 / 改进对比报告 / 自动入库记录 / 导入导出 / 一键模型训练 / 在线回测

### 准备种子公式

```bash
# 合并 alpha191.txt + research_formula_candidates.txt → all_formulas.txt
python -m factor_library.prepare_formulas
```

## 命令行工具

### 批量回测

```bash
# 回测 all_formulas.txt 中所有公式，满足阈值自动入库
python -m factor_library.batch_search

# 指定输入文件
python -m factor_library.batch_search --input my_formulas.txt

# 只回测不入库
python -m factor_library.batch_search --no-ingest
```

### GA 遗传搜索

```bash
# 混合模式：所有种子混入1个种群
python -m factor_library.batch_search --ga

# 逐种子模式（推荐）：每条公式独立进化
python -m factor_library.batch_search --ga --ga-per-seed

# 自参数：先筛选Top10种子，每条独立进化15代、种群40
python -m factor_library.batch_search --ga --ga-per-seed --ga-top-seed 10 --ga-per-gen 15 --ga-per-pop 40
```

**逐种子模式** 的核心逻辑：每条公式 → 解析为语法树 → 作为初始种群的1个个体 → 加随机个体填满种群 → 独立进化 N 代 → 收集结果。291 条种子 = 291 次独立搜索，探索不同方向。

### 单因子回测与搜索

```bash
# 回测单条公式
python -m factor_library.search_one

# 以指定公式为起点做 GA 搜索
python -m factor_library.search_one --formula "rank(ts_corr(volume, vwap, 10))"
```

### 模型训练

```bash
# 从因子库选 ICIR 最高的 50 个因子训练 XGBoost
python -m factor_library.model_trainer

python -m factor_library.model_trainer --top-k 30 --icir-min 0.3 --instruments csi300
```

## 项目结构

```
factor_library/
├── app.py                # Streamlit 可视化界面（8个页面）
├── config.py             # 全局配置（阈值、回测参数、数据路径）
├── database.py           # SQLite 因子库 CRUD
├── backtest_engine.py    # 统一回测引擎（IC/ICIR/分组收益）
├── batch_search.py       # 批量回测 + GA 搜索 CLI
├── search_one.py         # 单因子评估与 GA 搜索
├── model_trainer.py      # 一键 XGBoost 模型训练
├── prepare_formulas.py   # 合并多源公式、兼容性过滤
├── add_factor.py         # 手动添加因子脚本
├── factor_library.db     # SQLite 数据库
├── all_formulas.txt      # 种子公式池（291条）
├── requirements.txt      # 依赖
└── exports/              # 导出结果目录
```

## 因子表达式语法

支持的字段和算子：

**字段**: `open`, `high`, `low`, `close`, `volume`, `vwap`, `returns`, `adv5`~`adv180`

**截面算子**: `rank`, `scale`, `cs_mean`, `cs_std`, `cs_max`, `cs_min`, `cs_zscore`

**时序算子**: `ts_mean`, `ts_sum`, `ts_std`, `ts_max`, `ts_min`, `ts_arg_max`, `ts_arg_min`, `ts_corr`, `ts_covariance`, `ts_delta`, `ts_delay`, `ts_decay_linear`, `ts_product`, `ts_av_diff`, `ts_rank`

**数学**: `add`, `subtract`, `multiply`, `divide`, `abs`, `log`, `sign`, `sqrt`, `power`, `signed_power`, `max`, `min`

**逻辑**: `if_else`, `gt`, `lt`, `eq`

示例:
```
rank(ts_corr(high, volume, 5))
ts_arg_min(sqrt(max(cs_mean(ts_av_diff(adv5, 2)), abs(adv150))), 5)
-(low - close) * power(open, 5) / ((low - high) * power(close, 5))
```

## 入库阈值

| 指标 | 阈值 |
|------|------|
| ICIR | > 0.5 |
| \|IC均值\| | > 0.02 |
| IC胜率 | > 50% |

三个条件需同时满足，可在 `config.py` 中调整。

## 在线回测页面

`app.py` 第8个页面，功能包括：

- **快捷公式选择** — 8个预设公式一键填入
- **参数配置** — 股票池(csi300/csi500/all)、IC评估月数(1/3/6/12)
- **单周期回测** — 指标卡片 + 分组收益柱状图 + 逐日IC折线图 + IC分布直方图
- **多周期对比** — 1/3/6/12月同时回测，表格+柱状图对比
- **入库操作** — 满足阈值一键入库，不满足显示具体原因
