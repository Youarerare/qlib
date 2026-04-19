"""
因子库可视化筛选界面 - 基于 Streamlit

功能:
1. 因子列表页 - 展示所有因子，支持排序筛选
2. 因子详情页 - 点击查看公式、指标、分组收益
3. 对比功能 - 勾选2~4个因子对比IC序列、分组收益
4. 改进对比报告 - 原始因子 vs 改进因子对比
5. 导入/导出 - CSV/Excel 导出
6. 自动入库记录 - 查看最近自动入库的因子
7. 一键模型训练 - 筛选因子后启动模型训练

用法:
    streamlit run factor_library/app.py
    # 或
    python -m streamlit run factor_library/app.py
"""
import sys
import json
import logging
from pathlib import Path

# 确保 import 路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALPHA_FACTOR_DIR = _PROJECT_ROOT / "examples" / "alpha_factor_test"
for p in [_PROJECT_ROOT, _ALPHA_FACTOR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st
import pandas as pd
import numpy as np

from factor_library.database import (
    init_db, get_all_factors, get_factor_by_id, get_recent_factors,
    get_auto_ingested_factors, add_factor, update_factor, delete_factor,
    set_best_factor, export_to_csv, export_to_excel, import_from_csv,
    get_factors_for_training,
)
from factor_library.config import (
    ORIGINAL_FACTOR, IMPROVED_FACTORS, THRESHOLD, BACKTEST, DB_PATH
)

# 页面配置
st.set_page_config(
    page_title="因子回测管理与可视化筛选系统",
    page_icon="📊",
    layout="wide",
)

# 初始化数据库
init_db()

# 侧边栏导航
page = st.sidebar.selectbox(
    "选择页面",
    ["📋 因子列表", "🔍 因子详情", "⚖️ 因子对比", "📊 改进对比报告",
     "📥 自动入库记录", "💾 导入/导出", "🚀 一键模型训练", "🧪 在线回测"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**数据库路径**: `{DB_PATH}`")
st.sidebar.markdown(f"**入库阈值**: ICIR>{THRESHOLD.icir}, |IC|>{THRESHOLD.ic_mean}")
st.sidebar.markdown(f"**改进阈值**: ICIR>{THRESHOLD.improved_icir}, |IC|>{THRESHOLD.improved_ic_mean}")


# ============================================================
# 页面1: 因子列表
# ============================================================
def page_factor_list():
    st.title("📋 因子列表")

    # 筛选区域
    with st.expander("筛选条件", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            sort_by = st.selectbox(
                "排序字段",
                ["icir", "ic_mean", "rank_icir", "rank_ic", "long_short_return",
                 "top_group_excess", "ic_win_rate", "created_at"],
                index=0,
            )
            ascending = st.checkbox("升序", value=False)

        with col2:
            tags_filter = st.text_input("标签筛选", "")
            icir_min = st.number_input("ICIR 最小值", value=0.0, step=0.1)
            ic_min = st.number_input("|IC均值| 最小值", value=0.0, step=0.01)

        with col3:
            is_best = st.selectbox("是否最优", ["全部", "是最优", "非最优"])
            date_from = st.text_input("入库起始日期", "")
            date_to = st.text_input("入库截止日期", "")

    # 构建筛选条件
    filter_by = {}
    if tags_filter:
        filter_by["tags"] = tags_filter
    if icir_min > 0:
        filter_by["icir_min"] = icir_min
    if ic_min > 0:
        filter_by["ic_mean_min"] = ic_min
    if is_best == "是最优":
        filter_by["is_best"] = 1
    elif is_best == "非最优":
        filter_by["is_best"] = 0
    if date_from:
        filter_by["date_from"] = date_from
    if date_to:
        filter_by["date_to"] = date_to

    # 查询数据
    df = get_all_factors(sort_by=sort_by, ascending=ascending, filter_by=filter_by if filter_by else None)

    if df.empty:
        st.info("因子库为空，请先添加因子或运行回测。")
        return

    st.markdown(f"共 **{len(df)}** 个因子")

    # 选择因子用于对比/详情
    display_cols = [
        "factor_id", "expression", "ic_mean", "icir", "ic_win_rate",
        "rank_ic", "rank_icir", "long_short_return", "top_group_excess",
        "is_best", "tags", "created_at"
    ]
    available_cols = [c for c in display_cols if c in df.columns]

    # 截断表达式显示
    display_df = df[available_cols].copy()
    if "expression" in display_df.columns:
        display_df["expression"] = display_df["expression"].apply(
            lambda x: x[:60] + "..." if len(str(x)) > 60 else x
        )
    if "is_best" in display_df.columns:
        display_df["is_best"] = display_df["is_best"].apply(lambda x: "⭐" if x else "")

    # 使用 data_editor 允许行选择
    st.dataframe(display_df, use_container_width=True, height=500)

    # 操作区
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_id = st.text_input("输入 factor_id 查看详情", "")
        if st.button("查看详情") and selected_id:
            st.session_state["view_factor_id"] = selected_id
            st.info(f"请切换到「因子详情」页面查看 {selected_id}")

    with col2:
        best_id = st.text_input("设置最优因子 factor_id", "")
        if st.button("设为最优") and best_id:
            set_best_factor(best_id)
            st.success(f"已设置 {best_id} 为最优因子")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

    with col3:
        del_id = st.text_input("删除因子 factor_id", "")
        if st.button("删除因子") and del_id:
            if delete_factor(del_id):
                st.success(f"已删除 {del_id}")
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            else:
                st.error(f"未找到 {del_id}")


# ============================================================
# 页面2: 因子详情
# ============================================================
def page_factor_detail():
    st.title("🔍 因子详情")

    factor_id = st.text_input(
        "输入 factor_id",
        value=st.session_state.get("view_factor_id", ""),
    )

    if not factor_id:
        st.info("请输入 factor_id 或从因子列表页跳转。")
        return

    info = get_factor_by_id(factor_id)
    if not info:
        st.error(f"未找到因子: {factor_id}")
        return

    # 基本信息
    st.subheader("基本信息")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**因子ID**: `{info['factor_id']}`")
        st.markdown(f"**表达式**:")
        st.code(info["expression"], language="python")
        st.markdown(f"**描述**: {info.get('description', '-')}")
        st.markdown(f"**标签**: {info.get('tags', '-')}")
        st.markdown(f"**股票池**: {info.get('asset_universe', '-')}")
        st.markdown(f"**入库时间**: {info.get('created_at', '-')}")
        st.markdown(f"**更新时间**: {info.get('updated_at', '-')}")

    with col2:
        st.markdown(f"**是否最优**: {'⭐ 是' if info.get('is_best') else '否'}")
        st.markdown(f"**回测日期**: {info.get('test_start_date', '-')} ~ {info.get('test_end_date', '-')}")

    # 核心指标
    st.subheader("核心指标")
    metric_cols = {
        "IC均值": "ic_mean",
        "ICIR": "icir",
        "IC胜率": "ic_win_rate",
        "Rank IC": "rank_ic",
        "Rank ICIR": "rank_icir",
        "多空收益": "long_short_return",
        "第一组超额": "top_group_excess",
        "换手率": "turnover",
        "最大回撤": "max_drawdown",
    }

    col_widths = [1] * min(3, len(metric_cols))
    metrics_display = {}
    for i, (label, key) in enumerate(metric_cols.items()):
        val = info.get(key, 0) or 0
        metrics_display[label] = val

    # 显示为指标卡片
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("IC均值", f"{metrics_display['IC均值']:+.4f}")
    with m_col2:
        st.metric("ICIR", f"{metrics_display['ICIR']:+.4f}")
    with m_col3:
        st.metric("IC胜率", f"{metrics_display['IC胜率']:.2%}")
    with m_col4:
        st.metric("Rank ICIR", f"{metrics_display['Rank ICIR']:+.4f}")
    with m_col5:
        st.metric("多空收益", f"{metrics_display['多空收益']:+.4f}")

    m_col6, m_col7, m_col8, m_col9 = st.columns(4)
    with m_col6:
        st.metric("Rank IC", f"{metrics_display['Rank IC']:+.4f}")
    with m_col7:
        st.metric("第一组超额", f"{metrics_display['第一组超额']:+.4f}")
    with m_col8:
        st.metric("换手率", f"{metrics_display['换手率']:.4f}")
    with m_col9:
        st.metric("最大回撤", f"{metrics_display['最大回撤']:.4f}")

    # 分组收益图
    st.subheader("分组收益")
    group_returns_str = info.get("group_returns", "")
    if group_returns_str:
        try:
            group_returns = json.loads(group_returns_str)
            if group_returns:
                # 过滤掉 long_short 和 top_excess
                group_only = {k: v for k, v in group_returns.items()
                              if k.startswith("g") and k[1:].isdigit()}
                if group_only:
                    gr_df = pd.DataFrame(
                        list(group_only.items()),
                        columns=["分组", "日均收益"],
                    )
                    st.bar_chart(gr_df.set_index("分组"))

                # 多空和超额信息
                ls = group_returns.get("long_short", 0)
                te = group_returns.get("top_excess", 0)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("多空收益", f"{ls:+.4f}")
                with col2:
                    st.metric("第一组超额", f"{te:+.4f}")
            else:
                st.info("暂无分组收益数据")
        except json.JSONDecodeError:
            st.warning("分组收益数据格式异常")
    else:
        st.info("暂无分组收益数据（分组收益需在回测时计算并存储）")

    # 更新标签/描述
    st.subheader("编辑因子信息")
    new_desc = st.text_area("描述", value=info.get("description", ""))
    new_tags = st.text_input("标签", value=info.get("tags", ""))

    if st.button("更新信息"):
        update_factor(factor_id, description=new_desc, tags=new_tags)
        st.success("更新成功!")
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()


# ============================================================
# 页面3: 因子对比
# ============================================================
def page_factor_compare():
    st.title("⚖️ 因子对比")

    # 获取所有因子
    df = get_all_factors()
    if df.empty:
        st.info("因子库为空，无法对比。")
        return

    # 选择因子
    factor_options = {
        f"{row['factor_id']} | ICIR={row.get('icir', 0):+.3f} | {row['expression'][:40]}": row["factor_id"]
        for _, row in df.iterrows()
    }

    selected = st.multiselect(
        "选择 2~4 个因子进行对比",
        list(factor_options.keys()),
        max_selections=4,
    )

    if len(selected) < 2:
        st.info("请至少选择 2 个因子进行对比。")
        return

    selected_ids = [factor_options[s] for s in selected]
    factors_data = []
    for fid in selected_ids:
        info = get_factor_by_id(fid)
        if info:
            factors_data.append(info)

    # 对比表格
    st.subheader("指标对比")
    compare_rows = []
    for f in factors_data:
        compare_rows.append({
            "因子ID": f["factor_id"],
            "表达式": f["expression"][:50],
            "IC均值": f.get("ic_mean", 0),
            "ICIR": f.get("icir", 0),
            "IC胜率": f.get("ic_win_rate", 0),
            "Rank IC": f.get("rank_ic", 0),
            "Rank ICIR": f.get("rank_icir", 0),
            "多空收益": f.get("long_short_return", 0),
            "第一组超额": f.get("top_group_excess", 0),
            "标签": f.get("tags", ""),
        })
    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(compare_df, use_container_width=True)

    # 分组收益对比柱状图
    st.subheader("分组收益对比")
    group_data = {}
    for f in factors_data:
        gr_str = f.get("group_returns", "")
        if gr_str:
            try:
                gr = json.loads(gr_str)
                group_only = {k: v for k, v in gr.items() if k.startswith("g") and k[1:].isdigit()}
                group_data[f["factor_id"]] = group_only
            except json.JSONDecodeError:
                pass

    if group_data:
        chart_df = pd.DataFrame(group_data)
        chart_df.index.name = "分组"
        st.bar_chart(chart_df)
    else:
        st.info("暂无分组收益数据")

    # 指标雷达图（用柱状图替代）
    st.subheader("关键指标柱状图对比")
    metrics_to_compare = ["ic_mean", "icir", "rank_icir", "ic_win_rate", "long_short_return"]
    chart_data = {}
    for f in factors_data:
        chart_data[f["factor_id"]] = {m: f.get(m, 0) or 0 for m in metrics_to_compare}
    radar_df = pd.DataFrame(chart_data)
    st.bar_chart(radar_df)


# ============================================================
# 页面4: 改进对比报告
# ============================================================
def page_improved_report():
    st.title("📊 改进对比报告")

    st.markdown(f"""
    **原始因子**:
    ```
    {ORIGINAL_FACTOR}
    ```

    **三个改进因子**:

    | 编号 | 表达式 | 说明 |
    |------|--------|------|
    | A | `{IMPROVED_FACTORS['A']['expression']}` | {IMPROVED_FACTORS['A']['description']} |
    | B | `{IMPROVED_FACTORS['B']['expression']}` | {IMPROVED_FACTORS['B']['description']} |
    | C | `{IMPROVED_FACTORS['C']['expression']}` | {IMPROVED_FACTORS['C']['description']} |
    """)

    # 从数据库查询已有的对比结果
    df = get_all_factors(sort_by="icir", ascending=False)

    # 查找原始因子和改进因子
    orig_factor = None
    improved_factors_db = {}
    for _, row in df.iterrows():
        if row["expression"] == ORIGINAL_FACTOR:
            orig_factor = row
        for name, info in IMPROVED_FACTORS.items():
            if row["expression"] == info["expression"]:
                improved_factors_db[name] = row

    if orig_factor is None and not improved_factors_db:
        st.warning("尚未运行改进因子对比回测。请先运行:")
        st.code("python -m factor_library.test_improved_factors", language="bash")
        return

    # 显示对比表格
    st.subheader("对比结果")

    rows = []
    if orig_factor is not None:
        rows.append({
            "名称": "原始因子",
            "表达式": orig_factor["expression"][:50],
            "IC均值": orig_factor.get("ic_mean", 0),
            "ICIR": orig_factor.get("icir", 0),
            "IC胜率": orig_factor.get("ic_win_rate", 0),
            "Rank ICIR": orig_factor.get("rank_icir", 0),
            "多空收益": orig_factor.get("long_short_return", 0),
            "标签": orig_factor.get("tags", ""),
        })
    else:
        rows.append({"名称": "原始因子", "表达式": ORIGINAL_FACTOR[:50], "备注": "未入库"})

    for name in ["A", "B", "C"]:
        if name in improved_factors_db:
            f = improved_factors_db[name]
            is_better = "优于原始" in f.get("tags", "")
            rows.append({
                "名称": f"改进{name}",
                "表达式": f["expression"][:50],
                "IC均值": f.get("ic_mean", 0),
                "ICIR": f.get("icir", 0),
                "IC胜率": f.get("ic_win_rate", 0),
                "Rank ICIR": f.get("rank_icir", 0),
                "多空收益": f.get("long_short_return", 0),
                "标签": f.get("tags", ""),
                "优于原始": "✅ 是" if is_better else "❌ 否",
            })
        else:
            rows.append({"名称": f"改进{name}", "备注": "未入库"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 分组收益对比
    st.subheader("分组收益对比")
    all_group_data = {}
    if orig_factor is not None:
        gr_str = orig_factor.get("group_returns", "")
        if gr_str:
            try:
                gr = json.loads(gr_str)
                group_only = {k: v for k, v in gr.items() if k.startswith("g") and k[1:].isdigit()}
                all_group_data["原始因子"] = group_only
            except:
                pass

    for name, f in improved_factors_db.items():
        gr_str = f.get("group_returns", "")
        if gr_str:
            try:
                gr = json.loads(gr_str)
                group_only = {k: v for k, v in gr.items() if k.startswith("g") and k[1:].isdigit()}
                all_group_data[f"改进{name}"] = group_only
            except:
                pass

    if all_group_data:
        chart_df = pd.DataFrame(all_group_data)
        chart_df.index.name = "分组"
        st.bar_chart(chart_df)
    else:
        st.info("暂无分组收益数据")

    # 运行对比按钮
    st.markdown("---")
    st.subheader("运行对比回测")
    st.info("点击下方按钮运行原始因子与改进因子的对比回测（需要 QLib 数据环境）")
    if st.button("🔄 运行改进因子对比回测"):
        with st.spinner("正在运行对比回测，请耐心等待..."):
            from factor_library.test_improved_factors import compare_factors
            comparison = compare_factors()
            st.success("对比回测完成!")
            st.dataframe(comparison, use_container_width=True)
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()


# ============================================================
# 页面5: 自动入库记录
# ============================================================
def page_auto_ingest():
    st.title("📥 自动入库记录")

    df = get_auto_ingested_factors(limit=100)

    if df.empty:
        st.info("暂无自动入库的因子。")
    else:
        st.markdown(f"共 **{len(df)}** 个自动入库因子")

        display_cols = [
            "factor_id", "expression", "ic_mean", "icir", "ic_win_rate",
            "rank_icir", "long_short_return", "tags", "created_at"
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        display_df = df[available_cols].copy()

        if "expression" in display_df.columns:
            display_df["expression"] = display_df["expression"].apply(
                lambda x: x[:50] + "..." if len(str(x)) > 50 else x
            )

        st.dataframe(display_df, use_container_width=True, height=400)

    # 最近入库
    st.markdown("---")
    st.subheader("最近入库因子（全部）")
    recent_df = get_recent_factors(limit=20)

    if not recent_df.empty:
        display_cols = [
            "factor_id", "expression", "ic_mean", "icir", "ic_win_rate",
            "rank_icir", "tags", "created_at"
        ]
        available_cols = [c for c in display_cols if c in recent_df.columns]
        display_df = recent_df[available_cols].copy()

        if "expression" in display_df.columns:
            display_df["expression"] = display_df["expression"].apply(
                lambda x: x[:50] + "..." if len(str(x)) > 50 else x
            )

        st.dataframe(display_df, use_container_width=True)


# ============================================================
# 页面6: 导入/导出
# ============================================================
def page_import_export():
    st.title("💾 导入/导出")

    # 导出
    st.subheader("导出因子库")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("导出 CSV"):
            path = export_to_csv()
            st.success(f"已导出: {path}")
            with open(path, "rb") as f:
                st.download_button(
                    "下载 CSV",
                    data=f.read(),
                    file_name=Path(path).name,
                    mime="text/csv",
                )
    with col2:
        if st.button("导出 Excel"):
            try:
                path = export_to_excel()
                st.success(f"已导出: {path}")
                with open(path, "rb") as f:
                    st.download_button(
                        "下载 Excel",
                        data=f.read(),
                        file_name=Path(path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(f"Excel导出失败: {e}，请安装 openpyxl 或 xlsxwriter")

    # 导入
    st.subheader("导入因子")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded_file is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            count = import_from_csv(tmp_path)
            st.success(f"导入完成: {count} 条因子")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")


# ============================================================
# 页面7: 一键模型训练
# ============================================================
def page_model_training():
    st.title("🚀 一键模型训练")

    st.markdown("""
    选择因子后，可以一键启动 XGBoost 模型训练。
    系统将自动使用选中的因子表达式作为特征，进行训练与评估。
    """)

    # 筛选因子
    df = get_all_factors(sort_by="icir", ascending=False)
    if df.empty:
        st.warning("因子库为空，请先添加因子。")
        return

    st.subheader("选择训练因子")

    # 筛选条件
    col1, col2 = st.columns(2)
    with col1:
        icir_threshold = st.slider("ICIR 最小值", 0.0, 3.0, 0.3, step=0.1)
    with col2:
        max_factors = st.slider("最大因子数", 5, 100, 50, step=5)

    # 筛选后的因子
    filtered = df[df["icir"].abs() >= icir_threshold].head(max_factors)

    if filtered.empty:
        st.info("无满足条件的因子")
        return

    st.markdown(f"满足条件的因子: **{len(filtered)}** 个")

    # 显示可选因子
    factor_options = {
        f"{row['factor_id']} | ICIR={row.get('icir', 0):+.3f} | {row['expression'][:40]}": row["expression"]
        for _, row in filtered.iterrows()
    }

    selected = st.multiselect(
        "选择因子（可多选，默认选择所有满足条件的因子）",
        list(factor_options.keys()),
        default=list(factor_options.keys()),  # 默认全选
    )

    if not selected:
        st.info("请至少选择一个因子")
        return

    selected_expressions = [factor_options[s] for s in selected]
    st.info(f"已选择 {len(selected_expressions)} 个因子用于训练")

    # 训练配置
    st.subheader("训练配置")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.number_input("n_estimators", value=1000, step=100)
        max_depth = st.number_input("max_depth", value=8, step=1)
    with col2:
        learning_rate = st.number_input("learning_rate", value=0.0421, step=0.01, format="%.4f")
        subsample = st.number_input("subsample", value=0.8789, step=0.01, format="%.4f")
    with col3:
        colsample_bytree = st.number_input("colsample_bytree", value=0.8879, step=0.01, format="%.4f")
        instruments = st.selectbox("股票池", ["csi300", "csi500", "all"], index=0)

    # 开始训练按钮
    if st.button("🚀 开始训练", type="primary"):
        st.info(f"正在训练模型，使用 {len(selected_expressions)} 个因子...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 动态导入训练模块
            sys.path.insert(0, str(_ALPHA_FACTOR_DIR))
            from clean.data_manager import init_qlib, load_ohlcv, load_alpha158_data
            from clean.alpha_engine import AlphaEngine, compute_factors
            from clean.model_trainer import train_xgboost, evaluate_predictions
            from clean.data_manager import (
                get_common_stocks, filter_by_stocks, split_train_test,
                clean_features, apply_cszscorenorm, winsorize_label,
            )
            from clean.config import BACKTEST as CLEAN_BACKTEST

            # Step 1: 初始化
            status_text.text("初始化 QLib...")
            progress_bar.progress(10)
            init_qlib()

            # Step 2: 加载数据并计算因子
            status_text.text("加载数据并计算因子...")
            progress_bar.progress(30)

            from qlib.data import D
            stock_list = D.list_instruments(
                D.instruments(instruments),
                start_time=CLEAN_BACKTEST.start_time,
                end_time=CLEAN_BACKTEST.end_time,
                freq="day", as_list=True,
            )
            fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
            raw_df = D.features(stock_list, fields,
                                start_time=CLEAN_BACKTEST.start_time,
                                end_time=CLEAN_BACKTEST.end_time)
            raw_df = raw_df.rename(columns={
                "$open": "open", "$high": "high", "$low": "low",
                "$close": "close", "$volume": "volume", "$factor": "factor",
            })
            raw_df.index = raw_df.index.rename(["datetime", "instrument"])

            if raw_df.index.get_level_values(0).dtype == object:
                raw_df = raw_df.swaplevel().sort_index()
                raw_df.index = raw_df.index.rename(["datetime", "instrument"])

            engine = AlphaEngine(raw_df)
            formulas_dict = {f"f_{i}": expr for i, expr in enumerate(selected_expressions)}
            feature_df = compute_factors(raw_df, formulas_dict)

            # Step 3: 准备训练数据
            status_text.text("准备训练数据...")
            progress_bar.progress(50)

            returns = raw_df.groupby(level="instrument")["close"].transform(
                lambda x: x.shift(-2) / x.shift(-1) - 1
            )
            feature_df["LABEL0"] = returns

            label = feature_df["LABEL0"]
            features = feature_df.drop(columns=["LABEL0"])

            # Split
            dt = features.index.get_level_values("datetime")
            train_mask = dt < CLEAN_BACKTEST.train_end
            test_mask = dt >= CLEAN_BACKTEST.test_start

            X_train = features[train_mask].fillna(0).replace([np.inf, -np.inf], 0)
            X_test = features[test_mask].fillna(0).replace([np.inf, -np.inf], 0)
            y_train = label[train_mask]
            y_test = label[test_mask]

            # Step 4: 训练
            status_text.text("训练 XGBoost 模型...")
            progress_bar.progress(70)

            pred = train_xgboost(
                X_train, y_train, X_test,
                config=None,  # 使用默认配置
            )

            # Step 5: 评估
            status_text.text("评估模型...")
            progress_bar.progress(90)

            metrics = evaluate_predictions(pred, y_test, X_test.index)
            progress_bar.progress(100)

            # 显示结果
            st.subheader("训练结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("IC均值", f"{metrics.get('ic_mean', 0):+.4f}")
            with col2:
                st.metric("ICIR", f"{metrics.get('icir', 0):+.4f}")
            with col3:
                st.metric("Rank ICIR", f"{metrics.get('rank_icir', 0):+.4f}")

            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("IC胜率", f"{metrics.get('ic_positive_ratio', 0):.2%}")
            with col5:
                st.metric("训练样本数", f"{len(X_train)}")
            with col6:
                st.metric("测试样本数", f"{len(X_test)}")

            st.success("训练完成!")

        except Exception as e:
            st.error(f"训练失败: {e}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================
# 页面8: 在线回测
# ============================================================
def page_online_backtest():
    st.title("🧪 在线回测")
    st.markdown("输入因子表达式，实时回测并展示 IC/ICIR/RankICIR 等指标。")

    # ---- 快捷公式选择 ----
    preset_formulas = {
        "自定义": "",
        "原始因子 (ts_av_diff)": "ts_arg_min(sqrt(max(cs_mean(ts_av_diff(adv5, 2)), abs(adv150))), 5)",
        "Alpha101: (close-open)/(high-low)": "((close - open) / ((high - low) + .001))",
        "Alpha042: rank(vwap-close)/rank(vwap+close)": "rank(vwap - close) / rank(vwap + close)",
        "Alpha044: -ts_corr(high,rank(volume),5)": "-ts_corr(high, rank(volume), 5)",
        "Alpha054: -(low-close)*power(open,5)": "-(low - close) * power(open, 5) / ((low - high) * power(close, 5))",
        "Alpha012: sign(ts_delta)*(-ts_delta)": "(sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1)))",
        "Alpha033: rank(-(1-open/close))": "rank(-(1 - (open / close)))",
    }
    selected_preset = st.selectbox("快捷公式", list(preset_formulas.keys()), index=0)

    default_expr = preset_formulas[selected_preset] if selected_preset != "自定义" else \
        "ts_arg_min(sqrt(max(cs_mean(ts_av_diff(adv5, 2)), abs(adv150))), 5)"

    # ---- 表达式输入 ----
    expression = st.text_area(
        "因子表达式",
        value=st.session_state.get("bt_expression", default_expr),
        height=100,
        help="支持 ts_*/rank/scale/if_else/power 等算子，字段: open/high/low/close/volume/vwap/returns/adv5~adv180",
    )

    # ---- 参数配置 ----
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        instruments = st.selectbox("股票池", ["csi300", "csi500", "all"], index=0)
    with col_cfg2:
        data_months = st.selectbox("IC 评估月数", [1, 3, 6, 12], index=1)
    with col_cfg3:
        multi_period = st.checkbox("多周期对比（1/3/6/12月）", value=False)

    # ---- 开始回测 ----
    if st.button("🚀 开始回测", type="primary"):
        if not expression.strip():
            st.error("请输入因子表达式")
            return

        st.session_state["bt_expression"] = expression

        try:
            from factor_library.backtest_engine import (
                run_backtest, _load_data, _compute_returns, _compute_ic_dates,
                compute_group_returns,
            )

            if multi_period:
                # ---- 多周期对比 ----
                st.subheader("多周期对比结果")
                periods = [1, 3, 6, 12]
                multi_results = []
                progress = st.progress(0)

                for pi, m in enumerate(periods):
                    with st.spinner(f"回测 {m} 个月周期 ({pi+1}/{len(periods)})..."):
                        r = run_backtest(expression, instruments=instruments, data_months=m)
                    if "error" not in r:
                        multi_results.append({
                            "评估月数": m,
                            "IC均值": r.get("ic_mean", 0),
                            "ICIR": r.get("icir", 0),
                            "Rank IC": r.get("rank_ic", 0),
                            "Rank ICIR": r.get("rank_icir", 0),
                            "IC胜率": r.get("ic_win_rate", 0),
                            "多空收益": r.get("long_short_return", 0),
                            "评估区间": f"{r.get('test_start_date', '')} ~ {r.get('test_end_date', '')}",
                        })
                    else:
                        multi_results.append({
                            "评估月数": m,
                            "IC均值": "失败",
                            "ICIR": "失败",
                            "Rank IC": "失败",
                            "Rank ICIR": "失败",
                            "IC胜率": "失败",
                            "多空收益": "失败",
                            "评估区间": "-",
                        })
                    progress.progress((pi + 1) / len(periods))

                if multi_results:
                    multi_df = pd.DataFrame(multi_results)
                    st.dataframe(multi_df, use_container_width=True, hide_index=True)

                    # 柱状图对比
                    valid_rows = [r for r in multi_results if isinstance(r["ICIR"], (int, float))]
                    if valid_rows:
                        chart_data = pd.DataFrame({
                            "月数": [r["评估月数"] for r in valid_rows],
                            "ICIR": [r["ICIR"] for r in valid_rows],
                            "RankICIR": [r["Rank ICIR"] for r in valid_rows],
                        }).set_index("月数")
                        st.bar_chart(chart_data)

            else:
                # ---- 单周期回测 ----
                with st.spinner("正在回测，请耐心等待（约20~40秒）..."):
                    result = run_backtest(
                        expression, instruments=instruments, data_months=data_months,
                    )

                if "error" in result:
                    st.error(f"回测失败: {result['error']}")
                    st.info("提示: 请检查表达式语法和字段名是否正确。支持的字段: open/high/low/close/volume/vwap/returns/adv5~adv180")
                    return

                # ---- 指标卡片 ----
                st.subheader("核心指标")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1:
                    st.metric("IC均值", f"{result.get('ic_mean', 0):+.4f}")
                with mc2:
                    st.metric("ICIR", f"{result.get('icir', 0):+.4f}")
                with mc3:
                    st.metric("IC胜率", f"{result.get('ic_win_rate', 0):.2%}")
                with mc4:
                    st.metric("Rank IC", f"{result.get('rank_ic', 0):+.4f}")
                with mc5:
                    st.metric("Rank ICIR", f"{result.get('rank_icir', 0):+.4f}")

                mc6, mc7, mc8 = st.columns(3)
                with mc6:
                    st.metric("多空收益", f"{result.get('long_short_return', 0):+.4f}")
                with mc7:
                    st.metric("第一组超额", f"{result.get('top_group_excess', 0):+.4f}")
                with mc8:
                    st.metric("评估区间", f"{result.get('test_start_date', '')} ~ {result.get('test_end_date', '')}")

                # ---- 分组收益柱状图 ----
                group_returns = result.get("group_returns", {})
                if group_returns:
                    st.subheader("分组收益")
                    group_only = {k: v for k, v in group_returns.items()
                                  if k.startswith("g") and k[1:].isdigit()}
                    if group_only:
                        gr_df = pd.DataFrame(
                            list(group_only.items()),
                            columns=["分组", "日均收益"],
                        )
                        st.bar_chart(gr_df.set_index("分组"))

                # ---- 逐日 IC 序列 + IC 分布 ----
                st.subheader("IC 序列与分布")
                try:
                    from clean.data_manager import init_qlib, load_ohlcv
                    from clean.alpha_engine import AlphaEngine
                    from clean.ic_analyzer import calc_ic_series

                    with st.spinner("计算逐日 IC 序列..."):
                        df = _load_data(instruments=instruments)
                        engine = AlphaEngine(df)
                        factor = engine.calculate(expression)
                        returns = _compute_returns(df)
                        start_date = result.get("test_start_date")
                        end_date = result.get("test_end_date")

                        pearson_ic = calc_ic_series(
                            factor, returns, method="pearson", min_stocks=10,
                            start_date=start_date, end_date=end_date,
                        )
                        rank_ic_s = calc_ic_series(
                            factor, returns, method="spearman", min_stocks=10,
                            start_date=start_date, end_date=end_date,
                        )

                    ic_col1, ic_col2 = st.columns(2)

                    with ic_col1:
                        st.markdown("**Pearson IC 逐日序列**")
                        if len(pearson_ic) > 0:
                            ic_df = pd.DataFrame({
                                "Pearson IC": pearson_ic.values,
                            }, index=pearson_ic.index)
                            st.line_chart(ic_df)
                            # IC 滚动均值
                            if len(pearson_ic) >= 20:
                                ic_df_rolling = pd.DataFrame({
                                    "Pearson IC": pearson_ic.values,
                                    "IC_20日均线": pearson_ic.rolling(20, min_periods=5).mean().values,
                                }, index=pearson_ic.index)
                                st.line_chart(ic_df_rolling)
                        else:
                            st.info("IC 序列为空")

                    with ic_col2:
                        st.markdown("**IC 分布直方图**")
                        if len(pearson_ic) > 0:
                            bins = np.linspace(pearson_ic.min() - 0.01, pearson_ic.max() + 0.01, 30)
                            hist = np.histogram(pearson_ic.values, bins=bins)
                            hist_df = pd.DataFrame({
                                "IC区间": [(hist[1][i] + hist[1][i+1]) / 2 for i in range(len(hist[0]))],
                                "频次": hist[0],
                            })
                            st.bar_chart(hist_df.set_index("IC区间"))
                            st.markdown(
                                f"IC 均值={pearson_ic.mean():+.4f}, "
                                f"标准差={pearson_ic.std():.4f}, "
                                f"正占比={((pearson_ic > 0).mean()):.1%}"
                            )
                        else:
                            st.info("IC 序列为空")

                except Exception as e:
                    st.warning(f"IC 序列计算失败: {e}")

                # ---- 入库操作 ----
                st.markdown("---")
                st.subheader("入库操作")
                ingest_col1, ingest_col2 = st.columns([1, 2])

                with ingest_col1:
                    if st.button("📥 入库此因子"):
                        with st.spinner("判断入库条件..."):
                            from factor_library.database import should_auto_ingest as _should
                            if _should(result):
                                fid = add_factor(
                                    expression=expression,
                                    metrics=result,
                                    tags="在线回测,手动入库",
                                    asset_universe=instruments,
                                    test_start_date=result.get("test_start_date", ""),
                                    test_end_date=result.get("test_end_date", ""),
                                    group_returns=json.dumps(group_returns, ensure_ascii=False) if group_returns else "",
                                )
                                st.success(f"入库成功! factor_id = `{fid}`")
                            else:
                                st.warning("未满足入库阈值")

                with ingest_col2:
                    from factor_library.database import should_auto_ingest as _check
                    if _check(result):
                        st.info("✅ 该因子满足入库阈值")
                    else:
                        icir_val = abs(result.get("icir", 0))
                        ic_val = abs(result.get("ic_mean", 0))
                        wr_val = result.get("ic_win_rate", 0)
                        reasons = []
                        if icir_val <= THRESHOLD.icir:
                            reasons.append(f"|ICIR|={icir_val:.4f} ≤ {THRESHOLD.icir}")
                        if ic_val <= THRESHOLD.ic_mean:
                            reasons.append(f"|IC均值|={ic_val:.4f} ≤ {THRESHOLD.ic_mean}")
                        if wr_val <= THRESHOLD.ic_win_rate:
                            reasons.append(f"IC胜率={wr_val:.2%} ≤ {THRESHOLD.ic_win_rate:.0%}")
                        st.warning("❌ 未满足入库阈值: " + "；".join(reasons))

        except Exception as e:
            st.error(f"回测异常: {e}")
            import traceback
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())


# ============================================================
# 路由
# ============================================================
if page == "📋 因子列表":
    page_factor_list()
elif page == "🔍 因子详情":
    page_factor_detail()
elif page == "⚖️ 因子对比":
    page_factor_compare()
elif page == "📊 改进对比报告":
    page_improved_report()
elif page == "📥 自动入库记录":
    page_auto_ingest()
elif page == "💾 导入/导出":
    page_import_export()
elif page == "🚀 一键模型训练":
    page_model_training()
elif page == "🧪 在线回测":
    page_online_backtest()
