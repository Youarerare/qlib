"""
SQLite 数据库操作模块 - 因子库 CRUD
"""
import hashlib
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd

from .config import DB_PATH, THRESHOLD

logger = logging.getLogger(__name__)

# 建表 SQL
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS factors (
    factor_id       TEXT PRIMARY KEY,
    expression      TEXT NOT NULL UNIQUE,
    description     TEXT DEFAULT '',
    ic_mean         REAL DEFAULT 0,
    icir            REAL DEFAULT 0,
    ic_win_rate     REAL DEFAULT 0,
    rank_ic         REAL DEFAULT 0,
    rank_icir       REAL DEFAULT 0,
    long_short_return  REAL DEFAULT 0,
    top_group_excess   REAL DEFAULT 0,
    turnover        REAL DEFAULT 0,
    max_drawdown    REAL DEFAULT 0,
    group_returns   TEXT DEFAULT '',        -- JSON: 分组收益 {"g1":0.05,"g2":0.03,...}
    net_values      TEXT DEFAULT '',        -- JSON: 累计净值序列（可选，格式待定）
    test_start_date TEXT DEFAULT '',
    test_end_date   TEXT DEFAULT '',
    asset_universe  TEXT DEFAULT 'csi300',
    is_best         INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags            TEXT DEFAULT ''
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_expression ON factors(expression);
CREATE INDEX IF NOT EXISTS idx_icir ON factors(icir);
CREATE INDEX IF NOT EXISTS idx_tags ON factors(tags);
CREATE INDEX IF NOT EXISTS idx_is_best ON factors(is_best);
CREATE INDEX IF NOT EXISTS idx_created_at ON factors(created_at);
"""


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化数据库（建表 + 索引）"""
    conn = _get_conn()
    try:
        conn.executescript(_CREATE_TABLE_SQL)
        conn.executescript(_CREATE_INDEX_SQL)
        conn.commit()
        logger.info(f"数据库初始化完成: {DB_PATH}")
    finally:
        conn.close()


def _make_factor_id(expression: str) -> str:
    """根据表达式生成唯一 factor_id（哈希前8位 + 可读前缀）"""
    h = hashlib.md5(expression.encode("utf-8")).hexdigest()[:8]
    return f"factor_{h}"


def add_factor(expression: str, metrics: Dict[str, Any],
               description: str = "", tags: str = "",
               asset_universe: str = "csi300",
               test_start_date: str = "", test_end_date: str = "",
               group_returns: str = "", net_values: str = "") -> str:
    """
    添加因子到库中。若表达式已存在则更新指标。

    Parameters
    ----------
    expression : str
        因子公式字符串
    metrics : dict
        回测指标字典，可包含:
        ic_mean, icir, ic_win_rate, rank_ic, rank_icir,
        long_short_return, top_group_excess, turnover, max_drawdown
    description : str
        因子描述
    tags : str
        标签，逗号分隔
    asset_universe : str
        股票池
    test_start_date, test_end_date : str
        回测日期范围
    group_returns : str
        分组收益 JSON 字符串
    net_values : str
        净值序列 JSON 字符串

    Returns
    -------
    str
        factor_id
    """
    conn = _get_conn()
    try:
        factor_id = _make_factor_id(expression)

        # 检查是否已存在
        existing = conn.execute(
            "SELECT factor_id FROM factors WHERE expression = ?",
            (expression,)
        ).fetchone()

        if existing:
            # 更新已有因子
            old_id = existing["factor_id"]
            _update_metrics(conn, old_id, metrics, group_returns, net_values,
                            test_start_date, test_end_date, asset_universe, tags)
            conn.commit()
            logger.info(f"因子已存在，更新指标: {old_id} | {expression[:60]}")
            return old_id

        # 新增因子
        conn.execute("""
            INSERT INTO factors (
                factor_id, expression, description,
                ic_mean, icir, ic_win_rate, rank_ic, rank_icir,
                long_short_return, top_group_excess, turnover, max_drawdown,
                group_returns, net_values,
                test_start_date, test_end_date, asset_universe,
                is_best, tags, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
        """, (
            factor_id, expression, description,
            metrics.get("ic_mean", metrics.get("ic", 0)),
            metrics.get("icir", 0),
            metrics.get("ic_win_rate", metrics.get("ic_positive_ratio", 0)),
            metrics.get("rank_ic", 0),
            metrics.get("rank_icir", 0),
            metrics.get("long_short_return", 0),
            metrics.get("top_group_excess", 0),
            metrics.get("turnover", 0),
            metrics.get("max_drawdown", 0),
            group_returns, net_values,
            test_start_date, test_end_date, asset_universe,
            tags, datetime.now().isoformat(), datetime.now().isoformat(),
        ))
        conn.commit()
        logger.info(f"因子入库成功: {factor_id} | {expression[:60]}")
        return factor_id
    finally:
        conn.close()


def _update_metrics(conn, factor_id: str, metrics: Dict[str, Any],
                    group_returns: str = "", net_values: str = "",
                    test_start_date: str = "", test_end_date: str = "",
                    asset_universe: str = "", tags: str = ""):
    """更新因子指标"""
    sets = [
        "ic_mean = ?", "icir = ?", "ic_win_rate = ?",
        "rank_ic = ?", "rank_icir = ?",
        "long_short_return = ?", "top_group_excess = ?",
        "turnover = ?", "max_drawdown = ?",
        "updated_at = ?",
    ]
    vals = [
        metrics.get("ic_mean", metrics.get("ic", 0)),
        metrics.get("icir", 0),
        metrics.get("ic_win_rate", metrics.get("ic_positive_ratio", 0)),
        metrics.get("rank_ic", 0),
        metrics.get("rank_icir", 0),
        metrics.get("long_short_return", 0),
        metrics.get("top_group_excess", 0),
        metrics.get("turnover", 0),
        metrics.get("max_drawdown", 0),
        datetime.now().isoformat(),
    ]
    if group_returns:
        sets.append("group_returns = ?")
        vals.append(group_returns)
    if net_values:
        sets.append("net_values = ?")
        vals.append(net_values)
    if test_start_date:
        sets.append("test_start_date = ?")
        vals.append(test_start_date)
    if test_end_date:
        sets.append("test_end_date = ?")
        vals.append(test_end_date)
    if asset_universe:
        sets.append("asset_universe = ?")
        vals.append(asset_universe)
    if tags:
        sets.append("tags = ?")
        vals.append(tags)

    vals.append(factor_id)
    conn.execute(
        f"UPDATE factors SET {', '.join(sets)} WHERE factor_id = ?",
        vals
    )


def get_all_factors(sort_by: str = "icir", ascending: bool = False,
                    filter_by: Optional[Dict] = None) -> pd.DataFrame:
    """
    获取所有因子，支持排序和筛选

    Parameters
    ----------
    sort_by : str
        排序字段
    ascending : bool
        是否升序
    filter_by : dict, optional
        筛选条件，如 {"tags": "改进因子", "is_best": 1, "icir_min": 0.3}

    Returns
    -------
    pd.DataFrame
    """
    conn = _get_conn()
    try:
        where_clauses = []
        params = []

        if filter_by:
            if "tags" in filter_by:
                where_clauses.append("tags LIKE ?")
                params.append(f"%{filter_by['tags']}%")
            if "is_best" in filter_by:
                where_clauses.append("is_best = ?")
                params.append(int(filter_by["is_best"]))
            if "icir_min" in filter_by:
                where_clauses.append("icir >= ?")
                params.append(float(filter_by["icir_min"]))
            if "ic_mean_min" in filter_by:
                where_clauses.append("ABS(ic_mean) >= ?")
                params.append(float(filter_by["ic_mean_min"]))
            if "date_from" in filter_by:
                where_clauses.append("created_at >= ?")
                params.append(filter_by["date_from"])
            if "date_to" in filter_by:
                where_clauses.append("created_at <= ?")
                params.append(filter_by["date_to"])
            if "asset_universe" in filter_by:
                where_clauses.append("asset_universe = ?")
                params.append(filter_by["asset_universe"])

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # 允许的排序字段白名单，防注入
        allowed_sort = {
            "factor_id", "expression", "ic_mean", "icir", "ic_win_rate",
            "rank_ic", "rank_icir", "long_short_return", "top_group_excess",
            "turnover", "max_drawdown", "is_best", "created_at", "updated_at", "tags"
        }
        if sort_by not in allowed_sort:
            sort_by = "icir"

        order = "ASC" if ascending else "DESC"
        sql = f"SELECT * FROM factors {where_sql} ORDER BY {sort_by} {order}"

        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        conn.close()


def get_factor_by_id(factor_id: str) -> Optional[Dict]:
    """根据 factor_id 获取因子详情"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM factors WHERE factor_id = ?", (factor_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_factor_by_expression(expression: str) -> Optional[Dict]:
    """根据表达式获取因子"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM factors WHERE expression = ?", (expression,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def exists(expression: str) -> bool:
    """检查因子是否已存在"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM factors WHERE expression = ?", (expression,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def update_factor(factor_id: str, metrics: Optional[Dict] = None,
                  description: Optional[str] = None,
                  tags: Optional[str] = None,
                  is_best: Optional[int] = None):
    """更新因子信息"""
    conn = _get_conn()
    try:
        if metrics:
            _update_metrics(conn, factor_id, metrics)
        if description is not None:
            conn.execute("UPDATE factors SET description = ? WHERE factor_id = ?",
                         (description, factor_id))
        if tags is not None:
            conn.execute("UPDATE factors SET tags = ? WHERE factor_id = ?",
                         (tags, factor_id))
        if is_best is not None:
            conn.execute("UPDATE factors SET is_best = ? WHERE factor_id = ?",
                         (is_best, factor_id))
        conn.execute("UPDATE factors SET updated_at = ? WHERE factor_id = ?",
                     (datetime.now().isoformat(), factor_id))
        conn.commit()
    finally:
        conn.close()


def delete_factor(factor_id: str) -> bool:
    """删除因子"""
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM factors WHERE factor_id = ?", (factor_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def set_best_factor(factor_id: str):
    """设置某因子为当前最优（取消其他因子的 is_best 标记）"""
    conn = _get_conn()
    try:
        conn.execute("UPDATE factors SET is_best = 0 WHERE is_best = 1")
        conn.execute("UPDATE factors SET is_best = 1, updated_at = ? WHERE factor_id = ?",
                     (datetime.now().isoformat(), factor_id))
        conn.commit()
        logger.info(f"已设置最优因子: {factor_id}")
    finally:
        conn.close()


def get_recent_factors(limit: int = 20) -> pd.DataFrame:
    """获取最近入库的因子"""
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM factors ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,)
        )
        return df
    finally:
        conn.close()


def get_auto_ingested_factors(limit: int = 50) -> pd.DataFrame:
    """获取自动入库的因子（标签含"自动入库"）"""
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM factors WHERE tags LIKE '%自动入库%' ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,)
        )
        return df
    finally:
        conn.close()


def export_to_csv(filepath: Optional[str] = None) -> str:
    """导出因子库为 CSV"""
    from .config import EXPORT_DIR
    if filepath is None:
        filepath = str(EXPORT_DIR / f"factor_library_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df = get_all_factors()
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info(f"因子库已导出: {filepath}")
    return filepath


def export_to_excel(filepath: Optional[str] = None) -> str:
    """导出因子库为 Excel"""
    from .config import EXPORT_DIR
    if filepath is None:
        filepath = str(EXPORT_DIR / f"factor_library_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    df = get_all_factors()
    try:
        df.to_excel(filepath, index=False, engine="openpyxl")
    except ImportError:
        df.to_excel(filepath, index=False, engine="xlsxwriter")
    logger.info(f"因子库已导出: {filepath}")
    return filepath


def import_from_csv(filepath: str) -> int:
    """从 CSV 导入因子"""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    count = 0
    for _, row in df.iterrows():
        expression = row.get("expression", "")
        if not expression:
            continue
        metrics = {
            "ic_mean": row.get("ic_mean", 0),
            "icir": row.get("icir", 0),
            "ic_win_rate": row.get("ic_win_rate", 0),
            "rank_ic": row.get("rank_ic", 0),
            "rank_icir": row.get("rank_icir", 0),
            "long_short_return": row.get("long_short_return", 0),
            "top_group_excess": row.get("top_group_excess", 0),
        }
        add_factor(
            expression=expression,
            metrics=metrics,
            description=row.get("description", ""),
            tags=row.get("tags", ""),
            asset_universe=row.get("asset_universe", "csi300"),
            test_start_date=str(row.get("test_start_date", "")),
            test_end_date=str(row.get("test_end_date", "")),
        )
        count += 1
    logger.info(f"CSV导入完成: {count} 条因子")
    return count


def should_auto_ingest(metrics: Dict[str, Any], threshold=None) -> bool:
    """
    判断因子是否满足自动入库条件

    Parameters
    ----------
    metrics : dict
        回测指标
    threshold : IngestThreshold, optional
        入库阈值配置，默认使用全局配置

    Returns
    -------
    bool
    """
    th = threshold or THRESHOLD
    ic_mean = abs(metrics.get("ic_mean", metrics.get("ic", 0)))
    icir = abs(metrics.get("icir", 0))
    ic_win = metrics.get("ic_win_rate", metrics.get("ic_positive_ratio", 0))

    return icir > th.icir and ic_mean > th.ic_mean and ic_win > th.ic_win_rate


def get_factors_for_training(filter_by: Optional[Dict] = None, limit: int = 50) -> List[str]:
    """
    获取适合模型训练的因子表达式列表（用于一键训练）

    Parameters
    ----------
    filter_by : dict, optional
        筛选条件
    limit : int
        最多返回因子数

    Returns
    -------
    list[str]
        因子表达式列表
    """
    df = get_all_factors(sort_by="icir", ascending=False, filter_by=filter_by)
    if df.empty:
        return []
    return df["expression"].head(limit).tolist()


# 模块加载时自动初始化数据库
try:
    init_db()
except Exception as e:
    logger.warning(f"数据库自动初始化失败: {e}")
