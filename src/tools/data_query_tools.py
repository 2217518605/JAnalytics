"""数据查询与分析工具 - 供 Agent 调用。核心分析委托给 ecommerce.analysis 模块。"""

import os
import json
import re
import warnings
import chardet
import pandas as pd
from typing import Optional
from langchain.tools import tool

warnings.filterwarnings("ignore", message=".*mixed types.*")

from utils.paths import PROJECT_ROOT

# 导入统一分析模块
try:
    from ecommerce.analysis import (
        analyze_csv_text, analyze_csv_structured, analyze_comments_text,
        get_uploaded_files, find_file_by_month, detect_columns, FIELD_MAP
    )
    _HAS_ANALYSIS_MODULE = True
except ImportError:
    _HAS_ANALYSIS_MODULE = False


def _find_file(file_name: str) -> Optional[str]:
    """在 uploads 和 comments 目录查找文件"""
    for subdir in ("assets/uploads", "assets/comments"):
        dir_path = os.path.join(PROJECT_ROOT, subdir)
        if not os.path.isdir(dir_path):
            continue
        for fn in os.listdir(dir_path):
            if fn.lower() == file_name.lower():
                return os.path.join(dir_path, fn)
            # 也支持不带路径的模糊匹配
            if file_name.lower() in fn.lower():
                return os.path.join(dir_path, fn)
    return None


def _read_file_to_df(fp: str) -> Optional[pd.DataFrame]:
    """将 CSV/Excel 文件读为 DataFrame"""
    ext = fp.rsplit(".", 1)[-1].lower() if "." in fp else ""
    try:
        if ext == "csv":
            with open(fp, "rb") as _f:
                _raw = _f.read(100000)
            _detect = chardet.detect(_raw)
            _enc = _detect.get("encoding", "utf-8") or "utf-8"
            _enc = _enc.lower().replace("-", "")
            _enc_map = {"utf8": "utf-8", "utf8sig": "utf-8-sig", "gb2312": "gbk",
                        "ascii": "utf-8", "gb18030": "gbk", "big5": "big5"}
            _enc = _enc_map.get(_enc, _enc)
            if _enc == "utf-8-sig":
                df = pd.read_csv(fp, encoding="utf-8-sig")
            else:
                try:
                    df = pd.read_csv(fp, encoding=_enc)
                except (UnicodeDecodeError, UnicodeError):
                    df = pd.read_csv(fp, encoding="utf-8", encoding_errors="replace")
            return df
        elif ext in ("xlsx", "xls"):
            return pd.read_excel(fp)
    except Exception:
        return None
    return None


@tool
def list_files(year: int = 0, user_id: int = 0) -> str:
    """获取已上传的文件列表，返回文件名称、行数、类型等信息。year=0 表示所有年份。user_id 按用户过滤。"""
    from storage.database.db import get_session
    from ecommerce.models import ReportFile

    db = get_session()
    try:
        q = db.query(ReportFile)
        if user_id > 0:
            q = q.filter(ReportFile.user_id == user_id)
        if year > 0:
            q = q.filter(ReportFile.data_year == year)
        files = q.order_by(ReportFile.created_at.desc()).all()
        if not files:
            return "暂无已上传的文件。"
        lines = [f"📂 共 {len(files)} 个文件："]
        for f in files:
            ftype = f.file_type or "未知"
            lines.append(f"- {f.file_name}（{ftype.upper()}，{f.row_count or 0}行，{f.report_period or '无日期'}）")
        return "\n".join(lines)
    except Exception as e:
        return f"查询文件列表失败：{e}"
    finally:
        db.close()


@tool
def read_file_data(file_name: str, page: int = 1, page_size: int = 50) -> str:
    """
    读取已上传文件的原始内容（支持CSV/XLSX），按分页返回。
    在分析数据前，先使用此工具了解文件有哪些列、前几行数据是什么样的。
    file_name: 文件名（如"11月.csv"）
    page: 页码（从1开始）
    page_size: 每页行数（最大500）
    """
    page_size = min(max(page_size, 1), 500)

    fp = _find_file(file_name)
    if not fp:
        return f"文件 '{file_name}' 不存在。请先用 list_files 查看可用的文件列表。"

    df = _read_file_to_df(fp)
    if df is None:
        return f"无法读取文件 '{file_name}'。"

    total = len(df)
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    lines = [f"📄 {file_name} | 共 {total} 行 | 第 {page} 页（{start+1}-{end}行）"]
    lines.append(f"📋 列名：{' | '.join(str(c) for c in df.columns)}")
    lines.append("")

    # 表头
    lines.append(" | ".join(str(c) for c in df.columns))
    lines.append("-" * 80)

    # 数据行
    for idx in range(start, end):
        row = df.iloc[idx]
        vals = [str(v)[:80] if pd.notna(v) else "" for v in row]
        lines.append(" | ".join(vals))

    lines.append(f"\n📌 提示：共 {total} 行，当前第 {page} 页。可用 read_file_data(file_name='{file_name}', page={page+1}) 查看下一页。")
    return "\n".join(lines)


@tool
def analyze_file_by_column(file_name: str, group_by_columns: str, agg_columns: str = "") -> str:
    """
    对已上传的文件按指定列分组聚合分析。
    例如按"主播"分组统计销量、销售额，就能知道每个主播的业绩。

    file_name: 文件名（如"11月.csv"）
    group_by_columns: 分组列名，多个列用逗号分隔（如"主播"或"类目,颜色"）
    agg_columns: 要聚合的数值列名，多个列用逗号分隔。留空则自动识别所有数值列。
                 常用列：销量,销售额,成本,利润,退货数,库存
                 或原文件中的：销量(扣退),销售额(扣退),付款金额,实发件数,毛利额

    返回按分组列聚合后的结果，包含每组的总数、平均值等统计。
    """
    fp = _find_file(file_name)
    if not fp:
        return f"文件 '{file_name}' 不存在。"

    df = _read_file_to_df(fp)
    if df is None:
        return f"无法读取文件 '{file_name}'。"

    # 清理列名：去除多余空格，转为字符串
    df.columns = [str(c).strip() for c in df.columns]

    # 解析分组列
    group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip()]
    missing_cols = [c for c in group_cols if c not in df.columns]
    if missing_cols:
        return f"找不到分组列：{missing_cols}。文件中的列名：{'、'.join(df.columns)}"

    # 自动检测数值列
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if agg_columns:
        user_cols = [c.strip() for c in agg_columns.split(",") if c.strip()]
        user_numeric = [c for c in user_cols if c in numeric_cols]
        if not user_numeric:
            # 可能是列名有空格或特殊字符，尝试精确匹配
            user_numeric = [c for c in user_cols if c in df.columns]
            # 再尝试清除后匹配
            if not user_numeric:
                df_clean = df.rename(columns=lambda x: x.strip())
                user_numeric = [c for c in user_cols if c in df_clean.columns]
                if user_numeric:
                    df = df_clean
                    group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip()]
        agg_cols = user_numeric or numeric_cols
    else:
        agg_cols = numeric_cols

    if not agg_cols:
        return f"未找到可聚合的数值列。文件中的列名：{'、'.join(df.columns)}"

    # 按列分组聚合
    try:
        grouped = df.groupby(group_cols, dropna=False)[agg_cols].agg(["sum", "mean", "count"]).fillna(0)
    except Exception as e:
        return f"分组聚合失败：{e}"

    lines = [
        f"📊 文件 '{file_name}' 按 {', '.join(group_cols)} 分析结果",
        f"📋 聚合列：{', '.join(agg_cols)}",
        f"🔢 数据总行数：{len(df)}，分组数：{len(grouped)}",
        ""
    ]

    # 格式化输出每个分组
    for group_vals, row in grouped.iterrows():
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        label = " | ".join(str(v) for v in group_vals)
        lines.append(f"▶ {label}")
        for col in agg_cols:
            try:
                s = row[(col, "sum")]
                m = row[(col, "mean")]
                c = row[(col, "count")]
                lines.append(f"   {col}: 总计={s:.2f}, 均值={m:.2f}, 出现次数={int(c)}")
            except (KeyError, TypeError):
                pass
        lines.append("")

    lines.append("💡 如需查看某一组的详细数据，可用 read_file_data 逐页浏览。")
    return "\n".join(lines)


@tool
def analyze_data_deep(file_name: str = "") -> str:
    """
    **核心分析工具**：对上传的 CSV 文件做全维度深度聚合分析，一次性返回所有维度的数据。
    适合回答"分析数据"、"整体情况"、"表现怎么样"等开放性问题。

    返回维度包括：
    - 总览：总销量/销售额/利润/利润率/退款率/退款金额
    - 费用结构：账单费/快递费/推广费/佣金等
    - 主播排行 TOP15：每个主播的销量/销售额/利润率/退款率
    - SKU 集中度：TOP5/10/20/50 占比
    - 品牌分布、价格带分析、类目分布

    file_name: 留空则分析最新上传的文件，或指定文件名如"9月.csv"
    """
    if not _HAS_ANALYSIS_MODULE:
        return "分析模块未加载，请使用 analyze_file_by_column 进行手动分析。"

    files = get_uploaded_files("uploads")
    if file_name:
        files = [f for f in files if os.path.basename(f) == file_name or file_name in os.path.basename(f)]
    if not files:
        comment_files = get_uploaded_files("comments")
        if file_name:
            comment_files = [f for f in comment_files if file_name in os.path.basename(f)]
        if comment_files:
            return analyze_comments_text(comment_files[0])
        return "未找到可分析的数据文件。请先用 list_files 查看可用文件。"

    return analyze_csv_text(files[0])


@tool
def query_sales_data(year: int = 0, month: int = 0, report_name: str = "", limit: int = 50, user_id: int = 0) -> str:
    """
    从数据库查询销售明细数据（SalesData表）。
    year=0 查所有年份，month=0 查所有月份。
    report_name 可按文件名筛选。
    user_id 按用户过滤（0=管理员，查看全部）。
    此表包含以下字段：sku编码、商品名称、类目、版型、颜色、尺码、销量、销售额、成本、利润、利润率、退货数、退货率、库存。
    注意：数据库中可能不包含原始CSV中的"主播"字段，如需按主播分析请使用 analyze_file_by_column。
    """
    from storage.database.db import get_session
    from ecommerce.models import SalesData

    db = get_session()
    try:
        q = db.query(SalesData)
        if user_id > 0:
            q = q.filter(SalesData.user_id == user_id)
        if year > 0:
            q = q.filter(SalesData.data_year == year)
        if month > 0:
            q = q.filter(SalesData.data_month == month)
        if report_name:
            q = q.filter(SalesData.report_name == report_name)
        records = q.order_by(SalesData.id).limit(limit).all()

        if not records:
            return f"未查询到数据。year={year}, month={month}, report_name='{report_name}'"

        lines = [f"📊 销售明细数据（共 {len(records)} 条）："]
        for r in records:
            lines.append(
                f"  [{r.id}] {r.product_name or '-'} | "
                f"SKU:{r.sku_code or '-'} | "
                f"{r.category or '-'}/{r.style or '-'}/{r.color or '-'}/{r.size or '-'} | "
                f"销量:{r.sales_volume} | "
                f"销售额:{r.sales_amount:.2f} | "
                f"成本:{r.cost:.2f} | "
                f"利润:{r.profit:.2f} | "
                f"退货:{r.return_count}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"查询失败：{e}"
    finally:
        db.close()


def analyze_file_data_direct(file_path: str, group_by_columns: str = "主播",
                              agg_columns: str = "商品销售数据-商品销售数量(扣退),商品销售数据-商品销售金额(扣退),商品销售数据-商品销售成本(扣退),利润-毛利额,利润-经营利润"):
    """
    直接从指定路径的 CSV 文件读取并分组聚合（非 @tool 版本，可被内部调用）。
    返回 pandas DataFrame 或 None。
    """
    try:
        df = _read_file_to_df(file_path)
        if df is None or df.empty:
            return None

        group_cols = [c.strip() for c in group_by_columns.split(",")]
        agg_col_names = [c.strip() for c in agg_columns.split(",")]

        # 只取存在的列
        valid_group = [c for c in group_cols if c in df.columns]
        valid_agg = [c for c in agg_col_names if c in df.columns]

        if not valid_group or not valid_agg:
            # 尝试自动检测列名
            hint = f"可用列: {list(df.columns)[:30]}"
            print(f"[analyze_file_data_direct] columns not found. {hint}")
            # 如果 group 或 agg 列不存在，返回前几行做预览
            return df.head(10)

        result = df.groupby(valid_group)[valid_agg].sum().fillna(0).sort_values(valid_agg[0], ascending=False)
        return result
    except Exception as e:
        print(f"[analyze_file_data_direct] error: {e}")
        return None