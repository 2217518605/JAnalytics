"""统一数据分析模块 — 所有 CSV 聚合分析和 Dashboard 计算的唯一入口

替代原来散落在 main.py / ecommerce/app.py 中的:
  - _get_sales_summary()
  - get_sales_summary_with_extra()
  - _get_sku_from_csv()
  - _get_sku_enhanced()
  - _get_zhubo_markdown_table()
  - _calc_dashboard_from_csv()
  - _deep_analyze_csv()
"""

import os
import re
import glob
import logging
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ============================================================
# 通用工具函数
# ============================================================

def _parse_sku_display(sku_code: str):
    """解析 SKU 编码为展示格式。

    将 "522003C丹宁色28" 解析为:
      display: "522003C 丹宁色 28"  (用于 SKU 编码列)
      model:   "522003C"             (款号)

    若无法解析，退回原字符串作为 display 和 model。
    """
    code = sku_code.strip()
    if not code:
        return code, code
    # 匹配: 字母数字款号 + 中文颜色 + 数字尺码
    m = re.match(r'^([A-Za-z0-9]+)([一-鿿]+)(\d+)$', code)
    if m:
        model, color, size = m.group(1), m.group(2), m.group(3)
        return f"{model} {color} {size}", model
    # Fallback: 尝试把末尾数字分离出来
    m2 = re.match(r'^(.+?)(\d+)$', code)
    if m2:
        return f"{m2.group(1)} {m2.group(2)}", m2.group(1)
    return code, code


# ============================================================
# 通用 CSV 读取 + 列名检测
# ============================================================

# 全量列名映射: 逻辑名 → CSV 候选列名列表（按优先级排列）
FIELD_MAP: Dict[str, List[str]] = {
    "vol_koutui":        ["商品销售数据-商品销售数量(扣退)", "利润-销售数量(扣退)", "销售数量(扣退)"],
    "amt_koutui":        ["商品销售数据-商品销售金额(扣退)", "利润-销售金额(扣退)", "销售金额(扣退)"],
    "cost_koutui":       ["商品销售数据-商品销售成本(扣退)", "利润-销售成本(扣退)", "销售成本(扣退)"],
    "ship_qty":          ["商品数据-实发数量", "实发数量"],
    "ship_amt":          ["商品数据-实发金额", "实发金额"],
    "ship_cost":         ["商品数据-实发成本", "实发成本"],
    "gross_profit":      ["利润-毛利额", "毛利额"],
    "oper_profit":       ["利润-经营利润", "经营利润"],
    "total_fee":         ["利润-费用", "费用合计"],
    "promo_fee":         ["利润-其中:推广费", "推广费"],
    "refund_qty":        ["售后合计-退款数量合计", "退款数量合计", "退款数量"],
    "refund_amt":        ["售后合计-退款金额合计", "退款金额合计", "退款金额"],
    "zhubo":             ["主播", "主播名称", "达人", "达人名称"],
    "sku_code":          ["商品编码", "商品款号"],
    "sku_name":          ["商品简称", "商品名称", "【商品资料】：商品简称"],
    "brand":             ["品牌"],
    "category":          ["商品类目", "分类"],
    "shop_name":         ["店铺名称"],
    "supplier":          ["供应商名称"],
    "express_fee":       ["订单费用-快递费（自动匹配）", "快递费"],
    "pack_fee":          ["订单费用-包材费（自动匹配）", "包材费"],
    "bill_fee":          ["订单费用-账单费用", "账单费用"],
    "commission":        ["线上预估达人佣金", "60010502达人佣金", "达人佣金"],
    "order_qty":         ["商品数据-商品数量"],
    "order_amt":         ["商品数据-商品金额"],
    "order_cost":        ["商品数据-商品成本"],
    "pay_amt":           ["商品数据-付款金额", "付款金额"],
    "source":            ["成交端", "来源", "渠道", "流量来源", "成交来源"],
}


def _read_csv_safe(file_path: str) -> Optional[pd.DataFrame]:
    """安全读取 CSV/Excel（自动检测编码），失败返回 None"""
    ext = os.path.splitext(file_path)[1].lower()
    # Excel 文件直接用 pandas 读取
    if ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(file_path)
            if len(df) > 0:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            return None
    # CSV 文件自动检测编码
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            if len(df) > 0:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """在 DataFrame 中自动检测标准字段对应的实际列名。

    返回 {逻辑名: 实际列名}，找不到的逻辑名不会出现在结果中。
    """
    detected = {}
    for logical, candidates in FIELD_MAP.items():
        for c in candidates:
            if c in df.columns:
                detected[logical] = c
                break
    return detected


def get_uploaded_files(subdir: str = "uploads", user_id: int = 0) -> List[str]:
    """获取用户专属目录下的所有数据文件路径"""
    d = os.path.join(PROJECT_ROOT, "assets", subdir, str(user_id)) if user_id else os.path.join(PROJECT_ROOT, "assets", subdir)
    if not os.path.isdir(d):
        return []
    files = (
        glob.glob(os.path.join(d, "*.csv")) +
        glob.glob(os.path.join(d, "*.xlsx")) +
        glob.glob(os.path.join(d, "*.xls"))
    )
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def get_all_uploaded_files(user_id: int = 0) -> List[Dict[str, str]]:
    """获取用户专属目录下的所有文件"""
    results = []
    for subdir in ("uploads", "comments"):
        d = os.path.join(PROJECT_ROOT, "assets", subdir, str(user_id)) if user_id else os.path.join(PROJECT_ROOT, "assets", subdir)
        if not os.path.isdir(d):
            continue
        for ext in ("*.csv", "*.xlsx", "*.xls"):
            for fp in glob.glob(os.path.join(d, ext)):
                fname = os.path.basename(fp)
                if any(r["file_name"] == fname for r in results):
                    continue
                results.append({"file_name": fname, "file_path": fp, "source": subdir})
    results.sort(key=lambda x: os.path.getmtime(x["file_path"]), reverse=True)
    return results


def find_file_by_month(month: int, subdir: str = "uploads") -> Optional[str]:
    """根据月份数字查找对应的 CSV 文件"""
    for fp in get_uploaded_files(subdir):
        fname = os.path.basename(fp)
        if re.search(rf'(?<!\d){month}\s*月', fname):
            return fp
    return None


def safe_sum(series: pd.Series) -> float:
    """安全求和：处理混合类型、NaN、空字符串"""
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


# ============================================================
# 核心：单文件深度分析 → 结构化 dict
# ============================================================

def analyze_csv_structured(file_path: str) -> Dict[str, Any]:
    """对单个 CSV 文件做全维度聚合，返回结构化 dict。

    维度：总览 / 主播 TOP N / SKU 集中度 / 品牌分布 / 价格带 / 类目 / 费用
    """
    result = {
        "file_name": os.path.basename(file_path),
        "error": None,
        "overview": {},
        "top_zhubo": [],
        "top_sku": [],
        "brands": [],
        "price_bands": [],
        "categories": [],
        "sources": [],
        "fees": {},
        "sku_count": 0,
        "zhubo_count": 0,
    }

    df = _read_csv_safe(file_path)
    if df is None or len(df) == 0:
        result["error"] = "文件为空或无法读取"
        return result

    result["row_count"] = len(df)
    cols = detect_columns(df)

    # ---------- 辅助函数 ----------
    def _sum(logical_key: str, default: float = 0.0) -> float:
        c = cols.get(logical_key)
        return safe_sum(df[c]) if c else default

    def _has(logical_key: str) -> bool:
        return cols.get(logical_key) in df.columns if cols.get(logical_key) else False

    # ---------- 总览 ----------
    total_vol = _sum("vol_koutui")
    total_amt = _sum("amt_koutui")
    total_cost = _sum("cost_koutui")
    total_ship_qty = _sum("ship_qty")
    total_ship_amt = _sum("ship_amt")
    gross_profit = _sum("gross_profit")
    oper_profit = _sum("oper_profit")
    total_fee = _sum("total_fee")
    promo_fee = _sum("promo_fee")
    refund_qty = _sum("refund_qty")
    refund_amt = _sum("refund_amt")
    express_fee = _sum("express_fee")
    bill_fee = _sum("bill_fee")
    pack_fee = _sum("pack_fee")
    commission = _sum("commission")

    gross_margin = (total_amt - total_cost) / total_amt * 100 if total_amt > 0 else 0
    oper_margin = oper_profit / total_amt * 100 if total_amt > 0 else 0
    # 修正退款率 = 退款量 / (销量 + 退款量)
    refund_rate = refund_qty / (total_vol + refund_qty) * 100 if (total_vol + refund_qty) > 0 else 0
    avg_price = total_amt / total_vol if total_vol > 0 else 0

    result["overview"] = {
        "total_rows": len(df),
        "koutui_vol": int(total_vol),
        "koutui_amt": round(total_amt, 2),
        "cost": round(total_cost, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_margin": round(gross_margin, 1),
        "oper_profit": round(oper_profit, 2),
        "oper_margin": round(oper_margin, 1),
        "total_fee": round(total_fee, 2),
        "ship_qty": int(total_ship_qty),
        "ship_amt": round(total_ship_amt, 2),
        "refund_qty": int(refund_qty),
        "refund_amt": round(refund_amt, 2),
        "refund_rate": round(refund_rate, 1),
        "avg_price": round(avg_price, 0),
    }

    result["fees"] = {
        "express": round(express_fee, 2),
        "bill": round(bill_fee, 2),
        "pack": round(pack_fee, 2),
        "promo": round(promo_fee, 2),
        "commission": round(commission, 2),
        "total": round(total_fee, 2),
    }

    # ---------- 主播分析 ----------
    zhubo_col = cols.get("zhubo")
    if zhubo_col and zhubo_col in df.columns:
        agg_map = {}
        for lk in ["vol_koutui", "amt_koutui", "gross_profit", "oper_profit", "ship_qty", "ship_amt", "refund_qty", "refund_amt"]:
            c = cols.get(lk)
            if c and c in df.columns:
                agg_map[c] = "sum"

        if agg_map:
            grouped = df.groupby(zhubo_col).agg(agg_map).fillna(0)
            sort_col = cols.get("amt_koutui")
            if sort_col and sort_col in grouped.columns:
                grouped = grouped.sort_values(sort_col, ascending=False)

            result["zhubo_count"] = len(grouped)
            for name, row in grouped.head(100).iterrows():
                name_clean = str(name).strip().replace('\t', ' ').replace('\n', '').replace('\r', '')
                # 去掉 "3567520 - " 这样的 ID 前缀
                name_clean = re.sub(r'^\d+\s*[-—]\s*', '', name_clean).strip()
                if not name_clean or name_clean.lower() in ('nan', 'none', ''):
                    continue
                vol = float(row[cols["vol_koutui"]]) if _has("vol_koutui") else 0
                amt = float(row[cols["amt_koutui"]]) if _has("amt_koutui") else 0
                gp = float(row[cols["gross_profit"]]) if _has("gross_profit") else 0
                op = float(row[cols["oper_profit"]]) if _has("oper_profit") else 0
                sq = float(row[cols["ship_qty"]]) if _has("ship_qty") else 0
                sa = float(row[cols["ship_amt"]]) if _has("ship_amt") else 0
                rq = float(row[cols["refund_qty"]]) if _has("refund_qty") else 0
                ra = float(row[cols["refund_amt"]]) if _has("refund_amt") else 0
                margin = (op / amt * 100) if amt > 0 else 0
                rr = (rq / (vol + rq) * 100) if (vol + rq) > 0 else 0

                result["top_zhubo"].append({
                    "zhubo": name_clean,
                    "vol": int(vol), "amt": round(amt, 2),
                    "gross_profit": round(gp, 2), "oper_profit": round(op, 2),
                    "ship_qty": int(sq), "ship_amt": round(sa, 2),
                    "refund_qty": int(rq), "refund_amt": round(ra, 2),
                    "profit_margin": round(margin, 1), "refund_rate": round(rr, 1),
                })

    # 集中度
    if result["top_zhubo"] and total_amt > 0:
        top1_share = result["top_zhubo"][0]["amt"] / total_amt * 100
        top3_share = sum(z["amt"] for z in result["top_zhubo"][:3]) / total_amt * 100
        result["overview"]["top1_zhubo_share"] = round(top1_share, 1)
        result["overview"]["top3_zhubo_share"] = round(top3_share, 1)

    # ---------- SKU 分析 ----------
    sku_col = cols.get("sku_code")
    sku_name_col = cols.get("sku_name")
    vol_col = cols.get("vol_koutui")
    amt_col = cols.get("amt_koutui")
    profit_col = cols.get("gross_profit") or cols.get("oper_profit")
    if sku_col and sku_col in df.columns and vol_col and vol_col in df.columns:
        agg_s = {vol_col: "sum"}
        if amt_col:
            agg_s[amt_col] = "sum"
        if profit_col and profit_col in df.columns:
            agg_s[profit_col] = "sum"
        sku_grouped = df.groupby(sku_col).agg(agg_s).fillna(0).sort_values(vol_col, ascending=False)
        result["sku_count"] = len(sku_grouped)

        # 预建 SKU→商品名称 的映射（取每个 SKU 第一条非空名称）
        sku_name_map = {}
        if sku_name_col and sku_name_col in df.columns:
            for _, row in df[[sku_col, sku_name_col]].dropna(subset=[sku_name_col]).iterrows():
                code = str(row[sku_col]).strip()
                if code and code not in sku_name_map:
                    nm = str(row[sku_name_col]).strip()
                    if nm and nm.lower() not in ("nan", "none", ""):
                        sku_name_map[code] = nm

        for sku, row in sku_grouped.head(20).iterrows():
            code = str(sku).strip()
            v = int(row[vol_col])
            a = float(row[amt_col]) if amt_col else 0
            p = float(row[profit_col]) if profit_col and profit_col in row.index else 0
            result["top_sku"].append({
                "product_code": code,
                "product_name": sku_name_map.get(code, ""),
                "vol": v,
                "amt": round(a, 2),
                "profit": round(p, 2),
                "share": round(v / total_vol * 100, 1) if total_vol > 0 else 0,
            })

        for top_n in [5, 10, 20, 50]:
            if len(sku_grouped) >= top_n:
                share = sku_grouped[vol_col].head(top_n).sum() / total_vol * 100 if total_vol > 0 else 0
                result["overview"][f"top{top_n}_sku_share"] = round(share, 1)

    # ---------- 品牌分布 ----------
    brand_col = cols.get("brand")
    if brand_col and brand_col in df.columns and vol_col and vol_col in df.columns:
        b_grouped = df.groupby(brand_col)[vol_col].sum().fillna(0).sort_values(ascending=False)
        for brand, qty in b_grouped.head(10).items():
            b = str(brand).strip()
            if b and b.lower() != 'nan':
                result["brands"].append({
                    "brand": b,
                    "vol": int(qty),
                    "share": round(qty / total_vol * 100, 1) if total_vol > 0 else 0,
                })

    # ---------- 价格带 ----------
    if vol_col and vol_col in df.columns and amt_col and amt_col in df.columns:
        df_valid = df[(df[vol_col] > 0) & (pd.to_numeric(df[amt_col], errors='coerce') > 0)].copy()
        if len(df_valid) > 0:
            df_valid['_price'] = pd.to_numeric(df_valid[amt_col], errors='coerce') / pd.to_numeric(df_valid[vol_col], errors='coerce')
            for label, lo, hi in [
                ("<¥100", 0, 100), ("¥100-150", 100, 150), ("¥150-200", 150, 200),
                ("¥200-300", 200, 300), ("¥300+", 300, float('inf'))
            ]:
                mask = (df_valid['_price'] >= lo) & (df_valid['_price'] < hi)
                subset = df_valid[mask]
                if len(subset) > 0:
                    q = subset[vol_col].sum()
                    a = subset[amt_col].sum()
                    result["price_bands"].append({
                        "label": label, "vol": int(q), "amt": round(a, 2),
                        "avg_price": round(a / q, 0) if q > 0 else 0,
                    })

    # ---------- 类目 ----------
    cat_col = cols.get("category")
    if cat_col and cat_col in df.columns and vol_col and vol_col in df.columns:
        c_grouped = df.groupby(cat_col)[vol_col].sum().fillna(0).sort_values(ascending=False)
        for cat, qty in c_grouped.head(10).items():
            c = str(cat).strip()
            if c and c.lower() != 'nan':
                result["categories"].append({
                    "category": c,
                    "vol": int(qty),
                    "share": round(qty / total_vol * 100, 1) if total_vol > 0 else 0,
                })

    # ---------- 来源/渠道分布 ----------
    source_col = cols.get("source")
    if source_col and source_col in df.columns and vol_col and vol_col in df.columns:
        # 按来源聚合：销量、销售额、利润
        s_agg = {vol_col: "sum"}
        if amt_col and amt_col in df.columns:
            s_agg[amt_col] = "sum"
        profit_logical = "gross_profit" if "gross_profit" in cols else "oper_profit"
        profit_col = cols.get(profit_logical)
        if profit_col and profit_col in df.columns:
            s_agg[profit_col] = "sum"

        s_grouped = df.groupby(source_col).agg(s_agg).fillna(0)
        if amt_col and amt_col in s_grouped.columns:
            s_grouped = s_grouped.sort_values(amt_col, ascending=False)

        for src, row in s_grouped.iterrows():
            s_name = str(src).strip()
            if not s_name or s_name.lower() in ('nan', 'none', ''):
                continue
            v = int(row[vol_col])
            a = float(row[amt_col]) if amt_col and amt_col in s_grouped.columns else 0.0
            p = float(row[profit_col]) if profit_col and profit_col in row.index else 0.0
            margin = round(p / a * 100, 1) if a > 0 else 0
            result["sources"].append({
                "source": s_name,
                "vol": v,
                "amt": round(a, 2),
                "profit": round(p, 2),
                "profit_margin": margin,
                "share": round(v / total_vol * 100, 1) if total_vol > 0 else 0,
            })

    return result


# ============================================================
# 文本格式输出（给 LLM）
# ============================================================

def analyze_csv_text(file_path: str) -> str:
    """对单个 CSV 文件做全维度聚合，返回 Markdown 格式的结构化摘要。"""
    r = analyze_csv_structured(file_path)
    if r.get("error"):
        return f"[CSV分析失败: {r['error']}]"

    ov = r["overview"]
    nl = chr(10)

    # ---- 总览表格 ----
    rows = [
        f"## 📊 {r['file_name']}  —  {ov['total_rows']:,} 条记录",
        "",
        "| 指标 | 数值 | 指标 | 数值 |",
        "|------|------|------|------|",
        f"| 总销量(扣退) | **{ov['koutui_vol']:,} 件** | 总销售额(扣退) | **¥{ov['koutui_amt']:,.2f}** |",
        f"| 商品成本 | ¥{ov['cost']:,.2f} ({ov['cost']/ov['koutui_amt']*100:.1f}%) | 毛利率 | **{ov['gross_margin']}%** |" if ov['koutui_amt'] > 0 else "",
        f"| 经营利润 | ¥{ov['oper_profit']:,.2f} | 经营利润率 | **{ov['oper_margin']}%** |",
        f"| 退款数量 | {ov['refund_qty']:,} 件 | **退款率** | **{ov['refund_rate']}%** |",
        f"| 退款金额 | ¥{ov['refund_amt']:,.2f} | 整体均价 | ¥{ov['avg_price']:.0f} |",
        f"| 实发数量 | {ov['ship_qty']:,} 件 | 实发金额 | ¥{ov['ship_amt']:,.2f} |",
        "",
    ]

    # ---- 费用表格 ----
    fees = r["fees"]
    fee_rows = []
    if fees["bill"] > 0: fee_rows.append(f"| 账单费用 | ¥{fees['bill']:,.2f} | {fees['bill']/ov['koutui_amt']*100:.1f}% |" if ov['koutui_amt'] > 0 else "")
    if fees["express"] > 0: fee_rows.append(f"| 快递费 | ¥{fees['express']:,.2f} | {fees['express']/ov['koutui_amt']*100:.1f}% |" if ov['koutui_amt'] > 0 else "")
    if fees["pack"] > 0: fee_rows.append(f"| 包材费 | ¥{fees['pack']:,.2f} | {fees['pack']/ov['koutui_amt']*100:.1f}% |" if ov['koutui_amt'] > 0 else "")
    if fees["promo"] > 0: fee_rows.append(f"| 推广费 | ¥{fees['promo']:,.2f} | {fees['promo']/ov['koutui_amt']*100:.1f}% |" if ov['koutui_amt'] > 0 else "")
    if fees["commission"] > 0: fee_rows.append(f"| 达人佣金 | ¥{fees['commission']:,.2f} | {fees['commission']/ov['koutui_amt']*100:.1f}% |" if ov['koutui_amt'] > 0 else "")
    if fees["total"] > 0: fee_rows.append(f"| **费用合计** | **¥{fees['total']:,.2f}** | **{fees['total']/ov['koutui_amt']*100:.1f}%** |" if ov['koutui_amt'] > 0 else "")
    if fee_rows:
        rows.append("### 💰 费用结构")
        rows.append("")
        rows.append("| 费用项 | 金额 | 占销售额 |")
        rows.append("|--------|------|----------|")
        rows.extend(fee_rows)
        rows.append("")

    # ---- 主播表格 ----
    if r["top_zhubo"]:
        rows.append(f"### 👤 主播排名（共 {r['zhubo_count']} 位）")
        rows.append("")
        rows.append("| 排名 | 主播 | 销量 | 销售额 | 利润率 | 退款率 |")
        rows.append("|------|------|------|--------|--------|--------|")
        for i, z in enumerate(r["top_zhubo"][:15], 1):
            rows.append(
                f"| {i} | {z['zhubo'][:30]} | {z['vol']:,}件 | ¥{z['amt']:,.2f} | "
                f"{z['profit_margin']}% | {z['refund_rate']}% |"
            )
        if ov.get("top1_zhubo_share"):
            rows.append(f"| | **TOP1占比** | **{ov['top1_zhubo_share']}%** | **TOP3占比** | **{ov['top3_zhubo_share']}%** | |")
        rows.append("")

    # ---- SKU 表格 ----
    if r["top_sku"]:
        rows.append(f"### 📦 SKU TOP10（共 {r['sku_count']} 个）")
        rows.append("")
        rows.append("| 排名 | SKU 编码 | 款号 | 商品名称 | 总销量 | 总销售额 |")
        rows.append("|------|----------|------|----------|--------|----------|")
        for i, s in enumerate(r["top_sku"][:10], 1):
            display_code, model_code = _parse_sku_display(s["product_code"])
            name = s.get("product_name", "") or "-"
            rows.append(
                f"| {i} | {display_code} | {model_code} | {name[:20]} | "
                f"{s['vol']:,} 件 | {s['amt']:,.2f} 元 |"
            )
        # 集中度
        conc_parts = []
        for top_n in [5, 10, 20, 50]:
            key = f"top{top_n}_sku_share"
            if ov.get(key):
                conc_parts.append(f"TOP{top_n}={ov[key]}%")
        if conc_parts:
            rows.append(f"| | **集中度** | {' · '.join(conc_parts)} | | | |")
        rows.append("")

    # ---- 品牌表格 ----
    if r["brands"]:
        rows.append("### 🏷️ 品牌分布")
        rows.append("")
        rows.append("| 品牌 | 销量 | 占比 |")
        rows.append("|------|------|------|")
        for b in r["brands"][:10]:
            rows.append(f"| {b['brand'][:20]} | {b['vol']:,}件 | {b['share']}% |")
        rows.append("")

    # ---- 价格带表格 ----
    if r["price_bands"]:
        rows.append("### 💵 价格带分布")
        rows.append("")
        rows.append("| 价格区间 | 销量 | 销售额 | 均价 |")
        rows.append("|----------|------|--------|------|")
        for pb in r["price_bands"]:
            if pb["vol"] > 0:
                rows.append(f"| {pb['label']} | {pb['vol']:,}件 | ¥{pb['amt']:,.2f} | ¥{pb['avg_price']:.0f} |")
        rows.append(f"| **整体** | **{ov['koutui_vol']:,}件** | **¥{ov['koutui_amt']:,.2f}** | **¥{ov['avg_price']:.0f}** |")
        rows.append("")

    # ---- 类目 ----
    if r["categories"]:
        rows.append("### 📂 类目分布")
        rows.append("")
        rows.append("| 类目 | 销量 | 占比 |")
        rows.append("|------|------|------|")
        for c in r["categories"][:10]:
            rows.append(f"| {c['category'][:20]} | {c['vol']:,}件 | {c['share']}% |")
        rows.append("")

    # ---- 来源/渠道 ----
    if r["sources"]:
        rows.append("### 📱 成交来源分布")
        rows.append("")
        rows.append("| 来源 | 销量 | 销售额 | 利润 | 利润率 | 占比 |")
        rows.append("|------|------|--------|------|--------|------|")
        for s in r["sources"]:
            p_str = f"¥{s['profit']:,.2f}" if s['profit'] > 0 else "—"
            rows.append(f"| {s['source'][:20]} | {s['vol']:,}件 | ¥{s['amt']:,.2f} | {p_str} | {s['profit_margin']}% | {s['share']}% |")
        rows.append("")

    return nl.join(p for p in rows if p)


# ============================================================
# 多文件聚合
# ============================================================

def analyze_multiple_csvs(file_paths: List[str]) -> Dict[str, Any]:
    """多文件批量分析，返回每个文件的 structured 结果 + 汇总对比"""
    results = {}
    combined_overview = {
        "total_koutui_vol": 0, "total_koutui_amt": 0.0, "total_oper_profit": 0.0,
        "total_gross_profit": 0.0, "total_cost": 0.0,
        "total_refund_qty": 0, "total_refund_amt": 0.0,
        "total_ship_qty": 0, "total_ship_amt": 0.0,
        "total_fee": 0.0, "total_rows": 0,
        "file_count": len(file_paths),
    }
    all_zhubo: Dict[str, dict] = {}

    for fp in file_paths:
        r = analyze_csv_structured(fp)
        fname = os.path.basename(fp)
        results[fname] = r
        if not r.get("error"):
            ov = r["overview"]
            combined_overview["total_koutui_vol"] += ov.get("koutui_vol", 0) or 0
            combined_overview["total_koutui_amt"] += ov.get("koutui_amt", 0) or 0
            combined_overview["total_oper_profit"] += ov.get("oper_profit", 0) or 0
            combined_overview["total_gross_profit"] += ov.get("gross_profit", 0) or 0
            combined_overview["total_cost"] += ov.get("cost", 0) or 0
            combined_overview["total_refund_qty"] += ov.get("refund_qty", 0) or 0
            combined_overview["total_refund_amt"] += ov.get("refund_amt", 0) or 0
            combined_overview["total_ship_qty"] += ov.get("ship_qty", 0) or 0
            combined_overview["total_ship_amt"] += ov.get("ship_amt", 0) or 0
            combined_overview["total_fee"] += ov.get("total_fee", 0) or 0
            combined_overview["total_rows"] += ov.get("total_rows", 0) or 0
            for z in r["top_zhubo"]:
                nm = z["zhubo"]
                if nm not in all_zhubo:
                    all_zhubo[nm] = {"vol": 0, "amt": 0.0, "profit": 0.0, "ship_qty": 0, "ship_amt": 0.0}
                all_zhubo[nm]["vol"] += z["vol"]
                all_zhubo[nm]["amt"] += z["amt"]
                all_zhubo[nm]["profit"] += z.get("oper_profit", 0) or 0
                all_zhubo[nm]["ship_qty"] += z.get("ship_qty", 0) or 0
                all_zhubo[nm]["ship_amt"] += z.get("ship_amt", 0) or 0

    if combined_overview["total_koutui_amt"] > 0:
        combined_overview["overall_margin"] = round(
            combined_overview["total_oper_profit"] / combined_overview["total_koutui_amt"] * 100, 1
        )

    return {
        "files": results,
        "combined": combined_overview,
        "all_zhubo": sorted(all_zhubo.items(), key=lambda x: -x[1]["amt"]),
    }


# ============================================================
# 评论文件分析
# ============================================================

def analyze_comments_from_csv(file_path: str, top_n: int = 20) -> Dict[str, Any]:
    """分析评论 CSV 文件，返回结构化结果"""
    df = _read_csv_safe(file_path)
    if df is None:
        return {"error": "无法读取文件", "total": 0}

    total = len(df)

    # 检测关键列
    content_col = next((c for c in df.columns if c in ("评价内容", "评论内容", "评语", "评价", "评论")), None)
    if content_col is None:
        content_col = next((c for c in df.columns if "内容" in c or "评语" in c), df.columns[0])

    rating_col = next((c for c in df.columns if c in ("商品评价得分", "评分", "星级")), None)

    time_col = next((c for c in df.columns if c in ("评论时间", "评价时间", "创建时间")), None)

    product_col = next((c for c in df.columns if c in ("商品名称", "商品编码", "商品ID", "产品名称")), None)

    # 评分统计
    positive = neutral = negative = 0
    avg_rating = 0.0
    if rating_col:
        ratings = pd.to_numeric(df[rating_col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        positive = int((ratings >= 4).sum())
        neutral = int((ratings == 3).sum())
        negative = int((ratings <= 2).sum())
        avg_rating = round(float(ratings.mean()), 2) if ratings.notna().any() else 0

    # 评论最多的商品
    top_products = []
    if product_col and product_col in df.columns:
        counts = df[product_col].value_counts().head(top_n)
        for prod, cnt in counts.items():
            top_products.append({"product": str(prod).strip(), "count": int(cnt)})

    # 抽样评论
    samples = []
    if content_col:
        for text in df[content_col].dropna().head(10):
            samples.append(str(text)[:200])

    return {
        "file_name": os.path.basename(file_path),
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_rate": round(positive / total * 100, 1) if total > 0 else 0,
        "avg_rating": avg_rating,
        "top_products": top_products,
        "samples": samples,
    }


def analyze_comments_text(file_path: str) -> str:
    """评论分析 → LLM 可读文本"""
    r = analyze_comments_from_csv(file_path)
    if r.get("error"):
        return f"[评论分析失败: {r['error']}]"

    lines = [
        f"📄 {r['file_name']} — 共 {r['total']} 条评论",
        f"好评: {r['positive']} 条 | 中评: {r['neutral']} 条 | 差评: {r['negative']} 条",
        f"好评率: {r['positive_rate']}% | 平均评分: {r['avg_rating']}",
    ]
    if r["top_products"]:
        lines.append("\n评论最多的商品:")
        for p in r["top_products"][:10]:
            lines.append(f"  {p['product'][:30]}: {p['count']}条")
    if r["samples"]:
        lines.append("\n评论示例:")
        for s in r["samples"][:5]:
            lines.append(f"  - {s}")
    return "\n".join(lines)


# ============================================================
# 向后兼容的封装 (供 main.py 中老代码平滑迁移)
# ============================================================

def get_dashboard_data(file_name: str = "", month: int = 0, user_id: int = 0) -> Optional[Dict[str, Any]]:
    """替代 _calc_dashboard_from_csv: 从 CSV 构建完整 Dashboard JSON。
    支持：单文件 / 单月 / 全部文件。评论文件自动走评论分析路径。
    """
    all_files_full = get_all_uploaded_files(user_id=user_id)  # 始终返回全部文件给前端展示
    if not all_files_full:
        return None

    # 筛选用于分析的文件
    selected_files = all_files_full
    if file_name:
        selected_files = [f for f in selected_files if f["file_name"] == file_name]
    if month > 0:
        selected_files = [f for f in selected_files if re.search(rf'(?<!\d){month}\s*月', f["file_name"])]

    if not selected_files:
        return None

    # 检测是否为评论文件
    def _is_comment(fname):
        fn = fname.lower()
        return any(kw in fn for kw in ["评论", "评价", "comment", "review"])

    files_for_analysis = [f["file_path"] for f in selected_files]

    # 评论文件走评论分析路径
    if len(files_for_analysis) == 1 and _is_comment(selected_files[0]["file_name"]):
        r = analyze_comments_from_csv(files_for_analysis[0])
        if r.get("error"):
            return None
        ov = {
            "total_reviews": r["total"],
            "positive": r["positive"],
            "neutral": r["neutral"],
            "negative": r["negative"],
            "positive_rate": r["positive_rate"],
            "avg_rating": r["avg_rating"],
            "product_count": len(r.get("top_products", [])),
        }
        return {
            "success": True,
            "file_name": file_name,
            "dashboard_type": "comment",
            "overview": ov,
            "top_products": r.get("top_products", [])[:20],
            "files": [{"file_name": f["file_name"]} for f in all_files_full],
        }

    # 排除评论文件（销售分析不包含评论文件）
    sales_files = [f for f in files_for_analysis if not _is_comment(os.path.basename(f))]
    if not sales_files:
        return None

    multi = analyze_multiple_csvs(sales_files)
    total_file_count = len(sales_files)

    # 多文件聚合（"全部"）：用 combined 数据
    if total_file_count > 1 and month == 0 and not file_name:
        combined = multi["combined"]
        all_zhubo = multi.get("all_zhubo", [])
        # 汇总所有文件的 SKU 数和主播数
        total_sku = 0
        total_zhubo_set = set()
        for fname, r in multi["files"].items():
            if not r.get("error"):
                total_sku += r.get("sku_count", 0)
        total_zhubo = len(all_zhubo)

        # ── 多文件 SKU 聚合 ──
        from collections import defaultdict as _dd
        sku_agg: dict[str, dict] = _dd(lambda: {"vol": 0, "amt": 0.0, "profit": 0.0, "name": ""})
        for _fname, r in multi["files"].items():
            if r.get("error"):
                continue
            for s in r.get("top_sku", []):
                code = s.get("product_code", "").strip()
                if not code:
                    continue
                sku_agg[code]["vol"] += s.get("vol", 0) or 0
                sku_agg[code]["amt"] += s.get("amt", 0) or 0
                sku_agg[code]["profit"] += s.get("profit", 0) or 0
                # 取第一个非空名称
                if not sku_agg[code]["name"] and s.get("product_name", ""):
                    sku_agg[code]["name"] = s["product_name"]
        # 按销量排序取 TOP 20
        sorted_sku = sorted(sku_agg.items(), key=lambda x: -x[1]["vol"])
        all_top_sku = [
            {
                "product_code": code,
                "product_name": d["name"],
                "vol": d["vol"],
                "amt": round(d["amt"], 2),
                "profit": round(d["profit"], 2),
            }
            for code, d in sorted_sku[:20]
        ]

        total_amt = combined["total_koutui_amt"]
        profit_rate = round(combined["total_oper_profit"] / total_amt * 100, 1) if total_amt > 0 else 0
        refund_rate = round(combined["total_refund_qty"] / (combined["total_koutui_vol"] + combined["total_refund_qty"]) * 100, 1) if (combined["total_koutui_vol"] + combined["total_refund_qty"]) > 0 else 0

        gross_margin = round((total_amt - combined["total_cost"]) / total_amt * 100, 1) if total_amt > 0 else 0
        avg_price = round(total_amt / combined["total_koutui_vol"], 0) if combined["total_koutui_vol"] > 0 else 0

        ov = {
            "total_rows": combined["total_rows"],
            "koutui_vol": combined["total_koutui_vol"],
            "koutui_amt": round(total_amt, 2),
            "cost": round(combined["total_cost"], 2),
            "gross_profit": round(combined["total_gross_profit"], 2),
            "gross_margin": gross_margin,
            "oper_profit": round(combined["total_oper_profit"], 2),
            "oper_margin": profit_rate,
            "profit_rate": profit_rate,
            "profit": round(combined["total_gross_profit"], 2),
            "total_fee": round(combined["total_fee"], 2),
            "ship_qty": combined["total_ship_qty"],
            "ship_amt": round(combined["total_ship_amt"], 2),
            "refund_qty": combined["total_refund_qty"],
            "refund_amt": round(combined["total_refund_amt"], 2),
            "refund_rate": refund_rate,
            "avg_price": avg_price,
            "sku_count": total_sku,
            "zhubo_count": total_zhubo,
            "file_count": combined["file_count"],
        }
        top_zhubo = []
        for name, d in all_zhubo[:20]:
            top_zhubo.append({
                "zhubo": name,
                "vol": d["vol"], "amt": round(d["amt"], 2),
                "ship_qty": d.get("ship_qty", 0), "ship_amt": round(d.get("ship_amt", 0), 2),
                "profit": round(d["profit"], 2),
            })

        return {
            "success": True,
            "file_name": file_name,
            "overview": ov,
            "top_zhubo": top_zhubo,
            "top_sku": all_top_sku,
            "files": [{"file_name": f["file_name"]} for f in all_files_full],
            "combined": combined,
        }

    # 单文件：用第一个文件的分析
    first_key = list(multi["files"].keys())[0] if multi["files"] else None
    single = multi["files"].get(first_key) if first_key else None

    if not single:
        return None

    ov = single["overview"]
    # 补上 dashboard 前端需要的字段
    ov["sku_count"] = single.get("sku_count", 0)
    ov["zhubo_count"] = single.get("zhubo_count", 0)
    if "oper_margin" in ov:
        ov["profit_rate"] = ov["oper_margin"]
    if "gross_profit" in ov:
        ov["profit"] = ov["gross_profit"]
    # 给 top_zhubo 加 profit 别名（单文件用 gross_profit）
    top_zhubo_items = []
    for z in single.get("top_zhubo", [])[:20]:
        z = dict(z)
        if "profit" not in z:
            z["profit"] = z.get("gross_profit", z.get("oper_profit", 0)) or 0
        if "ship_qty" not in z:
            z["ship_qty"] = z.get("ship_qty", 0) or 0
        if "ship_amt" not in z:
            z["ship_amt"] = z.get("ship_amt", 0) or 0
        top_zhubo_items.append(z)

    return {
        "success": True,
        "file_name": file_name,
        "overview": ov,
        "fees": single["fees"],
        "top_zhubo": top_zhubo_items,
        "top_sku": single["top_sku"][:20],
        "brands": single["brands"],
        "price_bands": single["price_bands"],
        "categories": single["categories"],
        "files": [{"file_name": f["file_name"]} for f in all_files_full],
        "combined": multi["combined"],
    }


def get_analysis_for_llm(file_paths: List[str], question: str = "") -> Tuple[str, str]:
    """为 LLM 准备分析上下文。

    返回 (data_text, source_hint)
    - data_text: 给 LLM 的结构化数据文本
    - source_hint: 数据来源说明
    """
    if not file_paths:
        return "", "无可用数据文件"

    # 判断文件类型
    is_comment = any("评论" in os.path.basename(f) or "评价" in f for f in file_paths)

    parts = []
    for fp in file_paths:
        fname = os.path.basename(fp)
        if is_comment:
            parts.append(analyze_comments_text(fp))
        else:
            parts.append(analyze_csv_text(fp))

    data_text = "\n\n---\n\n".join(p for p in parts if p and "失败" not in p)

    # Source hint
    if len(file_paths) == 1:
        source_hint = f"**数据来源：{os.path.basename(file_paths[0])}**"
    else:
        source_hint = f"数据来源：{len(file_paths)} 个文件"

    return data_text, source_hint
