"""女装牛仔裤电商智能助手 - 主应用"""

import os
import io
import json
import uuid
import hashlib
import logging
import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import asyncio

from sqlalchemy import text, func, and_, insert
from sqlalchemy.orm import Session

from storage.database.db import get_engine, get_session
from ecommerce.models import Base, User, SalesData, ReportFile, Conversation, GeneratedImage, DashboardCache, SessionFile, TaskStatus

from coze_coding_dev_sdk import LLMClient, ImageGenerationClient
from coze_coding_utils.runtime_ctx.context import new_context
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# ============================================================
# 应用初始化
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("🚀 女装牛仔裤电商智能助手启动中...")
    # 创建数据库表
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("✅ 数据库表创建/确认完成")
    yield
    logger.info("👋 应用关闭")

app = FastAPI(
    title="女装牛仔裤电商智能助手",
    description="AI驱动的女装牛仔裤电商智能运营平台",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务 - 前端
STATIC_DIR = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets", "ecommerce")
os.makedirs(STATIC_DIR, exist_ok=True)

# ============================================================
# 请求/响应模型
# ============================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class AskRequest(BaseModel):
    question: str
    session_id: str = ""
    file_id: int = 0  # 指定文件ID，0表示自动匹配

class CopywritingRequest(BaseModel):
    product_name: str = "高腰显瘦牛仔裤"
    style: str = "直筒"
    scene: str = "抖音短视频"
    target_audience: str = "年轻女性"
    key_selling_points: str = "显瘦、高腰、百搭"
    tone: str = "亲和"

class ScriptRequest(BaseModel):
    script_type: str = "种草"
    product_name: str = "高腰显瘦牛仔裤"
    duration: str = "30s"
    target_audience: str = "年轻女性"

class StrategyRequest(BaseModel):
    focus: str = "综合"
    period: str = "本月"

class ImageGenRequest(BaseModel):
    prompt: str
    style: str = "写实"
    size: str = "2K"

class DataEditRequest(BaseModel):
    id: int
    field: str
    value: Any

class SessionHistoryRequest(BaseModel):
    session_id: str
    msg_type: str = ""

# ============================================================
# 工具函数
# ============================================================

def get_db() -> Session:
    return get_session()

def make_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_llm_client():
    ctx = new_context(method="ecommerce_llm")
    return LLMClient(ctx=ctx)

def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
    """调用大模型统一入口"""
    client = get_llm_client()
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    response = client.invoke(
        messages=messages,
        model="doubao-seed-2-0-lite-260215",
        temperature=temperature,
        max_completion_tokens=8192,
    )
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        parts = []
        for item in response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(response.content)

def detect_sales_fields(db: Session, report_name: str = None) -> dict:
    """自动检测extra_data中的销售字段名，兼容不同ERP版本"""
    base_q = db.query(SalesData)
    if report_name:
        base_q = base_q.filter(SalesData.report_name == report_name)
    
    # 取一条有extra_data的记录
    sample = base_q.filter(
        SalesData.extra_data.isnot(None),
        text("extra_data::text != 'null'")
    ).first()
    
    if not sample or not sample.extra_data:
        return {}
    
    keys = set(sample.extra_data.keys())
    
    # 字段检测优先级：按顺序匹配，第一个命中的为准
    field_map = {
        'koutui_vol': ['商品销售数据-商品销售数量(扣退)', '利润-销售数量(扣退)', '商品销售数据(其中分类单)-分类单销售数量(扣退)'],
        'koutui_amt': ['商品销售数据-商品销售金额(扣退)', '利润-销售金额(扣退)', '商品销售数据(其中分类单)-分类单销售金额(扣退)'],
        'koutui_cost': ['商品销售数据-商品销售成本(扣退)', '利润-销售成本(扣退)', '商品销售数据(其中分类单)-分类单销售成本(扣退)'],
        'ship_qty': ['商品数据-实发数量'],
        'ship_amt': ['商品数据-实发金额'],
        'ship_cost': ['商品数据-实发成本'],
        'order_qty': ['商品数据-商品数量'],
        'order_amt': ['商品数据-商品金额'],
        'order_cost': ['商品数据-商品成本'],
        'profit': ['利润-毛利额', '利润-经营利润'],
        'refund_qty': ['售后合计-退款数量合计', '退款数量合计'],
        'refund_amt': ['售后合计-退款金额合计', '退款金额合计'],
        'return_qty': ['售后合计-退货数量合计'],
        'return_amt': ['售后合计-退货金额合计'],
        'product_code': ['商品编码'],
        'product_name': ['商品简称', '商品名称', '【商品资料】：商品简称'],
        'zhubo': ['主播'],
        'category': ['分类', '商品类目'],
        'brand': ['品牌'],
        'supplier': ['供应商名称'],
        'shop_name': ['店铺名称'],
    }
    
    detected = {}
    for logical_name, candidates in field_map.items():
        for c in candidates:
            if c in keys:
                detected[logical_name] = c
                break
    
    return detected


def _build_extra_agg_sql(detected: dict, prefix: str = "extra_data->>'") -> dict:
    """根据检测到的字段构建SQL聚合表达式"""
    agg = {}
    for logical, field in detected.items():
        safe_field = field.replace("'", "''")
        agg[logical] = f"SUM(COALESCE(NULLIF({prefix}{safe_field}','')::numeric, 0))"
    return agg


def get_sales_summary(db: Session, year: int = 2025, month: int = 0) -> str:
    """获取销售数据摘要用于RAG问答"""
    base = db.query(SalesData).filter(SalesData.data_year == year)
    if month > 0:
        base = base.filter(SalesData.data_month == month)
        month_label = f"{year}年{month}月"
    else:
        month_label = f"{year}年"
    
    # 总览
    total = base.with_entities(
        func.sum(SalesData.sales_volume).label("total_vol"),
        func.sum(SalesData.sales_amount).label("total_amt"),
        func.sum(SalesData.profit).label("total_profit"),
        func.sum(SalesData.cost).label("total_cost"),
        func.sum(SalesData.return_count).label("total_return"),
    ).first()

    # 按月统计
    monthly = base.with_entities(
        SalesData.data_month,
        func.sum(SalesData.sales_volume).label("vol"),
        func.sum(SalesData.sales_amount).label("amt"),
        func.sum(SalesData.profit).label("profit"),
    ).group_by(SalesData.data_month).order_by(SalesData.data_month).all()

    # 按SKU排行
    top_sku = base.with_entities(
        SalesData.sku_code,
        SalesData.product_name,
        func.sum(SalesData.sales_volume).label("vol"),
        func.sum(SalesData.sales_amount).label("amt"),
    ).group_by(
        SalesData.sku_code, SalesData.product_name
    ).order_by(
        func.sum(SalesData.sales_volume).desc()
    ).limit(20).all()

    lines = []
    lines.append(f"=== {month_label} 女装牛仔裤销售数据总览 ===")
    lines.append(f"总销量: {total[0] or 0}")
    lines.append(f"总销售额: ¥{total[1] or 0:,.2f}")
    lines.append(f"总利润: ¥{total[2] or 0:,.2f}")
    lines.append(f"总成本: ¥{total[3] or 0:,.2f}")
    lines.append(f"总退货数: {total[4] or 0}")
    lines.append("")

    lines.append("=== 月度销售数据 ===")
    for m in monthly:
        lines.append(f"  {year}年{m[0]}月 - 销量:{m[1] or 0}, 销售额:¥{m[2] or 0:,.2f}, 利润:¥{m[3] or 0:,.2f}")

    lines.append("")
    lines.append("=== TOP 20 SKU销量排行 ===")
    for i, sku in enumerate(top_sku):
        lines.append(f"  {i+1}. [{sku[0]}] {sku[1]} - 销量:{sku[2] or 0}, 销售额:¥{sku[3] or 0:,.2f}")

    # 检测是否有 extra_data 自定义字段
    try:
        from sqlalchemy import text as sa_text
        extra_with_data = base.filter(
            SalesData.extra_data.isnot(None),
            sa_text("extra_data::text != 'null'")
        ).limit(5).all()
        if extra_with_data:
            extra_keys = set()
            for rec in extra_with_data:
                if isinstance(rec.extra_data, dict):
                    extra_keys.update(rec.extra_data.keys())
            if extra_keys:
                lines.append("")
                lines.append(f"=== 自定义字段 (检测到: {', '.join(sorted(extra_keys))}) ===")
                # 识别可能包含销售数值的字段
                numeric_keys = set()
                text_keys = set()
                for ek in sorted(extra_keys):
                    for rec in extra_with_data[:10]:
                        if isinstance(rec.extra_data, dict) and ek in rec.extra_data:
                            val = rec.extra_data[ek]
                            if isinstance(val, (int, float)):
                                numeric_keys.add(ek)
                                break
                            if isinstance(val, str):
                                try:
                                    float(val)
                                    numeric_keys.add(ek)
                                    break
                                except:
                                    text_keys.add(ek)
                                    break
                            else:
                                text_keys.add(ek)
                                break
                
                # 对文本型字段（如主播）进行分组统计，累加数值型字段
                for ek in sorted(text_keys):
                    try:
                        ek_safe = ek.replace("'", "''")
                        # 构建数值累加表达式
                        sum_parts = []
                        for nk in sorted(numeric_keys):
                            nk_safe = nk.replace("'", "''")
                            sum_parts.append(
                                f"COALESCE((extra_data->>'{nk_safe}')::numeric, 0)"
                            )
                        # 如果有数值字段，累加；否则用标准字段
                        if sum_parts:
                            vol_expr = "+".join(sum_parts)
                        else:
                            vol_expr = "COALESCE(sales_volume, 0)"
                        
                        raw_res = db.execute(sa_text(f"""
                            SELECT 
                                extra_data->>'{ek_safe}' AS ekey,
                                COUNT(*) AS rec_cnt,
                                {vol_expr} AS total_vol
                            FROM ec_sales_data
                            WHERE data_year = :yr {f'AND data_month = :mo' if month > 0 else ''}
                                AND extra_data IS NOT NULL 
                                AND extra_data::text != 'null'
                                AND extra_data->>'{ek_safe}' IS NOT NULL
                            GROUP BY extra_data->>'{ek_safe}'
                            ORDER BY total_vol DESC
                            LIMIT 50
                        """), {"yr": year, "mo": month} if month > 0 else {"yr": year}).all()
                        
                        if raw_res:
                            lines.append(f"\n--- 按 [{ek}] 统计 ---")
                            for rr in raw_res:
                                if rr[0]:
                                    lines.append(f"  {rr[0]}: 关联{rr[1]}条记录, 合计销量约{rr[2] or 0}")
                    except Exception as e_ek:
                        lines.append(f"  ({ek} 统计跳过: {str(e_ek)[:50]})")
                
                # 对数值型字段，展示整体汇总
                if numeric_keys:
                    lines.append(f"\n--- 数值字段汇总 ---")
                    for nk in sorted(numeric_keys):
                        try:
                            nk_safe = nk.replace("'", "''")
                            total_val = db.execute(text(f"""
                                SELECT SUM((extra_data->>'{nk_safe}')::numeric)
                                FROM ec_sales_data
                                WHERE data_year = :yr {f'AND data_month = :mo' if month > 0 else ''}
                                    AND extra_data IS NOT NULL 
                                    AND extra_data::text != 'null'
                                    AND extra_data->>'{nk_safe}' ~ '^[0-9]+\\.?[0-9]*$'
                            """), {"yr": year, "mo": month} if month > 0 else {"yr": year}).scalar()
                            if total_val:
                                lines.append(f"  {nk}: 合计 {total_val:,.2f}")
                        except:
                            pass
    except Exception as e_extra:
        lines.append(f"  (额外字段检测跳过: {str(e_extra)[:50]})")

    return "\n".join(lines)


def get_sales_summary_with_extra(db: Session, year: int = 2025, file_id: int = 0, month: int = 0):
    """获取销售数据摘要，包含extra_data中的自定义字段"""
    # 第一步：按文件名匹配月份（用户说"3月" → 匹配文件名含"3月"的数据）
    # 同时兼容data_month字段
    month_label = f"{year}年{month}月" if month > 0 else f"{year}年"
    matched_files = []
    
    if file_id > 0:
        report = db.query(ReportFile).filter(ReportFile.id == file_id).first()
        if report:
            base_q = db.query(SalesData).filter(SalesData.report_name == report.file_name)
        else:
            base_q = db.query(SalesData).filter(SalesData.data_year == year)
    elif month > 0:
        # 优先按文件名匹配月份
        month_patterns = [f"{month}月", f"{month:02d}月", f"_{month}_", f"-{month}-", f"{month}月份"]
        file_names = db.execute(text("""
            SELECT DISTINCT report_name FROM ec_sales_data WHERE data_year = :yr
        """), {"yr": year}).scalars().all()
        
        matched_files = []
        for fn in file_names:
            if fn:
                for pat in month_patterns:
                    if pat in fn:
                        matched_files.append(fn)
                        break
        
        if matched_files:
            base_q = db.query(SalesData).filter(SalesData.report_name.in_(matched_files))
            month_label = f"{year}年{month}月（文件: {', '.join(matched_files)}）"
        else:
            # 回退到data_month
            base_q = db.query(SalesData).filter(SalesData.data_year == year, SalesData.data_month == month)
            cnt = base_q.count()
            if cnt == 0:
                base_q = db.query(SalesData).filter(SalesData.data_year == year)
                month_label = f"{year}年{month}月（无匹配文件，展示全部数据）"
    else:
        base_q = db.query(SalesData).filter(SalesData.data_year == year)
    
    total_count = base_q.count()
    if total_count == 0:
        return f"=== {month_label} ===\n数据库中暂无数据\n", []
    
    # 自动检测字段
    detected = detect_sales_fields(db, report_name=matched_files[0] if (month > 0 and matched_files) else None)
    kv_f = detected.get('koutui_vol', '商品销售数据-商品销售数量(扣退)')
    ka_f = detected.get('koutui_amt', '商品销售数据-商品销售金额(扣退)')
    sq_f = detected.get('ship_qty', '商品数据-实发数量')
    sa_f = detected.get('ship_amt', '商品数据-实发金额')
    pf_f = detected.get('profit', '利润-毛利额')
    rq_f = detected.get('refund_qty', '售后合计-退款数量合计')
    zb_f = detected.get('zhubo', '主播')
    pc_f = detected.get('product_code', '商品编码')
    pn_f = detected.get('product_name', '商品简称')
    
    def safe_col(fn):
        return fn.replace("'", "''")
    
    # 总览
    total = db.execute(text(f"""
        SELECT 
            COUNT(*) as cnt,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(kv_f)}','')::numeric, 0)) as koutui_vol,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(ka_f)}','')::numeric, 0)) as koutui_amt,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(sq_f)}','')::numeric, 0)) as ship_qty,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(sa_f)}','')::numeric, 0)) as ship_amt,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(pf_f)}','')::numeric, 0)) as profit,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(rq_f)}','')::numeric, 0)) as refund_qty
        FROM ec_sales_data
        WHERE id IN (SELECT id FROM ec_sales_data WHERE data_year = :yr)
          AND extra_data IS NOT NULL AND extra_data::text != 'null'
    """), {"yr": year}).first()
    
    # 如果base_q有限制，重新计算
    if month > 0 or file_id > 0:
        ids = [r[0] for r in base_q.with_entities(SalesData.id).all()]
        if ids:
            id_list = ','.join(str(i) for i in ids[:50000])
            total = db.execute(text(f"""
                SELECT 
                    COUNT(*) as cnt,
                    SUM(COALESCE(NULLIF(extra_data->>'{safe_col(kv_f)}','')::numeric, 0)) as koutui_vol,
                    SUM(COALESCE(NULLIF(extra_data->>'{safe_col(ka_f)}','')::numeric, 0)) as koutui_amt,
                    SUM(COALESCE(NULLIF(extra_data->>'{safe_col(sq_f)}','')::numeric, 0)) as ship_qty,
                    SUM(COALESCE(NULLIF(extra_data->>'{safe_col(sa_f)}','')::numeric, 0)) as ship_amt,
                    SUM(COALESCE(NULLIF(extra_data->>'{safe_col(pf_f)}','')::numeric, 0)) as profit,
                    SUM(COALESCE(NULLIF(extra_data->>'{safe_col(rq_f)}','')::numeric, 0)) as refund_qty
                FROM ec_sales_data
                WHERE id IN ({id_list})
                  AND extra_data IS NOT NULL AND extra_data::text != 'null'
            """)).first()
    
    koutui_vol = int(total[1] or 0)
    koutui_amt = float(total[2] or 0)
    ship_qty = int(total[3] or 0)
    ship_amt = float(total[4] or 0)
    profit = float(total[5] or 0)
    refund_qty = int(total[6] or 0)
    
    lines = []
    lines.append(f"=== {month_label} 女装牛仔裤销售数据总览 ===")
    lines.append(f"总记录数: {total_count}")
    lines.append(f"扣退后销量: {koutui_vol:,}件")
    lines.append(f"扣退后销售额: ¥{koutui_amt:,.2f}")
    lines.append(f"实发数量: {ship_qty:,}件")
    lines.append(f"实发金额: ¥{ship_amt:,.2f}")
    lines.append(f"毛利额: ¥{profit:,.2f}")
    lines.append(f"退款数量: {refund_qty:,}件")
    lines.append(f"检测字段: 销量={kv_f}, 金额={ka_f}, 毛利={pf_f}")
    lines.append("")
    
    # 按主播聚合（使用检测到的字段）
    zhubo_stats = {}
    all_data = base_q.filter(SalesData.extra_data.isnot(None)).filter(
        text(f"ec_sales_data.extra_data->>'{safe_col(zb_f)}' IS NOT NULL")
    ).filter(
        text(f"ec_sales_data.extra_data->>'{safe_col(zb_f)}' != ''")
    ).yield_per(500).all()
    
    for s in all_data:
        z_name = str(s.extra_data.get(zb_f, "")).strip().replace('\t', '')
        if z_name in ("nan", "NaN", "None", ""):
            continue
        if z_name not in zhubo_stats:
            zhubo_stats[z_name] = {"vol": 0, "amt": 0.0, "ship_qty": 0, "ship_amt": 0.0, "profit": 0.0, "sku_cnt": 0}
        zhubo_stats[z_name]["sku_cnt"] += 1
        for key in (s.extra_data or {}):
            val_str = str(s.extra_data.get(key, "0")).replace('\t', '').strip()
            try:
                val = float(val_str) if val_str.replace('.','',1).lstrip('-').isdigit() else 0
            except:
                val = 0
            if key == kv_f:
                zhubo_stats[z_name]["vol"] += int(val)
            elif key == ka_f:
                zhubo_stats[z_name]["amt"] += val
            elif key == sq_f:
                zhubo_stats[z_name]["ship_qty"] += int(val)
            elif key == sa_f:
                zhubo_stats[z_name]["ship_amt"] += val
            elif key == pf_f:
                zhubo_stats[z_name]["profit"] += val
    
    if zhubo_stats:
        sorted_zhubo = sorted(zhubo_stats.items(), key=lambda x: x[1]["vol"], reverse=True)
        lines.append(f"=== 按主播聚合（{zb_f}，按扣退后销量排序） ===")
        for i, (name, stats) in enumerate(sorted_zhubo[:30]):
            lines.append(f"  {i+1}. {name} - 扣退后销量:{stats['vol']}件, 扣退后销售额:¥{stats['amt']:,.0f}, 实发:{stats['ship_qty']}件 ¥{stats['ship_amt']:,.0f}, 毛利:¥{stats['profit']:,.0f}, 关联SKU:{stats['sku_cnt']}个")
        if len(sorted_zhubo) > 30:
            lines.append(f"  ...共{len(sorted_zhubo)}位主播/达人")
        lines.append("")
    
    # TOP SKU
    top_sku = db.execute(text(f"""
        SELECT 
            extra_data->>'{safe_col(pc_f)}' as product_code,
            extra_data->>'{safe_col(pn_f)}' as product_name,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(kv_f)}','')::numeric, 0)) as vol,
            SUM(COALESCE(NULLIF(extra_data->>'{safe_col(ka_f)}','')::numeric, 0)) as amt
        FROM ec_sales_data
        WHERE id IN (SELECT id FROM ec_sales_data WHERE data_year = :yr)
          AND extra_data IS NOT NULL AND extra_data::text != 'null'
        GROUP BY extra_data->>'{safe_col(pc_f)}', extra_data->>'{safe_col(pn_f)}'
        ORDER BY vol DESC
        LIMIT 20
    """), {"yr": year}).all()
    
    if top_sku:
        lines.append("=== TOP 20 SKU（按扣退后销量） ===")
        for s in top_sku:
            lines.append(f"  {s[0] or '?'} {s[1] or ''}: {int(s[2] or 0)}件 ¥{float(s[3] or 0):,.0f}")
        lines.append("")
    
    return "\n".join(lines), list(detected.keys())


# ============================================================
# API 路由 - 电商小助手 (文件管理 + 智能问答)
# ============================================================

def _process_sales_chunk(db, df, filename, chunk_idx):
    """处理一块DataFrame数据并写入数据库，返回处理行数（使用批量插入）"""
    from sqlalchemy import insert
    
    col_map = {
        "sku编码":"sku_code","sku_code":"sku_code","商品编号":"sku_code","skuid":"sku_code","商品id":"sku_code","spu":"sku_code",
        "商品名称":"product_name","product_name":"product_name","商品标题":"product_name","标题":"product_name",
        "类目":"category","category":"category","商品类目":"category","品类":"category",
        "版型":"style","style":"style","裤型":"style",
        "颜色":"color","color":"color","商品颜色":"color",
        "尺码":"size","size":"size","商品尺码":"size","规格":"size",
        "销量":"sales_volume","sales_volume":"sales_volume","销售数量":"sales_volume","商品销售数据-商品销售数量":"sales_volume",
        "商品销售数据-商品销售数量(扣退)":"sales_volume_net","销售数量(扣退)":"sales_volume_net",
        "销售额":"sales_amount","sales_amount":"sales_amount","销售金额":"sales_amount","商品销售数据-商品销售金额":"sales_amount",
        "商品销售数据-商品销售金额(扣退)":"sales_amount_net","销售金额(扣退)":"sales_amount_net",
        "成本":"cost","cost":"cost","商品成本":"cost","进货成本":"cost",
        "利润":"profit","profit":"profit","毛利":"profit","利润-毛利额":"profit","毛利额":"profit",
        "利润率":"profit_margin","profit_margin":"profit_margin","毛利率":"profit_margin",
        "退货数":"return_count","return_count":"return_count","退款数量":"return_count","售后退款数量":"return_count",
        "退货率":"return_rate","return_rate":"return_rate","退款率":"return_rate",
        "库存":"inventory","inventory":"inventory","库存数量":"inventory",
        "月份":"data_month","data_month":"data_month","月":"data_month","月份(数字)":"data_month",
    }
    known_cols = set(col_map.values()) | {"data_year", "data_quarter"}
    df_std = df.rename(columns=col_map)
    
    records = []
    for _, row in df_std.iterrows():
        try:
            extra = {}
            for col in df.columns:
                if col not in known_cols:
                    val = row.get(col)
                    if val is not None and val != "":
                        extra[col] = str(val)
            
            records.append({
                "report_name": filename,
                "data_year": 2025,
                "data_month": int(row.get("data_month", 1)),
                "sku_code": str(row.get("sku_code",""))[:100],
                "product_name": str(row.get("product_name",""))[:300],
                "category": str(row.get("category",""))[:100],
                "style": str(row.get("style",""))[:100],
                "color": str(row.get("color",""))[:50],
                "size": str(row.get("size",""))[:50],
                "sales_volume": int(float(row.get("sales_volume",0))),
                "sales_amount": float(row.get("sales_amount",0)),
                "cost": float(row.get("cost",0)),
                "profit": float(row.get("profit",0)),
                "profit_margin": float(row.get("profit_margin",0)),
                "return_count": int(float(row.get("return_count",0))),
                "return_rate": float(row.get("return_rate",0)),
                "inventory": int(float(row.get("inventory",0))),
                "extra_data": extra if extra else None,
            })
        except:
            pass
    
    if records:
        stmt = insert(SalesData).values(records)
        db.execute(stmt)
    return len(records)

@app.post("/api/ask/upload")
async def ask_upload_file(file: UploadFile = File(...), session_id: str = Form("")):
    """电商小助手界面上传文件（后台处理，避免超时）"""
    fn = file.filename.lower()
    if not fn.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="仅支持 CSV/Excel")

    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    tmp_path = os.path.join("/tmp", safe_name)
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    task_id = uuid.uuid4().hex[:16]
    db = get_db()
    try:
        task = TaskStatus(
            task_id=task_id, file_name=file.filename,
            status="pending", progress="文件已接收，正在后台处理..."
        )
        db.add(task)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    _run_in_background(_process_ask_upload_task, task_id, tmp_path,
                       file.filename, session_id or "default")

    return {
        "success": True,
        "message": "文件已接收，正在后台处理，请稍候...",
        "task_id": task_id,
        "file_name": file.filename,
    }


def _process_ask_upload_task(task_id, tmp_path, filename, session_id):
    """后台处理小助手上传文件"""
    _update_task(task_id, status="processing", progress="正在解析文件...")
    db = get_db()
    try:
        fn = filename.lower()
        total_rows = 0
        preview_lines = []
        all_columns = []
        first_chunk = True

        if fn.endswith(".csv"):
            chunks = list(pd.read_csv(tmp_path, encoding='utf-8',
                                      chunksize=2000, low_memory=False))
            for chunk_idx, df_chunk in enumerate(chunks):
                df_chunk.columns = [c.strip().lower() for c in df_chunk.columns]
                if first_chunk:
                    all_columns = list(df_chunk.columns)
                    first_chunk = False
                total_rows += _process_sales_chunk(db, df_chunk, filename, chunk_idx)
                db.commit()
                if len(preview_lines) < 20:
                    for _, row in df_chunk.head(20 - len(preview_lines)).iterrows():
                        preview_lines.append(str(dict(row)))
                _update_task(task_id,
                             progress=f"第 {chunk_idx+1}/{len(chunks)} 批完成 (已导入 {total_rows} 行)")
        else:
            df = pd.read_excel(tmp_path)
            df.columns = [c.strip().lower() for c in df.columns]
            all_columns = list(df.columns)
            total_rows = _process_sales_chunk(db, df, filename, 0)
            preview_lines = [str(dict(row)) for _, row in df.head(20).iterrows()]

        preview = (
            f"文件名: {filename}\n"
            f"列名: {', '.join(all_columns)}\n"
            f"总行数: {total_rows}\n"
            f"--- 数据预览前20行 ---\n" + "\n".join(preview_lines)
        )
        sf = SessionFile(
            session_id=session_id, file_name=filename, file_size=0,
            row_count=total_rows, columns=",".join(all_columns),
            data_preview=preview
        )
        db.add(sf)
        db.commit()
        _update_task(task_id, status="completed",
                     progress=f"成功导入 {total_rows} 条数据", row_count=total_rows)
    except Exception as e:
        db.rollback()
        logger.exception(f"后台任务失败: {task_id}")
        _update_task(task_id, status="failed", progress="处理失败", detail=str(e))
    finally:
        db.close()
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/api/ask/candidates")
async def ask_candidates(question: str = Query(""), session_id: str = Query("")):
    """根据问题关键词匹配合适的数据文件"""
    import re
    db = get_db()
    try:
        years = re.findall(r'20\d{2}', question)
        q = db.query(ReportFile).order_by(ReportFile.created_at.desc())
        if years:
            q = q.filter(ReportFile.data_year == int(years[0]))
        files = q.limit(50).all()
        return {"success": True, "items": [{"id":f.id,"file_name":f.file_name,"data_year":f.data_year,"report_period":f.report_period,"row_count":f.row_count,"created_at":f.created_at.isoformat() if f.created_at else ""} for f in files], "total": len(files)}
    finally:
        db.close()

@app.post("/api/ask/select-file")
async def ask_select_file(session_id: str = "", file_id: int = 0):
    """用户选择要分析的文件"""
    db = get_db()
    try:
        report = db.query(ReportFile).filter(ReportFile.id == file_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="文件不存在")
        sf = SessionFile(session_id=session_id or "default", file_name=report.file_name, file_size=report.file_size, row_count=report.row_count, columns="", data_preview=f"已选择文件: {report.file_name}")
        db.add(sf)
        db.commit()
        return {"success": True, "message": f"已选择文件: {report.file_name}", "file_id": file_id, "file_name": report.file_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ============================================================
# API 路由 - 电商小助手 (智能问答)
# ============================================================

@app.post("/api/login")
async def login(req: LoginRequest):
    db = get_db()
    try:
        user = db.query(User).filter(User.username == req.username).first()
        if not user:
            # 自动注册（无权限体系）
            user = User(
                username=req.username,
                password_hash=make_password_hash(req.password),
                display_name=req.username,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return {"success": True, "user_id": user.id, "username": user.username, "display_name": user.display_name, "is_new": True}
        
        if user.password_hash != make_password_hash(req.password):
            raise HTTPException(status_code=401, detail="密码错误")
        
        return {"success": True, "user_id": user.id, "username": user.username, "display_name": user.display_name}
    finally:
        db.close()


# ============================================================
# API 路由 - 数据中心
# ============================================================

@app.post("/api/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    data_year: int = Form(2025),
    data_month: int = Form(0),
):
    """上传销售数据文件（立即保存 + 后台处理，避免上游超时）"""
    filename = file.filename.lower()
    if not filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="仅支持 CSV / Excel 文件格式")

    # 立即保存文件到临时目录，快速返回
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    tmp_path = os.path.join("/tmp", safe_name)
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    file_size = len(content)

    # 创建任务记录
    task_id = uuid.uuid4().hex[:16]
    db = get_db()
    try:
        task = TaskStatus(
            task_id=task_id,
            file_name=file.filename,
            status="pending",
            progress="文件已接收，排队等待处理...",
        )
        db.add(task)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # 在后台线程中处理
    _run_in_background(_process_upload_task, task_id, tmp_path,
                       file.filename, file_size, data_year, data_month)

    return {
        "success": True,
        "message": "文件已接收，正在后台处理，请稍候...",
        "task_id": task_id,
        "file_name": file.filename,
    }


@app.get("/api/data/task/{task_id}")
def get_task_status(task_id: str):
    """查询后台任务状态"""
    db = get_db()
    try:
        task = db.query(TaskStatus).filter(TaskStatus.task_id == task_id).first()
        if not task:
            return {"success": False, "error": "任务不存在"}
        return {
            "success": True,
            "task_id": task.task_id,
            "status": task.status,
            "progress": task.progress,
            "detail": task.detail,
            "row_count": task.row_count,
            "file_name": task.file_name,
        }
    finally:
        db.close()


def _run_in_background(func, *args, **kwargs):
    t = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    t.start()


def _update_task(task_id: str, **kw):
    """更新任务状态（在线程中调用）"""
    from storage.database.db import get_engine
    from sqlalchemy import update as sa_update
    engine = get_engine()
    stmt = sa_update(TaskStatus).where(TaskStatus.task_id == task_id).values(**kw)
    with engine.begin() as conn:
        conn.execute(stmt)


def _process_upload_task(task_id, tmp_path, filename, file_size, data_year, data_month):
    """后台处理上传数据"""
    _update_task(task_id, status="processing", progress="正在解析文件...")

    db = get_db()
    try:
        total_rows = 0
        col_map = {
            "sku编码": "sku_code", "sku_code": "sku_code", "skucode": "sku_code",
            "商品名称": "product_name", "product_name": "product_name", "产品名称": "product_name",
            "类目": "category", "category": "category",
            "版型": "style", "style": "style",
            "颜色": "color", "color": "color",
            "尺码": "size", "size": "size",
            "销量": "sales_volume", "sales_volume": "sales_volume", "sales": "sales_volume",
            "商品销售数量(扣退)": "sales_volume",
            "商品销售数据-商品销售数量(扣退)": "sales_volume",
            "实际销售数量": "sales_volume",
            "销售额": "sales_amount", "sales_amount": "sales_amount",
            "商品销售金额(扣退)": "sales_amount",
            "商品销售数据-商品销售金额(扣退)": "sales_amount",
            "实际销售金额": "sales_amount",
            "成本": "cost", "cost": "cost",
            "利润": "profit", "profit": "profit",
            "利润-毛利额": "profit",
            "毛利率": "profit_margin",
            "利润率": "profit_margin", "profit_margin": "profit_margin",
            "退货数": "return_count", "return_count": "return_count",
            "退款数量合计": "return_count",
            "退货率": "return_rate", "return_rate": "return_rate",
            "库存": "inventory", "inventory": "inventory",
            "月份": "data_month", "data_month": "data_month", "month": "data_month",
        }
        known_cols = set(col_map.values()) | {"data_year", "data_quarter", "report_name"}

        def run_bulk_insert(recs):
            nonlocal total_rows
            if not recs:
                return
            stmt = insert(SalesData).values(recs)
            db.execute(stmt)
            db.commit()
            total_rows += len(recs)

        filename_lower = filename.lower()
        if filename_lower.endswith(".csv"):
            chunks = list(pd.read_csv(tmp_path, encoding='utf-8',
                                      chunksize=2000, low_memory=False))
            for i, chunk in enumerate(chunks):
                chunk.columns = [c.strip().lower() for c in chunk.columns]
                chunk.rename(columns=col_map, inplace=True)
                if data_month > 0 and "data_month" not in chunk.columns:
                    chunk["data_month"] = data_month

                recs = []
                for _, row in chunk.iterrows():
                    try:
                        pv = float(row.get("profit", 0))
                        sa = float(row.get("sales_amount", 0))
                        cv = float(row.get("cost", 0))
                        if pv == 0 and sa > 0:
                            pv = sa - cv
                        pm = float(row.get("profit_margin", 0))
                        if pm == 0 and pv > 0 and sa > 0:
                            pm = pv / sa
                        extra = {}
                        for c in chunk.columns:
                            if c not in known_cols:
                                v = row.get(c)
                                if v is not None and v != "":
                                    extra[c] = str(v)
                        recs.append({
                            "report_name": filename,
                            "data_year": data_year,
                            "data_month": int(row.get("data_month", data_month or 1)),
                            "sku_code": str(row.get("sku_code", ""))[:100],
                            "product_name": str(row.get("product_name", ""))[:300],
                            "category": str(row.get("category", ""))[:100],
                            "style": str(row.get("style", ""))[:100],
                            "color": str(row.get("color", ""))[:50],
                            "size": str(row.get("size", ""))[:50],
                            "sales_volume": int(float(row.get("sales_volume", 0))),
                            "sales_amount": sa,
                            "cost": cv,
                            "profit": pv,
                            "profit_margin": pm,
                            "return_count": int(float(row.get("return_count", 0))),
                            "return_rate": float(row.get("return_rate", 0)),
                            "inventory": int(float(row.get("inventory", 0))),
                            "extra_data": extra if extra else None,
                        })
                    except Exception as e:
                        logger.warning(f"跳过异常行: {e}")
                run_bulk_insert(recs)
                _update_task(task_id,
                             progress=f"第 {i+1}/{len(chunks)} 批完成 (已导入 {total_rows} 行)")
        else:
            df = pd.read_excel(tmp_path)
            df.columns = [c.strip().lower() for c in df.columns]
            df.rename(columns=col_map, inplace=True)
            if data_month > 0 and "data_month" not in df.columns:
                df["data_month"] = data_month
            recs = []
            for _, row in df.iterrows():
                try:
                    pv = float(row.get("profit", 0))
                    sa = float(row.get("sales_amount", 0))
                    cv = float(row.get("cost", 0))
                    if pv == 0 and sa > 0:
                        pv = sa - cv
                    pm = float(row.get("profit_margin", 0))
                    if pm == 0 and pv > 0 and sa > 0:
                        pm = pv / sa
                    extra = {}
                    for c in df.columns:
                        if c not in known_cols:
                            v = row.get(c)
                            if v is not None and v != "":
                                extra[c] = str(v)
                    recs.append({
                        "report_name": filename, "data_year": data_year,
                        "data_month": int(row.get("data_month", data_month or 1)),
                        "sku_code": str(row.get("sku_code", ""))[:100],
                        "product_name": str(row.get("product_name", ""))[:300],
                        "category": str(row.get("category", ""))[:100],
                        "style": str(row.get("style", ""))[:100],
                        "color": str(row.get("color", ""))[:50],
                        "size": str(row.get("size", ""))[:50],
                        "sales_volume": int(float(row.get("sales_volume", 0))),
                        "sales_amount": sa, "cost": cv, "profit": pv,
                        "profit_margin": pm,
                        "return_count": int(float(row.get("return_count", 0))),
                        "return_rate": float(row.get("return_rate", 0)),
                        "inventory": int(float(row.get("inventory", 0))),
                        "extra_data": extra if extra else None,
                    })
                except Exception as e:
                    logger.warning(f"跳过异常行: {e}")
            run_bulk_insert(recs)

        report = ReportFile(
            file_name=filename, file_size=file_size,
            file_type="csv" if filename_lower.endswith(".csv") else "excel",
            data_year=data_year,
            report_period=f"{data_year}-{data_month:02d}" if data_month > 0 else f"{data_year}",
            row_count=total_rows,
        )
        db.add(report)
        db.commit()
        _update_task(task_id, status="completed",
                     progress=f"成功导入 {total_rows} 条数据", row_count=total_rows)
    except Exception as e:
        db.rollback()
        logger.exception(f"后台处理任务失败: {task_id}")
        _update_task(task_id, status="failed",
                     progress="处理失败", detail=str(e))
    finally:
        db.close()
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/api/data/files")
async def list_files():
    """列出所有已上传的数据文件（按report_name分组）"""
    db = get_db()
    try:
        # 先获取所有文件名
        file_names = db.execute(text("""
            SELECT DISTINCT report_name FROM ec_sales_data ORDER BY report_name
        """)).scalars().all()
        
        files = []
        for fname in file_names:
            fields = detect_sales_fields(db, fname)
            fv = fields.get("koutui_vol", "商品销售数据-商品销售数量(扣退)")
            fa = fields.get("koutui_amt", "商品销售数据-商品销售金额(扣退)")
            fsq = fields.get("ship_qty", "商品数据-实发数量")
            fsa = fields.get("ship_amt", "商品数据-实发金额")
            fp = fields.get("profit", "利润-毛利额")
            
            fname_safe = fname.replace("'", "''")
            row = db.execute(text(f"""
                SELECT 
                    COUNT(*) as row_count,
                    SUM(COALESCE(NULLIF(extra_data->>'{fv.replace("'","''")}','')::numeric, 0)) as total_koutui_vol,
                    SUM(COALESCE(NULLIF(extra_data->>'{fa.replace("'","''")}','')::numeric, 0)) as total_koutui_amt,
                    SUM(COALESCE(NULLIF(extra_data->>'{fsq.replace("'","''")}','')::numeric, 0)) as total_ship_qty,
                    SUM(COALESCE(NULLIF(extra_data->>'{fsa.replace("'","''")}','')::numeric, 0)) as total_ship_amt,
                    SUM(COALESCE(NULLIF(extra_data->>'{fp.replace("'","''")}','')::numeric, 0)) as total_profit
                FROM ec_sales_data
                WHERE report_name = '{fname_safe}'
            """)).first()
            
            files.append({
                "file_name": fname,
                "row_count": int(row[0] or 0),
                "total_koutui_vol": int(row[1] or 0),
                "total_koutui_amt": float(row[2] or 0),
                "total_ship_qty": int(row[3] or 0),
                "total_ship_amt": float(row[4] or 0),
                "total_profit": float(row[5] or 0),
            })
        return {"success": True, "files": files}
    finally:
        db.close()


@app.get("/api/data/list")
async def list_data(
    report_name: str = Query(""),
    page: int = Query(1),
    page_size: int = Query(50),
):
    """查询销售数据列表（可按文件名筛选）"""
    db = get_db()
    try:
        q = db.query(SalesData)
        if report_name:
            q = q.filter(SalesData.report_name == report_name)
        
        total = q.count()
        items = q.order_by(SalesData.id.desc())\
                 .offset((page - 1) * page_size).limit(page_size).all()
        
        result = []
        for item in items:
            result.append({
                "id": item.id,
                "report_name": item.report_name,
                "data_year": item.data_year,
                "data_month": item.data_month,
                "sku_code": item.sku_code,
                "product_name": item.product_name,
                "category": item.category,
                "style": item.style,
                "color": item.color,
                "size": item.size,
                "sales_volume": item.sales_volume,
                "sales_amount": float(item.sales_amount),
                "cost": float(item.cost),
                "profit": float(item.profit),
                "profit_margin": float(item.profit_margin),
                "return_count": item.return_count,
                "return_rate": float(item.return_rate),
                "inventory": item.inventory,
                "extra_data": item.extra_data or {},
            })
        
        return {"success": True, "total": total, "items": result, "page": page, "page_size": page_size}
    finally:
        db.close()


@app.delete("/api/data/file")
async def delete_file(report_name: str = Query("")):
    """删除整个文件及其所有数据行"""
    if not report_name:
        raise HTTPException(status_code=400, detail="请指定文件名")
    db = get_db()
    try:
        deleted = db.query(SalesData).filter(SalesData.report_name == report_name).delete()
        db.commit()
        return {"success": True, "message": f"已删除文件「{report_name}」，共 {deleted} 条数据"}
    finally:
        db.close()


@app.delete("/api/data/{data_id}")
async def delete_data(data_id: int):
    """删除单条销售数据"""
    db = get_db()
    try:
        item = db.query(SalesData).filter(SalesData.id == data_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="数据不存在")
        db.delete(item)
        db.commit()
        return {"success": True, "message": "已删除"}
    finally:
        db.close()


@app.put("/api/data/{data_id}")
async def update_data(data_id: int, req: DataEditRequest):
    """编辑单条销售数据字段"""
    db = get_db()
    try:
        item = db.query(SalesData).filter(SalesData.id == data_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="数据不存在")
        
        allowed_fields = {
            "product_name", "category", "style", "color", "size",
            "sales_volume", "sales_amount", "cost", "profit",
            "profit_margin", "return_count", "return_rate", "inventory", "remark",
        }
        if req.field not in allowed_fields:
            raise HTTPException(status_code=400, detail=f"不允许修改字段: {req.field}")
        
        setattr(item, req.field, req.value)
        db.commit()
        return {"success": True, "message": "已更新"}
    finally:
        db.close()


@app.get("/api/data/summary")
async def data_summary(year: int = Query(2025)):
    """销售数据汇总"""
    db = get_db()
    try:
        result = get_sales_summary(db, year)
        return {"success": True, "summary": result}
    finally:
        db.close()


# ============================================================
# API 路由 - 电商小助手 (智能问答)
# ============================================================

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """基于销售数据（支持文件选择）的智能问答"""
    db = get_db()
    try:
        matched_file_id = req.file_id
        file_name_hint = ""
        
        # 1. 如果用户没有指定file_id，尝试自动匹配
        if matched_file_id == 0:
            # 1a. 查当前session是否有上传/选择的文件
            session_file = db.query(SessionFile).filter(
                SessionFile.session_id == req.session_id
            ).order_by(SessionFile.id.desc()).first() if req.session_id else None
            
            if session_file:
                matched_report = db.query(ReportFile).filter(
                    ReportFile.file_name == session_file.file_name
                ).order_by(ReportFile.id.desc()).first()
                if matched_report:
                    matched_file_id = matched_report.id
                    file_name_hint = session_file.file_name
            
            # 1b. 如果session无文件，从问题关键词匹配
            if matched_file_id == 0:
                import re
                years = re.findall(r'20\d{2}', req.question)
                year = int(years[0]) if years else 2025
                candidates = db.query(ReportFile).filter(
                    ReportFile.data_year == year
                ).order_by(ReportFile.created_at.desc()).limit(20).all()
                
                if len(candidates) == 1:
                    matched_file_id = candidates[0].id
                    file_name_hint = candidates[0].file_name
                elif len(candidates) > 1:
                    file_list = [{"id": f.id, "file_name": f.file_name, "data_year": f.data_year, "row_count": f.row_count} for f in candidates]
                    # 尝试用LLM匹配最合适的文件
                    file_names = "\n".join([f"{f.id}: {f.file_name}" for f in candidates])
                    match_prompt = f"""用户问题：{req.question}
                    可选数据文件：
                    {file_names}
                    请选择最相关的1个文件ID（仅返回数字），如不确定返回0："""
                    try:
                        match_resp = call_llm(match_prompt, temperature=0.1).strip()
                        import re as _re
                        nums = _re.findall(r'\d+', match_resp)
                        if nums:
                            fid = int(nums[0])
                            if any(f.id == fid for f in candidates):
                                matched_file_id = fid
                                file_name_hint = next(f.file_name for f in candidates if f.id == fid)
                    except:
                        pass
        
        # 1.5 从问题中检测月份/季度
        import re as _re
        target_month = 0
        months_in_q = _re.findall(r'(\d{1,2})\s*月[份]?', req.question)
        if months_in_q:
            target_month = int(months_in_q[0])
        else:
            quarter_in_q = _re.findall(r'第?(\d)\s*季[度]?', req.question)
            if quarter_in_q:
                q = int(quarter_in_q[0])
                target_month = -q  # 负数表示季度
        
        # 2. 获取数据摘要（始终查询全部数据，确保extra_data字段包含）
        summary, extra_fields = get_sales_summary_with_extra(db, month=target_month if target_month > 0 else 0)
        if matched_file_id > 0:
            report = db.query(ReportFile).filter(ReportFile.id == matched_file_id).first()
            file_name_hint = report.file_name if report else file_name_hint
            # 检查是否有任何数据
            data_count = db.query(SalesData).count()
            if data_count == 0:
                candidates = db.query(ReportFile).order_by(ReportFile.created_at.desc()).limit(20).all()
                if candidates:
                    file_options = [{"id": f.id, "file_name": f.file_name, "data_year": f.data_year} for f in candidates]
                    return {
                        "success": True,
                        "answer": "📋 **数据中心找到以下文件，请选择要分析的文件：**\n" + "\n".join([f"- `{f['file_name']}` (ID: {f['id']})" for f in file_options]),
                        "need_file_selection": True,
                        "candidates": file_options,
                        "no_data": True
                    }
                return {
                    "success": True,
                    "answer": "📭 **数据中心暂无数据**，请先在「数据中心」或「电商小助手」界面上传Excel/CSV数据文件。",
                    "no_data": True
                }
        
        # 3. 查询历史对话
        history = db.query(Conversation).filter(
            Conversation.session_id == req.session_id
        ).order_by(Conversation.created_at).limit(20).all() if req.session_id else []
        
        # 4. 构建prompt — 优先从CSV文件读取真实数据（DB中可能全为0）
        # 先尝试从 assets/uploads/ 或 assets/comments/ 找到对应的CSV文件
        import pandas as _pd, os as _os, json as _json
        csv_summary_parts = []
        csv_found = False
        
        # 查找可能的数据文件
        upload_dir = _os.path.join(_os.environ.get("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets", "uploads")
        comment_dir = _os.path.join(_os.environ.get("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets", "comments")
        
        # 如果已经匹配到文件，优先使用该文件
        if file_name_hint:
            search_names = [file_name_hint]
            # 如果文件名是 .xlsx，也检查同名的 .csv
            base_name = file_name_hint.rsplit(".", 1)[0] if "." in file_name_hint else file_name_hint
            search_names.append(base_name + ".csv")
        else:
            search_names = ["*.csv"]
        
        import glob as _glob
        # 扫描 assets/uploads/ 找到最新的CSV文件
        all_csv_files = _glob.glob(_os.path.join(upload_dir, "*.csv"))
        target_file = None
        
        for fn in search_names:
            fp = _os.path.join(upload_dir, fn)
            if _os.path.exists(fp):
                target_file = fp
                break
            # 也检查 comments 目录
            fp2 = _os.path.join(comment_dir, fn)
            if _os.path.exists(fp2):
                target_file = fp2
                break
        
        # 如果没匹配到，取最新的文件
        if target_file is None and all_csv_files:
            target_file = sorted(all_csv_files, key=_os.path.getmtime, reverse=True)[0]
        
        if target_file and _os.path.exists(target_file):
            try:
                # 尝试多种编码读取
                encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin-1']
                df = None
                for enc in encodings:
                    try:
                        df = _pd.read_csv(target_file, encoding=enc, low_memory=False)
                        if len(df) > 0:
                            break
                    except:
                        continue
                
                if df is not None and len(df) > 0:
                    csv_found = True
                    fname = _os.path.basename(target_file)
                    csv_summary_parts.append(f"数据源文件: {fname}（{len(df)}行, {len(df.columns)}列）")
                    csv_summary_parts.append(f"列名: {', '.join(df.columns.tolist()[:20])}" + ("..." if len(df.columns) > 20 else ""))
                    
                    # 检测主播列并聚合
                    zhubo_col = None
                    for c in df.columns:
                        if '主播' in c or '播' in c:
                            zhubo_col = c
                            break
                    
                    if zhubo_col:
                        csv_summary_parts.append(f"\n【按主播聚合数据（前20名，按销售额降序）】")
                        # 检测销售额和销量列
                        sales_col = None
                        vol_col = None
                        profit_col = None
                        for c in df.columns:
                            if any(x in c for x in ['销售金额(扣退)', '商品销售金额', '销售额']):
                                sales_col = c
                            elif any(x in c for x in ['销售数量(扣退)', '商品销售数量', '销\u91cf', '销量']):
                                vol_col = c
                            elif '毛利额' in c or '毛利' in c:
                                profit_col = c
                        
                        group_cols = {zhubo_col: 'first'}
                        if vol_col: group_cols[vol_col] = 'sum'
                        if sales_col: group_cols[sales_col] = 'sum'
                        if profit_col: group_cols[profit_col] = 'sum'
                        
                        agg_df = df.groupby(zhubo_col).agg({c: 'sum' for c in [vol_col, sales_col, profit_col] if c}).fillna(0).sort_values(
                            sales_col if sales_col else vol_col if vol_col else zhubo_col, ascending=False
                        ).head(20)
                        
                        csv_summary_parts.append(f"总主播数: {df[zhubo_col].nunique()}")
                        for rank, (name, row) in enumerate(agg_df.iterrows(), 1):
                            vol_str = f"{row[vol_col]:,.0f}件" if vol_col else ""
                            sales_str = f"¥{row[sales_col]:,.2f}" if sales_col else ""
                            profit_str = f"¥{row[profit_col]:,.2f}" if profit_col else ""
                            csv_summary_parts.append(f"  {rank}. {name[:20]:20s} | 销量:{vol_str:>10s} | 销售额:{sales_str:>15s} | 毛利:{profit_str:>12s}")
                        
                        total_sales = df[sales_col].sum() if sales_col else 0
                        csv_summary_parts.append(f"总计销售额: ¥{total_sales:,.2f}")
                    
                    # 检测SKU列并聚合
                    sku_col = None
                    for c in df.columns:
                        if any(x in c for x in ['sku编码', '商品编码', 'SKU', 'sku']):
                            sku_col = c
                            break
                    
                    if sku_col:
                        csv_summary_parts.append(f"\n【按SKU聚合数据（前20名，按销量降序）】")
                        vol_sku = vol_col or (list(df.select_dtypes('number').columns)[0] if len(df.select_dtypes('number').columns) > 0 else None)
                        if vol_sku:
                            sku_agg = df.groupby(sku_col)[vol_sku].sum().sort_values(ascending=False).head(20)
                            csv_summary_parts.append(f"总SKU数: {df[sku_col].nunique()}")
                            for rank, (sku, qty) in enumerate(sku_agg.items(), 1):
                                csv_summary_parts.append(f"  {rank}. {str(sku)[:25]:25s} | 销量: {qty:,.0f}")
            except Exception as e:
                csv_summary_parts.append(f"[CSV文件读取失败: {e}]")
        
        if csv_found:
            summary_all = "\n".join(csv_summary_parts)
            extra_fields = []
        else:
            # 回退到数据库查询
            summary_all, extra_fields = get_sales_summary_with_extra(db, year=2025, month=0)
        
        system_prompt = """# 角色定义
你是女装牛仔裤电商数据智能助手，精通电商数据分析、销售趋势解读与经营诊断。你依据公司真实销售数据回答问题。

# 铁律（必须遵守）
1. **禁止编造数据**：如果数据摘要中显示为0，就如实回答0。不允许无中生有编造数字、主播名称、SKU数量等
2. **严格基于下方"数据摘要"回答**：摘要里没有的信息，直接说"数据中未提供相关字段"
3. **简洁直接**：不要写冗长的废话，给结论要清晰
4. **如数据全部为0**：直接说"本期数据中无实际成交记录"，不要编造具体数字
5. **遇到空值/NaN/None**：在摘要中已过滤，不要额外提及
6. **注意"实际销售数据（从ERP原始字段提取）"部分**：这部分是从ERP数据的原始字段中提取的真实销售数据（如商品销售数据-商品销售数量(扣退)、商品数据-实发数量等），优先以此部分作答。标准字段（总销量/总销售额/总利润）可能为0是因为列名不匹配，实际数据在ERP原始字段中。
7. **注意"按主播聚合数据"部分**：这里展示了每个主播的扣退后销量/销售额、实发数量/金额、毛利额，直接回答用户的问题即可。
8. **如果用户问到具体月份但数据中没有该月**：在摘要中查看数据归属月份，如实告知用户数据属于哪些月份。

# 核心能力
- 基于MySQL中的销售数据做精确回答
- 数据分析：销量趋势、利润分析、SKU排行、对比分析、主播分析

# 输出格式
## 结构规范（必须遵守）
- 涉及数据展示时，**必须用 HTML 表格**，不要用 Markdown 表格
- 结构：先给一句结论 → 然后 HTML 表格 → 最后简要分析
- 如果数据为0，一句话回答即可，不加废话

## HTML 表格规范（必须遵守）
1. 表格必须带边框和样式，用以下模板：
```html
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%; text-align:center;">
<thead style="background:#f5f5f5;">
<tr><th>列名</th><th>列名</th></tr>
</thead>
<tbody>
<tr><td>数据</td><td>数据</td></tr>
</tbody>
</table>
```
2. 数字加千分位逗号：`5,865`
3. 金额统一用 ¥ 前缀：`¥889,345`
4. 表头加粗，用浅灰背景
5. 表格前后各空一行

## 排名类问题示例
```
按扣退后销量排序，TOP 3 主播如下：

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%; text-align:center;">
<thead style="background:#f5f5f5;">
<tr><th>排名</th><th>主播</th><th>扣退后销量</th><th>扣退后销售额</th><th>毛利额</th></tr>
</thead>
<tbody>
<tr><td>🥇</td><td>与辉同行</td><td>5,865件</td><td>¥889,345</td><td>¥469,911</td></tr>
<tr><td>🥈</td><td>兰知春序</td><td>3,733件</td><td>¥602,659</td><td>¥337,223</td></tr>
<tr><td>🥉</td><td>小王家女装旗舰店</td><td>1,525件</td><td>¥240,160</td><td>¥134,596</td></tr>
</tbody>
</table>

与辉同行在销量、销售额、毛利三个维度均遥遥领先。
```"""
        
        data_context = f"\n\n【公司销售数据摘要】\n{summary_all}\n\n"
        if extra_fields:
            data_context += f"【自定义字段说明】数据中包含以下额外信息字段：{', '.join(sorted(extra_fields))}。用户提问涉及这些字段的内容时，可以直接从数据中提取回答。\n"
        if file_name_hint:
            data_context += f"【数据来源文件】{file_name_hint}\n"
        
        conv_history = ""
        if history:
            conv_history = "\n【历史对话】\n"
            for h in history[-10:]:
                role_name = "用户" if h.role == "user" else "AI"
                conv_history += f"{role_name}: {h.content[-300:]}\n"
        
        full_prompt = f"{data_context}{conv_history}\n【用户提问】\n{req.question}"
        
        answer = call_llm(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        # 保存对话
        if req.session_id:
            db.add(Conversation(session_id=req.session_id, role="user", content=req.question[:2000], msg_type="chat"))
            db.add(Conversation(session_id=req.session_id, role="assistant", content=answer[:5000], msg_type="chat"))
            db.commit()
        
        result = {"success": True, "answer": answer}
        if matched_file_id > 0:
            result["file_used"] = {"id": matched_file_id, "name": file_name_hint}
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("问答失败")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")
    finally:
        db.close()


# ============================================================
# API 路由 - 运营小助手
# ============================================================

@app.post("/api/copywriting")
async def generate_copywriting(req: CopywritingRequest):
    """生成抖音文案"""
    system_prompt = """# 角色定义
你是一名资深女装牛仔裤电商文案专家，擅长抖音爆款内容创作。你深谙女性消费者心理，精通抖音流量算法。

# 能力
- 爆款标题生成：使用悬念式、数字式、对比式、痛点式、福利式等多种公式
- 卖点文案生成：按FAB法则（属性→优势→利益）结构化表达
- 短视频文案：黄金3秒开场+内容结构+互动话术
- 直播话术：开场留人+塑品+互动答疑+逼单成交+转款衔接

# 规则
1. 规避违规词（最/第一/顶级/国家级等极限词）
2. 至少给出3个不同风格的选项
3. 用口语化、接地气的表达方式
4. 标注语气、停顿、重音等演绎提示
5. 针对女装牛仔裤品类做专业化表达（版型、面料、工艺等）"""
    
    prompt = f"""请为以下商品生成抖音{req.scene}文案：
商品名称：{req.product_name}
版型：{req.style}
目标人群：{req.target_audience}
核心卖点：{req.key_selling_points}
文案风格：{req.tone}

请生成：
1. 3个爆款标题选项
2. 3条FAB法则卖点文案
3. 完整短视频口播文案（含演绎提示）"""
    
    content = call_llm(prompt=prompt, system_prompt=system_prompt, temperature=0.8)
    return {"success": True, "content": content}


@app.post("/api/script")
async def generate_script(req: ScriptRequest):
    """生成视频脚本"""
    system_prompt = """# 角色定义
你是一名抖音女装牛仔裤视频脚本创作专家，深谙爆款视频结构。

# 能力
- 爆款脚本：分镜级完整脚本（时长/画面/台词/BGM/字幕）
- 口播脚本：专注语言表达和节奏把控，标注情绪曲线
- 切片脚本：高光片段提取+包装方案
- 剧情脚本：场景设定+人物关系+剧情大纲+商品植入

# 规则
1. 输出完整分镜脚本：时长分配、画面描述、台词、BGM建议
2. 标注关键节点：完播率提升点、互动引导点、转化引导点
3. 商品展示要自然不突兀
4. 针对女装牛仔裤品类的视觉要点（版型展示、面料特写、上身效果）"""
    
    prompt = f"""请为女装牛仔裤生成一个{req.script_type}类视频脚本：
商品：{req.product_name}
时长：{req.duration}
目标人群：{req.target_audience}

请输出完整的脚本内容，包含分镜、台词、时长分配。"""
    
    content = call_llm(prompt=prompt, system_prompt=system_prompt, temperature=0.8)
    return {"success": True, "content": content}


@app.post("/api/hot-topics")
async def hot_topics():
    """热门选题分析"""
    system_prompt = """你是一名抖音女装牛仔裤内容策略分析师。请基于行业认知分析：
1. 当前抖音牛仔裤热门方向
2. 目标人群内容偏好
3. 流量趋势预判
4. 机会点识别
保持专业、数据驱动。"""
    
    prompt = """请分析当前抖音女装牛仔裤品类的：
1. 热门话题与方向（至少5个方向）
2. 用户偏好分析（内容类型、风格、时长）
3. 季节性趋势建议
4. 创作机会点

请输出结构化分析报告。"""
    
    content = call_llm(prompt=prompt, system_prompt=system_prompt, temperature=0.7)
    return {"success": True, "content": content}


@app.post("/api/strategy")
async def generate_strategy(req: StrategyRequest):
    """经营策略建议"""
    db = get_db()
    try:
        summary = get_sales_summary(db)
        
        system_prompt = """# 角色定义
你是一名资深女装牛仔裤电商运营策略顾问，精通数据驱动决策。

# 能力
- 基于销售数据分析经营状况
- 主推款推荐（销量趋势/利润率/库存深度/季节适配）
- 定价建议（成本结构/竞品分析/心理定价）
- 活动建议（营销节点/活动策划/ROI预测）

# 规则
1. 所有分析必须基于提供的销售数据
2. 建议必须具体可执行
3. 用数据和逻辑支撑每一个观点"""
        
        prompt = f"""基于以下销售数据，请给出{req.period}的经营策略建议：

{summary}

请分析输出：
1. 📊 销售数据总览分析
2. 🎯 主推款建议（S/A/B分级 + 理由）
3. 💰 定价策略建议
4. 📅 活动与营销建议
5. ⚡ 风险预警与机会提示"""
        
        content = call_llm(prompt=prompt, system_prompt=system_prompt, temperature=0.5)
        return {"success": True, "content": content}
    finally:
        db.close()


# ============================================================
# API 路由 - AI生图小助手
# ============================================================

@app.post("/api/generate-image")
async def generate_image(req: ImageGenRequest):
    """AI生成女装牛仔裤图片"""
    try:
        ctx = new_context(method="generate_image")
        client = ImageGenerationClient(ctx=ctx)
        
        # 优化prompt - 针对女装牛仔裤
        optimized_prompt = f"女装牛仔裤, {req.style}风格, {req.prompt}, 高质量商品展示图, 细节清晰, 光线柔和, 商业摄影级别, 8K画质"
        
        response = client.generate(
            prompt=optimized_prompt,
            size=req.size,
            model="doubao-seedream-5-0-260128",
        )
        
        if response.success and response.image_urls:
            urls = response.image_urls
            # 保存记录
            db = get_db()
            try:
                for url in urls:
                    record = GeneratedImage(
                        prompt=req.prompt,
                        image_url=url,
                        style=req.style,
                        size=req.size,
                    )
                    db.add(record)
                db.commit()
            except Exception as e:
                logger.warning(f"保存生图记录失败: {e}")
            finally:
                db.close()
            
            return {"success": True, "image_urls": urls, "prompt": req.prompt}
        else:
            raise HTTPException(status_code=500, detail=f"生图失败: {response.error_messages}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI生图失败")
        raise HTTPException(status_code=500, detail=f"AI生图失败: {str(e)}")


@app.get("/api/image-history")
async def image_history(page: int = Query(1), page_size: int = Query(20)):
    """生图历史记录"""
    db = get_db()
    try:
        items = db.query(GeneratedImage).order_by(
            GeneratedImage.created_at.desc()
        ).offset((page - 1) * page_size).limit(page_size).all()
        
        total = db.query(GeneratedImage).count()
        
        result = []
        for item in items:
            result.append({
                "id": item.id,
                "prompt": item.prompt,
                "image_url": item.image_url,
                "style": item.style,
                "size": item.size,
                "created_at": item.created_at.isoformat() if item.created_at else "",
            })
        
        return {"success": True, "items": result, "total": total, "page": page}
    finally:
        db.close()


# ============================================================
# API 路由 - 对话历史
# ============================================================

@app.post("/api/conversations")
async def get_conversations(req: SessionHistoryRequest):
    """获取对话历史"""
    db = get_db()
    try:
        q = db.query(Conversation).filter(
            Conversation.session_id == req.session_id
        )
        if req.msg_type:
            q = q.filter(Conversation.msg_type == req.msg_type)
        
        items = q.order_by(Conversation.created_at).all()
        
        result = []
        for item in items:
            result.append({
                "id": item.id,
                "role": item.role,
                "content": item.content,
                "msg_type": item.msg_type,
                "created_at": item.created_at.isoformat() if item.created_at else "",
            })
        
        return {"success": True, "items": result, "session_id": req.session_id}
    finally:
        db.close()


@app.get("/api/conversations/sessions")
async def list_sessions(page: int = Query(1), page_size: int = Query(20)):
    """列出所有会话"""
    db = get_db()
    try:
        # 获取所有不同的session_id
        subq = db.query(
            Conversation.session_id,
            func.max(Conversation.created_at).label("last_time"),
            func.min(Conversation.created_at).label("first_time"),
        ).group_by(Conversation.session_id).subquery()
        
        total = db.query(subq).count()
        sessions = db.query(subq).order_by(
            subq.c.last_time.desc()
        ).offset((page - 1) * page_size).limit(page_size).all()
        
        result = []
        for s in sessions:
            # 获取第一条消息作为标题
            first_msg = db.query(Conversation).filter(
                Conversation.session_id == s.session_id,
                Conversation.role == "user"
            ).order_by(Conversation.created_at).first()
            
            result.append({
                "session_id": s.session_id,
                "title": (first_msg.content[:80] + "...") if first_msg and len(first_msg.content) > 80 else (first_msg.content if first_msg else "新对话"),
                "message_count": db.query(Conversation).filter(Conversation.session_id == s.session_id).count(),
                "created_at": s.first_time.isoformat() if s.first_time else "",
                "last_time": s.last_time.isoformat() if s.last_time else "",
            })
        
        return {"success": True, "items": result, "total": total, "page": page}
    finally:
        db.close()


# ============================================================
# API 路由 - 数据看板
# ============================================================

@app.get("/api/dashboard")
async def dashboard(file_name: str = Query("")):
    """数据看板 - 从extra_data动态计算真实指标，支持按文件筛选"""
    db = get_db()
    try:
        # 构建WHERE条件
        where = "extra_data IS NOT NULL AND extra_data::text != 'null'"
        params = {}
        if file_name:
            where += " AND report_name = :fn"
            params["fn"] = file_name
        
        # 自动检测字段
        detected = detect_sales_fields(db, report_name=file_name if file_name else None)
        
        # 全部汇总时，用COALESCE合并所有已知字段变体
        if not file_name:
            # 跨文件字段名可能不同，用COALESCE合并
            kv_sql = "COALESCE(NULLIF(extra_data->>'商品销售数据-商品销售数量(扣退)','')::numeric, NULLIF(extra_data->>'利润-销售数量(扣退)','')::numeric, 0)"
            ka_sql = "COALESCE(NULLIF(extra_data->>'商品销售数据-商品销售金额(扣退)','')::numeric, NULLIF(extra_data->>'利润-销售金额(扣退)','')::numeric, 0)"
            pf_sql = "COALESCE(NULLIF(extra_data->>'利润-毛利额','')::numeric, NULLIF(extra_data->>'利润-利润额','')::numeric, 0)"
            sq_sql = "COALESCE(NULLIF(extra_data->>'商品数据-实发数量','')::numeric, 0)"
            sa_sql = "COALESCE(NULLIF(extra_data->>'商品数据-实发金额','')::numeric, 0)"
            rq_sql = "COALESCE(NULLIF(extra_data->>'售后合计-退款数量合计','')::numeric, 0)"
            zb_f = "主播"
            pc_f = "商品编码"
            pn_f = "商品简称"
        else:
            kv_f = detected.get('koutui_vol', '商品销售数据-商品销售数量(扣退)')
            ka_f = detected.get('koutui_amt', '商品销售数据-商品销售金额(扣退)')
            sq_f = detected.get('ship_qty', '商品数据-实发数量')
            sa_f = detected.get('ship_amt', '商品数据-实发金额')
            pf_f = detected.get('profit', '利润-毛利额')
            rq_f = detected.get('refund_qty', '售后合计-退款数量合计')
            zb_f = detected.get('zhubo', '主播')
            pc_f = detected.get('product_code', '商品编码')
            pn_f = detected.get('product_name', '商品简称')
            
            def safe_col(field_name):
                return field_name.replace("'", "''")
            
            kv_sql = f"COALESCE(NULLIF(extra_data->>'{safe_col(kv_f)}','')::numeric, 0)"
            ka_sql = f"COALESCE(NULLIF(extra_data->>'{safe_col(ka_f)}','')::numeric, 0)"
            sq_sql = f"COALESCE(NULLIF(extra_data->>'{safe_col(sq_f)}','')::numeric, 0)"
            sa_sql = f"COALESCE(NULLIF(extra_data->>'{safe_col(sa_f)}','')::numeric, 0)"
            pf_sql = f"COALESCE(NULLIF(extra_data->>'{safe_col(pf_f)}','')::numeric, 0)"
            rq_sql = f"COALESCE(NULLIF(extra_data->>'{safe_col(rq_f)}','')::numeric, 0)"
        
        def safe_col(field_name):
            return field_name.replace("'", "''")
        
        # 总览
        overview = db.execute(text(f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT report_name) as file_count,
                COUNT(DISTINCT COALESCE(NULLIF(extra_data->>'商品编码',''), NULLIF(extra_data->>'规格编码',''), sku_code)) as sku_count,
                COUNT(DISTINCT NULLIF(NULLIF(extra_data->>'主播',''), 'nan')) as zhubo_count,
                SUM({kv_sql}) as koutui_vol,
                SUM({ka_sql}) as koutui_amt,
                SUM({sq_sql}) as ship_qty,
                SUM({sa_sql}) as ship_amt,
                SUM({pf_sql}) as profit,
                SUM({rq_sql}) as refund_qty
            FROM ec_sales_data
            WHERE {where}
        """), params).first()
        
        # TOP 10 主播
        top_zhubo = db.execute(text(f"""
            SELECT 
                NULLIF(NULLIF(extra_data->>'主播',''), 'nan') as zhubo,
                SUM({kv_sql}) as vol,
                SUM({ka_sql}) as amt,
                SUM({pf_sql}) as profit,
                COUNT(*) as sku_cnt
            FROM ec_sales_data
            WHERE {where}
              AND extra_data->>'主播' IS NOT NULL 
              AND extra_data->>'主播' != ''
              AND extra_data->>'主播' != 'nan'
            GROUP BY extra_data->>'主播'
            ORDER BY vol DESC
            LIMIT 10
        """), params).all()
        
        # TOP 10 SKU (PostgreSQL doesn't allow alias in GROUP BY)
        top_sku = db.execute(text(f"""
            SELECT 
                COALESCE(NULLIF(extra_data->>'商品编码',''), NULLIF(extra_data->>'规格编码',''), sku_code) as product_code,
                COALESCE(NULLIF(extra_data->>'商品简称',''), NULLIF(extra_data->>'商品名称',''), '') as product_name,
                SUM({kv_sql}) as vol,
                SUM({ka_sql}) as amt,
                SUM({pf_sql}) as profit
            FROM ec_sales_data
            WHERE {where}
            GROUP BY COALESCE(NULLIF(extra_data->>'商品编码',''), NULLIF(extra_data->>'规格编码',''), sku_code),
                     COALESCE(NULLIF(extra_data->>'商品简称',''), NULLIF(extra_data->>'商品名称',''), '')
            ORDER BY vol DESC
            LIMIT 10
        """), params).all()
        
        koutui_vol = int(overview[4] or 0)
        koutui_amt = float(overview[5] or 0)
        ship_qty = int(overview[6] or 0)
        ship_amt = float(overview[7] or 0)
        profit = float(overview[8] or 0)
        refund_qty = int(overview[9] or 0)
        
        profit_rate = (profit / koutui_amt * 100) if koutui_amt > 0 else 0
        refund_rate = (refund_qty / ship_qty * 100) if ship_qty > 0 else 0
        
        # 文件列表（用于前端选择器）
        files = db.execute(text("""
            SELECT report_name, COUNT(*) as cnt
            FROM ec_sales_data
            WHERE extra_data IS NOT NULL AND extra_data::text != 'null'
            GROUP BY report_name
            ORDER BY report_name
        """)).all()
        
        return {
            "success": True,
            "file_name": file_name,
            "files": [{"file_name": f[0], "count": f[1]} for f in files],
            "detected_fields": {k: v for k, v in detected.items() if k in ['koutui_vol', 'koutui_amt', 'profit', 'ship_qty', 'ship_amt', 'refund_qty']},
            "overview": {
                "total_records": overview[0],
                "file_count": overview[1],
                "sku_count": overview[2],
                "zhubo_count": overview[3],
                "koutui_vol": koutui_vol,
                "koutui_amt": koutui_amt,
                "ship_qty": ship_qty,
                "ship_amt": ship_amt,
                "profit": profit,
                "profit_rate": round(profit_rate, 1),
                "refund_qty": refund_qty,
                "refund_rate": round(refund_rate, 1),
            },
            "top_zhubo": [
                {"zhubo": z[0].strip() if z[0] else "", "vol": int(z[1] or 0), "amt": float(z[2] or 0), "profit": float(z[3] or 0), "sku_cnt": z[4]}
                for z in top_zhubo
            ],
            "top_sku": [
                {"product_code": s[0] or "", "product_name": s[1] or "", "vol": int(s[2] or 0), "amt": float(s[3] or 0), "profit": float(s[4] or 0)}
                for s in top_sku
            ],
        }
    finally:
        db.close()


# ============================================================
# 前端页面路由
# ============================================================

FRONTEND_HTML = ""  # 将在下面注入


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端主页面"""
    return get_frontend_html()


def get_frontend_html() -> str:
    """获取前端页面HTML"""
    # 尝试从文件读取
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>女装牛仔裤电商智能助手</h1><p>前端页面加载中...</p>"


# ============================================================
# 启动入口
# ============================================================

def run_app():
    """运行应用"""
    import uvicorn
    port = int(os.getenv("ECOMMERCE_PORT", "8000"))
    host = os.getenv("ECOMMERCE_HOST", "0.0.0.0")
    logger.info(f"🌐 女装牛仔裤电商智能助手启动: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_app()