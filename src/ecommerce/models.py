"""数据库模型 - 女装牛仔裤电商智能助手"""

import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, Text, DateTime, JSON,
    Boolean, Date, ForeignKey, Enum as SAEnum, Numeric, Index, text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, registry
from typing import Optional, List
import enum

# 使用独立的 registry 避免与项目中其他 Base 冲突
mapper_registry = registry()
Base = mapper_registry.generate_base()


# ---------- 用户表 (无权限，全员可用) ----------
class User(Base):
    __tablename__ = "ec_user"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), default="")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )


# ---------- 后台任务状态跟踪 ----------
class TaskStatus(Base):
    """后台处理任务状态"""
    __tablename__ = "ec_task_status"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="任务ID")
    file_name: Mapped[str] = mapped_column(String(500), default="", comment="文件名")
    status: Mapped[str] = mapped_column(String(20), default="pending", comment="pending/processing/completed/failed")
    progress: Mapped[str] = mapped_column(String(200), default="", comment="进度描述")
    detail: Mapped[str] = mapped_column(Text, default="", comment="详细信息/错误信息")
    row_count: Mapped[int] = mapped_column(Integer, default=0, comment="成功导入行数")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )


# ---------- 销售数据主表 ----------
class SalesData(Base):
    """销售数据 - 按月份/季度存储的牛仔裤销售明细"""
    __tablename__ = "ec_sales_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 数据归属
    report_name: Mapped[str] = mapped_column(String(200), default="", comment="报表名称/来源文件名")
    data_year: Mapped[int] = mapped_column(Integer, default=2025, comment="年份")
    data_month: Mapped[int] = mapped_column(Integer, default=1, comment="月份 1-12")
    data_quarter: Mapped[int] = mapped_column(Integer, default=1, comment="季度 1-4")

    # SKU / 商品信息
    sku_code: Mapped[str] = mapped_column(String(100), default="", comment="SKU编码")
    product_name: Mapped[str] = mapped_column(String(300), default="", comment="商品名称")
    category: Mapped[str] = mapped_column(String(100), default="", comment="类目(如:高腰直筒/阔腿/紧身)")
    style: Mapped[str] = mapped_column(String(100), default="", comment="版型")
    color: Mapped[str] = mapped_column(String(50), default="", comment="颜色")
    size: Mapped[str] = mapped_column(String(50), default="", comment="尺码")

    # 核心指标
    sales_volume: Mapped[int] = mapped_column(Integer, default=0, comment="销量")
    sales_amount: Mapped[float] = mapped_column(Numeric(12, 2), default=0.0, comment="销售额")
    cost: Mapped[float] = mapped_column(Numeric(12, 2), default=0.0, comment="成本")
    profit: Mapped[float] = mapped_column(Numeric(12, 2), default=0.0, comment="利润")
    profit_margin: Mapped[float] = mapped_column(Numeric(6, 4), default=0.0, comment="利润率")

    # 扩展
    return_count: Mapped[int] = mapped_column(Integer, default=0, comment="退货数")
    return_rate: Mapped[float] = mapped_column(Numeric(6, 4), default=0.0, comment="退货率")
    inventory: Mapped[int] = mapped_column(Integer, default=0, comment="库存")
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="额外字段")
    remark: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        Index("idx_ec_sales_month", "data_year", "data_month"),
        Index("idx_ec_sales_sku", "sku_code"),
    )


# ---------- 报表文件记录 ----------
class ReportFile(Base):
    """上传的报表文件记录"""
    __tablename__ = "ec_report_file"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(500), nullable=False, comment="原文件名")
    file_size: Mapped[int] = mapped_column(BigInteger, default=0, comment="文件大小(bytes)")
    file_type: Mapped[str] = mapped_column(String(50), default="", comment="文件类型 csv/excel")
    data_year: Mapped[int] = mapped_column(Integer, default=2025, comment="数据年份")
    report_period: Mapped[str] = mapped_column(String(50), default="", comment="报表期间 如 2025-09")
    row_count: Mapped[int] = mapped_column(Integer, default=0, comment="数据行数")
    status: Mapped[str] = mapped_column(String(20), default="imported", comment="状态")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )


# ---------- 对话历史 ----------
class Conversation(Base):
    """AI问答对话历史"""
    __tablename__ = "ec_conversation"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, default=0, comment="用户ID")
    session_id: Mapped[str] = mapped_column(String(100), default="", comment="会话ID")
    role: Mapped[str] = mapped_column(String(20), default="user", comment="user/assistant")
    content: Mapped[str] = mapped_column(Text, default="", comment="内容")
    msg_type: Mapped[str] = mapped_column(String(50), default="chat", comment="chat/copywriting/image/strategy")
    extra_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        Index("idx_ec_conv_session", "session_id"),
        Index("idx_ec_conv_user", "user_id"),
    )


# ---------- AI生图记录 ----------
class GeneratedImage(Base):
    """AI生图记录"""
    __tablename__ = "ec_generated_image"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, default=0)
    prompt: Mapped[str] = mapped_column(Text, default="", comment="生成提示词")
    image_url: Mapped[str] = mapped_column(Text, default="", comment="图片URL")
    style: Mapped[str] = mapped_column(String(100), default="", comment="风格标签")
    size: Mapped[str] = mapped_column(String(20), default="2K")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )


# ---------- 会话关联文件 (电商小助手用) ----------
class SessionFile(Base):
    """会话关联的上传文件 - 用于电商小助手按文件问答"""
    __tablename__ = "ec_session_file"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), default="", comment="会话ID", index=True)
    file_name: Mapped[str] = mapped_column(String(500), default="", comment="原文件名")
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    row_count: Mapped[int] = mapped_column(Integer, default=0, comment="数据行数")
    columns: Mapped[str] = mapped_column(Text, default="", comment="文件包含的列名(逗号分隔)")
    data_preview: Mapped[str] = mapped_column(Text, default="", comment="数据预览前20行(文本)")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        Index("idx_ec_sf_session", "session_id"),
    )


# ---------- 数据看板缓存 ----------
class DashboardCache(Base):
    """数据看板聚合缓存"""
    __tablename__ = "ec_dashboard_cache"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cache_key: Mapped[str] = mapped_column(String(100), unique=True, comment="缓存键")
    cache_data: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )