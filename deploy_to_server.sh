#!/bin/bash
# =====================================================
# 电商看板系统 - 部署到云服务器
# =====================================================
# 使用方法：
# 1. 把整个项目目录上传到服务器
# 2. 运行本脚本
# 3. 访问 http://服务器IP:5000
# =====================================================

set -e

echo "=== 安装依赖 ==="
pip install fastapi uvicorn pandas openpyxl chardet python-multipart

echo "=== 启动服务 ==="
# 前台启动（调试用）
# python3 src/main.py

# 后台启动（生产用，关掉终端也不会停）
nohup python3 src/main.py > /tmp/ecommerce.log 2>&1 &

echo "=== 完成！==="
echo "访问地址: http://服务器IP:5000"
echo "登录账号: demo / demo123"
echo "日志文件: tail -f /tmp/ecommerce.log"
