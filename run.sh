#!/bin/bash
# ALIGNN 预测服务启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   ALIGNN 晶体性质预测服务启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 Python 版本
echo -e "${YELLOW}[1/5] 检查 Python 环境...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}✗ 需要 Python >= $REQUIRED_VERSION，当前版本: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"
echo ""

# 检查依赖
echo -e "${YELLOW}[2/5] 检查依赖包...${NC}"
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}  安装核心依赖...${NC}"
    pip install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || pip install torch
fi

if ! python3 -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}  安装 streamlit...${NC}"
    pip install streamlit
fi

if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}  安装 fastapi...${NC}"
    pip install fastapi uvicorn
fi

echo -e "${GREEN}✓ 依赖检查完成${NC}"
echo ""

# 创建必要目录
echo -e "${YELLOW}[3/5] 创建数据目录...${NC}"
mkdir -p "$PROJECT_ROOT/alignn_service/data/uploads"
mkdir -p "$PROJECT_ROOT/alignn_service/data/results"
mkdir -p "$PROJECT_ROOT/alignn_service/results"
mkdir -p "$PROJECT_ROOT/models"
echo -e "${GREEN}✓ 目录创建完成${NC}"
echo ""

# 解析命令行参数
SERVICE_TYPE="${1:-all}"

start_api() {
    echo -e "${BLUE}启动 API 服务 (端口 8000)...${NC}"
    cd "$PROJECT_ROOT"
    uvicorn alignn_service.main:app --host 0.0.0.0 --port 8000 --reload --app-dir "$PROJECT_ROOT" &
    API_PID=$!
    echo $API_PID > /tmp/alignn_api.pid
    echo -e "${GREEN}✓ API 服务已启动 (PID: $API_PID)${NC}"
}

start_worker() {
    echo -e "${BLUE}启动 Celery Worker...${NC}"
    cd "$PROJECT_ROOT"
    celery -A alignn_service.core.tasks celery_app worker --loglevel=info --concurrency=1 &
    WORKER_PID=$!
    echo $WORKER_PID > /tmp/alignn_worker.pid
    echo -e "${GREEN}✓ Celery Worker 已启动 (PID: $WORKER_PID)${NC}"
}

start_ui() {
    echo -e "${BLUE}启动 Streamlit UI (端口 8501)...${NC}"
    cd "$PROJECT_ROOT"
    streamlit run "$PROJECT_ROOT/alignn_service/ui/app.py" --server.port 8501 --server.headless true &
    UI_PID=$!
    echo $UI_PID > /tmp/alignn_ui.pid
    echo -e "${GREEN}✓ Streamlit UI 已启动 (PID: $UI_PID)${NC}"
}

# 启动服务
echo -e "${YELLOW}[4/5] 启动服务...${NC}"

case "$SERVICE_TYPE" in
    api)
        start_api
        ;;
    worker)
        start_worker
        ;;
    ui)
        start_ui
        ;;
    all)
        start_api
        sleep 2
        start_worker
        sleep 1
        start_ui
        ;;
    *)
        echo -e "${RED}未知的服务类型: $SERVICE_TYPE${NC}"
        echo "用法: ./run.sh [api|worker|ui|all]"
        exit 1
        ;;
esac

echo ""

# 完成
echo -e "${YELLOW}[5/5] 完成!${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   服务已启动${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "访问地址:"
echo -e "  • API 文档: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  • Streamlit: ${BLUE}http://localhost:8501${NC}"
echo ""
echo -e "停止服务: ./stop.sh"
echo ""

# 保持脚本运行（如果需要）
if [[ "$SERVICE_TYPE" == "all" ]]; then
    echo -e "${YELLOW}按 Ctrl+C 停止所有服务${NC}"
    wait
fi
