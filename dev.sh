#!/bin/bash
# ALIGNN 预测服务开发模式启动

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   ALIGNN 开发模式启动${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 .env 文件
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo -e "${YELLOW}警告: .env 文件不存在，创建默认配置...${NC}"
    cat > "$PROJECT_ROOT/.env" << EOF
# ALIGNN 服务配置
DEBUG=true
REDIS_URL=redis://localhost:6379/0
MODEL_BASE_DIR=$PROJECT_ROOT/models
UPLOAD_DIR=$PROJECT_ROOT/alignn_service/data/uploads
RESULTS_DIR=$PROJECT_ROOT/alignn_service/data/results
API_PORT=8000
UI_PORT=8501
LOG_LEVEL=DEBUG
EOF
    echo -e "${GREEN}✓ .env 文件已创建${NC}"
fi

# 创建必要目录
mkdir -p "$PROJECT_ROOT/alignn_service/data/uploads"
mkdir -p "$PROJECT_ROOT/alignn_service/data/results"
mkdir -p "$PROJECT_ROOT/alignn_service/results"
mkdir -p "$PROJECT_ROOT/models"

echo ""
echo -e "${GREEN}开发服务已准备就绪！${NC}"
echo ""
echo -e "启动命令:"
echo -e "  • API:  ${YELLOW}cd $PROJECT_ROOT && uvicorn alignn_service.main:app --reload${NC}"
echo -e "  • UI:   ${YELLOW}cd $PROJECT_ROOT && streamlit run alignn_service/ui/app.py --server.port 8501${NC}"
echo -e "  • Worker: ${YELLOW}cd $PROJECT_ROOT && celery -A alignn_service.core.tasks worker --loglevel=debug${NC}"
echo ""
