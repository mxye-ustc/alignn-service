#!/bin/bash
# ALIGNN 预测服务停止脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   ALIGNN 服务停止脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 停止各个服务
stop_service() {
    local name=$1
    local pid_file=$2

    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}停止 $name (PID: $pid)...${NC}"
            kill "$pid" 2>/dev/null || true
            rm -f "$pid_file"
            echo -e "${GREEN}✓ $name 已停止${NC}"
        else
            echo -e "${BLUE}$name 未在运行${NC}"
            rm -f "$pid_file"
        fi
    else
        echo -e "${BLUE}$name 未启动（无 PID 文件）${NC}"
    fi
}

# 停止服务
stop_service "API" "/tmp/alignn_api.pid"
stop_service "Worker" "/tmp/alignn_worker.pid"
stop_service "UI" "/tmp/alignn_ui.pid"

# 清理 Redis（如果有）
echo ""
echo -e "${BLUE}清理 Redis 缓存...${NC}"
redis-cli FLUSHALL 2>/dev/null || true

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   所有服务已停止${NC}"
echo -e "${GREEN}========================================${NC}"
