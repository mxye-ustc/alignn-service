#!/bin/bash
# ALIGNN Web 服务部署脚本

set -e

REMOTE_HOST="106.54.22.65"
REMOTE_USER="ubuntu"
SSH_KEY="~/.ssh/tencentcloud.pem"
REMOTE_PROJECT_DIR="/home/ubuntu/alignn_project"
LOCAL_PROJECT_DIR="/Users/mxye/Myprojects/alignn"

echo "=========================================="
echo "ALIGNN Web 服务部署脚本"
echo "=========================================="

# 测试 SSH 连接
echo "[1/6] 测试 SSH 连接..."
if ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "echo '连接成功'" 2>/dev/null; then
    echo "  ✅ SSH 连接成功"
else
    echo "  ❌ SSH 连接失败"
    exit 1
fi

# 同步 alignn_service 目录（核心后端服务）
echo "[2/6] 同步 alignn_service 核心文件..."
ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_PROJECT_DIR/alignn_service" 2>/dev/null

# 同步 alignn_service 目录结构
rsync -avz --delete \
    -e "ssh -i $SSH_KEY" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "$LOCAL_PROJECT_DIR/alignn_service/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_DIR/alignn_service/" 2>/dev/null || {

    # 如果 rsync 不可用，使用 scp 递归上传
    echo "  使用 scp 方式同步..."
    scp -i $SSH_KEY -r "$LOCAL_PROJECT_DIR/alignn_service/"* "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_DIR/alignn_service/" 2>/dev/null
}
echo "  ✅ alignn_service 同步完成"

# 同步 UI 文件（app.py 和前端资源）
echo "[3/6] 同步 Web UI 文件..."
scp -i $SSH_KEY "$LOCAL_PROJECT_DIR/alignn_service/ui/app.py" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_DIR/app.py" 2>/dev/null

# 同步前端静态文件
if [ -d "$LOCAL_PROJECT_DIR/alignn_service/ui/web_static" ]; then
    ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_PROJECT_DIR/web_static" 2>/dev/null
    scp -i $SSH_KEY -r "$LOCAL_PROJECT_DIR/alignn_service/ui/web_static/"* "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_DIR/web_static/" 2>/dev/null || true
fi

if [ -d "$LOCAL_PROJECT_DIR/alignn_service/ui/web_templates" ]; then
    ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_PROJECT_DIR/web_templates" 2>/dev/null
    scp -i $SSH_KEY -r "$LOCAL_PROJECT_DIR/alignn_service/ui/web_templates/"* "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_DIR/web_templates/" 2>/dev/null || true
fi
echo "  ✅ Web UI 文件同步完成"

# 检查并安装依赖
echo "[4/6] 检查 Python 依赖..."
ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST << 'ENDSSH' 2>/dev/null
cd /home/ubuntu/alignn_project

# 检查 streamlit 是否安装
if ! source alignn_env/bin/activate 2>/dev/null; then
    echo "  ⚠️ 虚拟环境不存在，需要创建"
    exit 1
fi

# 检查依赖
if ! pip show streamlit >/dev/null 2>&1; then
    echo "  📦 安装 streamlit 和其他依赖..."
    pip install streamlit requests fastapi uvicorn python-multipart pydantic pydantic-settings celery redis -q
    echo "  ✅ 依赖安装完成"
else
    echo "  ✅ 依赖已满足"
fi
ENDSSH

# 重启服务
echo "[5/6] 重启 Web 服务..."

# 停止现有进程
ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST << 'ENDSSH' 2>/dev/null
cd /home/ubuntu/alignn_project

# 停止现有进程
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "uvicorn.*main:app" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
sleep 2

# 启动 FastAPI 后端服务
source alignn_env/bin/activate
nohup uvicorn alignn_service.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
echo "  FastAPI PID: $!"
sleep 3

# 启动 Streamlit 前端
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true > web.log 2>&1 &
echo "  Streamlit PID: $!"
sleep 2

# 显示运行状态
echo "--- 运行中的进程 ---"
pgrep -fa "python" | grep -E "main.py|streamlit|uvicorn" || echo "  无相关进程"
ENDSSH

# 验证
echo "[6/6] 验证服务状态..."
sleep 3
API_STATUS=$(ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "curl -s http://localhost:8000/health 2>/dev/null | head -1" 2>/dev/null || echo "API_ERROR")
WEB_STATUS=$(ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "curl -s http://localhost:8501 2>/dev/null | head -1" 2>/dev/null || echo "WEB_ERROR")

echo "  API 服务: $API_STATUS"
echo "  Web 服务: $WEB_STATUS"

if [[ "$API_STATUS" == *"healthy"* ]] || [[ "$API_STATUS" == *"{"* ]]; then
    echo ""
    echo "=========================================="
    echo "✅ 部署成功！"
    echo "=========================================="
    echo "  - API 服务: http://106.54.22.65:8000"
    echo "  - Web 界面: http://106.54.22.65:8501"
    echo ""
else
    echo "⚠️  服务可能未完全启动，查看日志:"
    echo "   ssh ubuntu@106.54.22.65 'tail -f /home/ubuntu/alignn_project/api.log'"
    echo "   ssh ubuntu@106.54.22.65 'tail -f /home/ubuntu/alignn_project/web.log'"
fi
