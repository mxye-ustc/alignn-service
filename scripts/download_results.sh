#!/bin/bash
# 结果下载脚本 - 在本地 Mac 上运行
# 使用方法: bash download_results.sh [AutoDL机器IP]

set -e

# 配置 - 请修改以下信息
AUTODL_HOST="${1:-}"           # 例如: root@123.456.789.012
LOCAL_DIR="/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/predictions"
REMOTE_DIR="/root/alignn_lfp_prediction/lfp_dopant_configs_v4/predictions"

# 检查参数
if [ -z "$AUTODL_HOST" ]; then
    echo "=============================================="
    echo "  结果下载脚本"
    echo "=============================================="
    echo ""
    echo "用法: bash download_results.sh <AutoDL机器IP>"
    echo ""
    echo "示例: bash download_results.sh root@123.456.789.012"
    echo ""
    echo "请提供 AutoDL 机器 IP，例如:"
    echo "  bash download_results.sh root@auto-dl-xxx.region.autodl.com"
    exit 1
fi

echo "=============================================="
echo "  从 AutoDL 下载结果"
echo "=============================================="
echo "  主机: $AUTODL_HOST"
echo "  本地: $LOCAL_DIR"
echo ""

# 确保本地目录存在
mkdir -p "$LOCAL_DIR"

# 下载结果文件
echo "[1/2] 下载预测结果..."
rsync -avP --include='*.csv' --include='*.json' --exclude='*' \
    "$AUTODL_HOST:$REMOTE_DIR/" "$LOCAL_DIR/" \
    || scp -r "$AUTODL_HOST:$REMOTE_DIR/"*.csv "$AUTODL_HOST:$REMOTE_DIR/"*.json "$LOCAL_DIR/" 2>/dev/null

# 显示下载的文件
echo ""
echo "[2/2] 已下载文件:"
echo ""
ls -lh "$LOCAL_DIR/"*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh "$LOCAL_DIR/"*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

# 检查完整性
CSV_COUNT=$(ls "$LOCAL_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "=============================================="
echo "  下载完成!"
echo "  CSV 文件: $CSV_COUNT 个"
echo "  本地目录: $LOCAL_DIR"
echo "=============================================="

# 可选：同步日志文件
echo ""
echo "是否下载日志文件? (y/n)"
read -r -t 10 response || response="n"
if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
    echo "下载日志..."
    mkdir -p "$LOCAL_DIR/../logs"
    scp "$AUTODL_HOST:/root/alignn_lfp_prediction/"*.log "$LOCAL_DIR/../logs/" 2>/dev/null || true
    echo "日志已下载到: $LOCAL_DIR/../logs/"
fi
