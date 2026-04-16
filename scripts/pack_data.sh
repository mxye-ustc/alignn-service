#!/bin/bash
# 数据打包脚本 - 用于 AutoDL 传输
# 使用方法: bash pack_data.sh

set -e

echo "=============================================="
echo "  ALIGNN 预测数据打包"
echo "=============================================="

# 目标目录
ALIGNN_ROOT="/Users/mxye/Myprojects/alignn"
PACKAGE_NAME="alignn_lfp_prediction"

# 临时打包目录
TEMP_DIR="$ALIGNN_ROOT/${PACKAGE_NAME}_package"

echo ""
echo "[1/5] 创建临时打包目录..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR/alignn/models"
mkdir -p "$TEMP_DIR/lfp_dopant_configs_v4/poscar_files"
mkdir -p "$TEMP_DIR/lfp_dopant_configs_v4/metadata"
mkdir -p "$TEMP_DIR/scripts"

echo ""
echo "[2/5] 复制模型文件..."
# 复制 JARVIS 模型
cp -r "$ALIGNN_ROOT/alignn/models/ALIGNN models on JARVIS-DFT dataset" \
      "$TEMP_DIR/alignn/models/"

# 复制 MP 模型
cp -r "$ALIGNN_ROOT/alignn/models/ALIGNN models on MP dataset" \
      "$TEMP_DIR/alignn/models/"

echo "  JARVIS 模型: $(ls "$TEMP_DIR/alignn/models/ALIGNN models on JARVIS-DFT dataset" | wc -l) 个"
echo "  MP 模型: $(ls "$TEMP_DIR/alignn/models/ALIGNN models on MP dataset" | wc -l) 个"

echo ""
echo "[3/5] 复制构型文件..."
# 复制 POSCAR 文件
POSCAR_COUNT=$(ls "$ALIGNN_ROOT/lfp_dopant_configs_v4/poscar_files"/*.poscar 2>/dev/null | wc -l)
cp "$ALIGNN_ROOT/lfp_dopant_configs_v4/poscar_files"/*.poscar \
   "$TEMP_DIR/lfp_dopant_configs_v4/poscar_files/" 2>/dev/null || true

# 复制元数据
cp "$ALIGNN_ROOT/lfp_dopant_configs_v4/metadata/doping_library_metadata.json" \
   "$TEMP_DIR/lfp_dopant_configs_v4/metadata/" 2>/dev/null || true

echo "  POSCAR 文件: $POSCAR_COUNT 个"

echo ""
echo "[4/5] 复制脚本文件..."
# 复制预测与运行脚本
cp "$ALIGNN_ROOT/predict_lfp_gpu.py" "$TEMP_DIR/scripts/"
cp "$ALIGNN_ROOT/run_prediction.sh" "$TEMP_DIR/scripts/"
chmod +x "$TEMP_DIR/scripts/run_prediction.sh"

# 复制 requirements.txt
cp "$ALIGNN_ROOT/requirements.txt" "$TEMP_DIR/" 2>/dev/null || true

# 复制环境配置文件
if [ -f "$ALIGNN_ROOT/environment.yml" ]; then
    cp "$ALIGNN_ROOT/environment.yml" "$TEMP_DIR/"
fi

echo ""
echo "[5/5] 打包..."
cd "$ALIGNN_ROOT"

# 创建 tar.gz 包
tar -czvf "${PACKAGE_NAME}.tar.gz" -C "$TEMP_DIR" .

# 计算大小
PACKAGE_SIZE=$(du -h "${PACKAGE_NAME}.tar.gz" | cut -f1)

# 清理临时目录
rm -rf "$TEMP_DIR"

echo ""
echo "=============================================="
echo "  打包完成!"
echo "=============================================="
echo "  文件: ${PACKAGE_NAME}.tar.gz"
echo "  大小: $PACKAGE_SIZE"
echo "  路径: $ALIGNN_ROOT/${PACKAGE_NAME}.tar.gz"
echo ""
echo "  传输到 AutoDL:"
echo "    1. 使用公网网盘上传"
echo "    2. 或使用 SCP 上传:"
echo "       scp ${PACKAGE_NAME}.tar.gz root@你的机器ID.autodl.com:/root/data/"
echo ""
