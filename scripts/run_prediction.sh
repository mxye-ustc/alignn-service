#!/bin/bash
# 远程运行脚本 - 在 AutoDL 机器上运行
# 使用方法: bash run_prediction.sh [all|jarvis|mp|test]

set -e

echo "=============================================="
echo "  ALIGNN 批量预测 (GPU)"
echo "=============================================="

# 项目根目录（脚本位于 scripts/ 下时上一级为根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 设置 conda 环境
CONDA_ENV="lfp_doping"

# 检查 conda 环境是否存在
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo ""
    echo "[错误] conda 环境 '$CONDA_ENV' 不存在"
    echo "请先创建环境:"
    echo "  conda env create -f \$PROJECT_DIR/environment.yml -n $CONDA_ENV"
    exit 1
fi

# 检查 GPU
echo ""
echo "[1/4] 检查 GPU..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 解压数据（如果需要）
if [ -f "$PROJECT_DIR/alignn_lfp_prediction.tar.gz" ]; then
    echo ""
    echo "[2/4] 解压数据..."
    if [ ! -d "$PROJECT_DIR/alignn" ]; then
        tar -xzvf "$PROJECT_DIR/alignn_lfp_prediction.tar.gz" -C "$PROJECT_DIR/"
    else
        echo "  数据已解压，跳过"
    fi
fi

# 检查必要文件
echo ""
echo "[3/4] 检查文件..."
POSCAR_COUNT=$(ls "$PROJECT_DIR/lfp_dopant_configs_v4/poscar_files"/*.poscar 2>/dev/null | wc -l || echo "0")
JARVIS_COUNT=$(ls "$PROJECT_DIR/alignn/models/ALIGNN models on JARVIS-DFT dataset"/*/checkpoint_300.pt 2>/dev/null | wc -l || echo "0")
MP_COUNT=$(ls "$PROJECT_DIR/alignn/models/ALIGNN models on MP dataset"/*/checkpoint_300.pt 2>/dev/null | wc -l || echo "0")

echo "  POSCAR 文件: $POSCAR_COUNT 个"
echo "  JARVIS 模型: $JARVIS_COUNT 个"
echo "  MP 模型: $MP_COUNT 个"

if [ "$POSCAR_COUNT" -eq 0 ]; then
    echo "[错误] 未找到 POSCAR 文件!"
    exit 1
fi

# 运行预测
MODEL_FLAG="${1:-all}"
echo ""
echo "[4/4] 开始预测 (模式: $MODEL_FLAG)..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd "$PROJECT_DIR"

# 设置 nohup 日志
LOG_FILE="prediction_$(date '+%Y%m%d_%H%M%S').log"

# 根据参数选择模型
case "$MODEL_FLAG" in
    test)
        echo "测试模式 (3个样本)..."
        conda run -n "$CONDA_ENV" python scripts/predict_lfp_gpu.py --limit 3 2>&1 | tee "$LOG_FILE"
        ;;
    jarvis)
        echo "JARVIS-DFT 模型 (6个)..."
        conda run -n "$CONDA_ENV" python scripts/predict_lfp_gpu.py --models jarvis 2>&1 | tee "$LOG_FILE"
        ;;
    mp)
        echo "MP 模型 (2个)..."
        conda run -n "$CONDA_ENV" python scripts/predict_lfp_gpu.py --models mp 2>&1 | tee "$LOG_FILE"
        ;;
    all|*)
        echo "全部模型 (8个)..."
        conda run -n "$CONDA_ENV" python scripts/predict_lfp_gpu.py --models all 2>&1 | tee "$LOG_FILE"
        ;;
esac

echo ""
echo "=============================================="
echo "  预测完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志: $LOG_FILE"
echo "=============================================="

# 显示结果文件
echo ""
echo "结果文件:"
ls -lh "$PROJECT_DIR/lfp_dopant_configs_v4/predictions/"*.csv 2>/dev/null | tail -10
