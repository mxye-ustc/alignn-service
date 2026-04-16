# AutoDL RTX PRO 6000 ALIGNN 预测指南

## 目录
1. [前期准备 (本地 Mac)](#1-前期准备-本地-mac)
2. [租用 AutoDL 实例](#2-租用-autodl-实例)
3. [配置环境](#3-配置环境)
4. [上传数据](#4-上传数据)
5. [运行预测](#5-运行预测)
6. [下载结果](#6-下载结果)

---

## 1. 前期准备 (本地 Mac)

### 1.1 打包数据

```bash
cd /Users/mxye/Myprojects/alignn

# 运行打包脚本
bash pack_data.sh
```

这会创建一个约 **~500MB** 的压缩包 `alignn_lfp_prediction.tar.gz`。

### 1.2 文件结构

打包后包含：
```
alignn_lfp_prediction/
├── alignn/
│   └── models/
│       ├── ALIGNN models on JARVIS-DFT dataset/   (6个模型)
│       └── ALIGNN models on MP dataset/            (2个模型)
├── lfp_dopant_configs_v4/
│   ├── poscar_files/                               (600个构型)
│   └── metadata/
│       └── doping_library_metadata.json
├── scripts/
│   └── predict_lfp_gpu.py
└── environment.yml
```

### 1.3 上传到公网网盘

推荐使用 AutoDL 公网网盘：

1. 注册 AutoDL 账号: https://www.autodl.com
2. 登录后进入「公网网盘」
3. 上传 `alignn_lfp_prediction.tar.gz`
4. 复制下载链接

---

## 2. 租用 AutoDL 实例

### 2.1 选择 GPU

1. 进入 AutoDL 控制台: https://www.autodl.com
2. 点击「租用实例」
3. 选择 **RTX PRO 6000** (96GB)
4. 选择合适的镜像

### 2.2 推荐配置

| 配置项 | 推荐 |
|--------|------|
| GPU | RTX PRO 6000 96GB |
| 镜像 | PyTorch 2.2 + CUDA 12.1 |
| 系统盘 | 50GB |
| 数据盘 | 100GB |

### 2.3 基础镜像选择

选择包含以下内容的镜像：
- Python 3.10
- PyTorch 2.2+
- CUDA 12.1+
- Conda

---

## 3. 配置环境

### 3.1 SSH 连接

```bash
# 获取实例信息
# 登录信息在 AutoDL 控制台「我的实例」中查看

ssh root@你的机器IP
```

### 3.2 创建 Conda 环境

```bash
# 复制环境文件到机器
# 方法1: 使用公网网盘下载
cd /root/data
wget 你的公网网盘链接
tar -xzvf alignn_lfp_prediction.tar.gz

# 创建 conda 环境
conda env create -f environment.yml -n lfp_doping

# 激活环境
conda activate lfp_doping
```

### 3.3 验证 GPU

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

预期输出：
```
PyTorch: 2.2.1
CUDA: True
GPU: NVIDIA RTX PRO 6000
```

---

## 4. 上传数据

### 4.1 方法一：公网网盘（推荐）

```bash
# 在机器上下载
cd /root/data
wget 你的公网网盘链接
tar -xzvf alignn_lfp_prediction.tar.gz
```

### 4.2 方法二：SCP 上传

```bash
# 在本地 Mac 上执行
cd /Users/mxye/Myprojects/alignn
scp alignn_lfp_prediction.tar.gz root@你的机器IP:/root/data/
```

### 4.3 验证文件

```bash
# 检查文件完整性
ls -lh /root/data/

# 验证 POSCAR 文件数量
ls /root/data/lfp_dopant_configs_v4/poscar_files/*.poscar | wc -l
# 预期: 600

# 验证模型文件
ls /root/data/alignn/models/ALIGNN\ models\ on\ JARVIS-DFT\ dataset/*/checkpoint_300.pt | wc -l
# 预期: 6
```

---

## 5. 运行预测

### 5.1 测试模式（推荐先运行）

先测试 3 个样本，验证一切正常：

```bash
cd /root/data
bash scripts/run_prediction.sh test
```

预期输出：
```
==============================================
  ALIGNN 批量预测 (GPU)
==============================================

[1/4] 检查 GPU...
  PyTorch: 2.2.1
  CUDA: True
  GPU: NVIDIA RTX PRO 6000

[2/4] 解压数据...
  数据已解压，跳过

[3/4] 检查文件...
  POSCAR 文件: 600 个
  JARVIS 模型: 6 个
  MP 模型: 2 个

[4/4] 开始预测 (模式: test)...
开始时间: 2026-04-02 18:00:00

[测试模式] 只处理前 3 个文件
找到 3 个 POSCAR 文件
预估时间: ~0.1 小时
```

测试成功后再运行完整预测！

### 5.2 完整预测

```bash
# 运行全部 8 个模型（600 构型）
# 预估时间: ~4-6 小时
bash scripts/run_prediction.sh all

# 或者分开运行
bash scripts/run_prediction.sh jarvis    # 只运行 JARVIS 模型 (6个)
bash scripts/run_prediction.sh mp         # 只运行 MP 模型 (2个)
```

### 5.3 后台运行（可选）

如果担心 SSH 断开连接：

```bash
# 使用 screen
screen -S alignn
bash scripts/run_prediction.sh all

# 按 Ctrl+A, D 离开 screen
# 重新连接后查看
screen -r alignn
```

### 5.4 nohup 后台运行

```bash
nohup bash scripts/run_prediction.sh all > prediction.log 2>&1 &

# 查看进度
tail -f prediction.log
```

---

## 6. 下载结果

### 6.1 完成后检查

```bash
# 在 AutoDL 机器上检查
ls -lh /root/data/lfp_dopant_configs_v4/predictions/
```

预期文件：
```
combined_predictions.csv
combined_predictions.json
jarvis_alignn_predictions.csv
jarvis_alignn_predictions.json
mp_alignn_predictions.csv
mp_alignn_predictions.json
jarvis_alignn_total_energy.csv
jarvis_alignn_formation_energy.csv
jarvis_alignn_ehull.csv
jarvis_alignn_bandgap.csv
jarvis_alignn_bulk_modulus.csv
jarvis_alignn_magmom.csv
mp_alignn_mp_formation_energy.csv
mp_alignn_mp_bandgap.csv
```

### 6.2 SCP 下载

在本地 Mac 上：

```bash
cd /Users/mxye/Myprojects/alignn

# 下载结果目录
scp -r root@你的机器IP:/root/data/lfp_dopant_configs_v4/predictions \
      ./lfp_dopant_configs_v4/

# 或使用下载脚本
bash download_results.sh root@你的机器IP
```

### 6.3 公网网盘

上传到公网网盘，然后在本地下载。

---

## 预估成本

| 项目 | 说明 |
|------|------|
| GPU | RTX PRO 6000 96GB |
| 时间 | ~4-6 小时 |
| 预估费用 | ~50-80 元（按量计费）|

---

## 常见问题

### Q1: 显存不足 (OOM)

如果遇到 CUDA OOM 错误：

```python
# 修改 predict_lfp_gpu.py 中的 build_graph 函数
# 减小 batch_size 或逐个处理
```

### Q2: SSH 断开连接

使用 screen 或 nohup 保持运行：
```bash
screen -S alignn
```

### Q3: 预测中断

结果会保存在每个模型完成后，可以断点续传。

### Q4: 如何查看进度？

```bash
# 实时查看日志
tail -f prediction_*.log
```

---

## 快速命令汇总

```bash
# 1. 连接机器
ssh root@你的机器IP

# 2. 激活环境
conda activate lfp_doping

# 3. 测试运行
bash scripts/run_prediction.sh test

# 4. 完整运行
bash scripts/run_prediction.sh all

# 5. 下载结果（本地 Mac）
bash download_results.sh root@你的机器IP
```

---

## 脚本说明

| 脚本 | 位置 | 说明 |
|------|------|------|
| `pack_data.sh` | 本地 Mac | 打包数据 |
| `run_prediction.sh` | AutoDL 机器 | 运行预测 |
| `download_results.sh` | 本地 Mac | 下载结果 |
| `predict_lfp_gpu.py` | 双方都有 | GPU 预测脚本 |

---

祝预测顺利！
