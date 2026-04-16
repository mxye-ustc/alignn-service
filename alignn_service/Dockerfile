# ALIGNN 预测服务 Dockerfile
# 基于 CUDA 环境（如果使用 GPU）

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-mpi-dev \
    libgl1-mlib \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/bin/python /usr/local/bin/python

# 设置 PATH
ENV PATH=/opt/conda/bin:$PATH

# 创建应用目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 安装 alignn 包
RUN pip install -e .

# 创建数据和模型目录
RUN mkdir -p /data/uploads /data/results /models

# 设置工作目录
WORKDIR /app

# 暴露端口
EXPOSE 8000 8501

# 默认命令
CMD ["python", "-m", "alignn_service.main"]
