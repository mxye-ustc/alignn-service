# ALIGNN 预测服务部署指南

## 目录

- [服务器准备](#服务器准备)
- [腾讯云 CBS 云盘配置](#腾讯云-cbs-云盘配置)
- [快速部署](#快速部署)
- [详细配置](#详细配置)
- [验证部署](#验证部署)
- [常用操作](#常用操作)

---

## 服务器准备

### 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04 LTS |
| CPU | 4 核 |
| 内存 | 4 GB |
| 系统盘 | 40 GB |
| 数据盘 | CBS 云盘 50GB（必须） |
| 网络 | 公网 IP |

### 基础环境配置

#### 1. 更新系统

```bash
sudo apt update && sudo apt upgrade -y
```

#### 2. 安装 Docker

```bash
# 安装依赖
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# 添加 Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 添加 Docker 仓库
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker

# 添加当前用户到 docker 组（避免每次 sudo）
sudo usermod -aG docker $USER
```

#### 3. 安装 Docker Compose

```bash
sudo apt install -y docker-compose
```

#### 4. 配置 CBS 云盘（见下一节）

---

## 腾讯云 CBS 云盘配置

CBS 云盘用于存储模型文件（约 2GB）。

### 1. 创建 CBS 云盘

在腾讯云控制台：
1. 进入 **云硬盘** → **创建云硬盘**
2. 选择：
   - 可用区：与 CVM 相同
   - 容量：50GB
   - 类型：高性能云硬盘（或按需选择）
3. 关联到 CVM 实例

### 2. 挂载云盘

```bash
# 查看云盘设备名
sudo fdisk -l

# 假设云盘设备名为 /dev/vdb，格式化（如果是新盘）
sudo mkfs.ext4 /dev/vdb

# 创建挂载点
sudo mkdir -p /mnt/cbs

# 挂载
sudo mount /dev/vdb /mnt/cbs

# 创建模型目录
sudo mkdir -p /mnt/cbs/models

# 设置权限
sudo chown -R $USER:$USER /mnt/cbs
```

### 3. 设置开机自动挂载

```bash
# 获取云盘 UUID
sudo blkid /dev/vdb

# 编辑 fstab
sudo vim /etc/fstab
```

添加以下行（替换 UUID）：

```
UUID=<YOUR_UUID> /mnt/cbs ext4 defaults,nofail 0 2
```

### 4. 创建模型存储目录

```bash
mkdir -p /mnt/cbs/models
```

---

## 快速部署

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd alignn/alignn_service

# 2. 配置环境变量
cp .env.example .env
vim .env  # 编辑配置

# 3. 启动服务
docker-compose up -d

# 4. 查看服务状态
docker-compose ps
```

### 方式二：手动部署

```bash
# 1. 安装 Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
rm Miniconda3-latest-Linux-x86_64.sh

# 2. 创建环境
conda env create -f environment.yml
conda activate alignn-service

# 3. 安装服务
pip install -e .

# 4. 配置环境变量
cp .env.example .env
vim .env

# 5. 启动 Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 6. 启动 API
uvicorn alignn_service.main:app --host 0.0.0.0 --port 8000 &

# 7. 启动 Worker
celery -A alignn_service.core.tasks celery_app worker --loglevel=info &
```

---

## 详细配置

### 环境变量配置

创建 `.env` 文件：

```bash
# 应用配置
APP_NAME="ALIGNN 预测服务"
DEBUG=false

# Redis 配置
REDIS_URL=redis://localhost:6379/0

# 数据库配置
DATABASE_URL=sqlite:///./alignn_service.db

# 模型目录（云盘挂载点）
MODEL_BASE_DIR=/mnt/cbs/models

# 上传文件目录
UPLOAD_DIR=/data/uploads
RESULTS_DIR=/data/results

# 安全性
SECRET_KEY=your-super-secret-key-change-this

# 预测配置
MAX_CONCURRENT_PREDICTIONS=1
PREDICTION_TIMEOUT=300
```

### Docker Compose 启动参数

```bash
# 启动所有服务（包括 Web UI）
docker-compose up -d

# 只启动 API 和 Worker
docker-compose up -d api worker redis

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f api
```

---

## 验证部署

### 1. 检查服务状态

```bash
# 检查容器状态
docker-compose ps

# 检查 API 健康
curl http://localhost:8000/health
```

### 2. 访问服务

| 服务 | 地址 |
|------|------|
| API 文档 | http://your-ip:8000/docs |
| Web UI | http://your-ip:8501 |
| API 健康检查 | http://your-ip:8000/health |

### 3. 测试预测功能

使用 curl 测试：

```bash
# 提交预测任务
curl -X POST "http://localhost:8000/api/v1/predict/async" \
  -F "file=@/path/to/structure.poscar" \
  -F "models=jv_formation_energy_peratom_alignn,jv_optb88vdw_bandgap_alignn"

# 查询任务状态（替换 TASK_ID）
curl http://localhost:8000/api/v1/tasks/{TASK_ID}

# 获取结果（替换 TASK_ID）
curl http://localhost:8000/api/v1/tasks/{TASK_ID}/result
```

---

## 常用操作

### 查看服务日志

```bash
# 实时查看所有日志
docker-compose logs -f

# 查看特定服务
docker-compose logs -f api
docker-compose logs -f worker

# 查看最近 100 行
docker-compose logs --tail=100
```

### 重启服务

```bash
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart api

# 重启 Worker
docker-compose restart worker
```

### 更新服务

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build

# 重启服务
docker-compose up -d
```

### 停止服务

```bash
# 停止所有服务（保留数据）
docker-compose stop

# 停止并删除容器
docker-compose down

# 删除所有数据卷（谨慎！）
docker-compose down -v
```

### 备份数据

```bash
# 备份结果
tar -czf results_backup.tar.gz data/results/

# 备份数据库
cp alignn_service.db alignn_service.db.backup
```

---

## 故障排查

### 服务无法启动

```bash
# 查看详细日志
docker-compose logs --tail=100

# 检查端口占用
sudo netstat -tlnp | grep 8000
sudo netstat -tlnp | grep 8501
```

### 内存不足

```bash
# 查看内存使用
free -h

# 查看 Docker 内存使用
docker stats
```

### 磁盘空间不足

```bash
# 查看磁盘使用
df -h

# 清理 Docker
docker system prune -a
```

### Worker 无法连接 Redis

```bash
# 检查 Redis
docker-compose exec redis redis-cli ping

# 应该返回 PONG
```

---

## 性能优化

### 针对 CPU 环境的优化

1. **限制并发数**：配置 `MAX_CONCURRENT_PREDICTIONS=1`
2. **增加 swap**：创建 swap 文件
   ```bash
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```
3. **优化内存**：Docker Compose 中设置内存限制

### 模型加载优化

首次使用时模型会自动从 Figshare 下载到 `MODEL_BASE_DIR`。下载完成后，后续使用会从本地加载，加快启动速度。

---

## 安全建议

1. **修改 SECRET_KEY**：使用强密码
2. **配置防火墙**：只开放 80/443 端口
3. **启用 HTTPS**：使用 Let's Encrypt 证书
4. **定期备份**：备份数据和配置
