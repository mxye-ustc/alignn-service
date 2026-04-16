# ALIGNN 预测服务 API 文档

## 目录

- [概述](#概述)
- [认证](#认证)
- [API 端点](#api-端点)
- [使用示例](#使用示例)

---

## 概述

ALIGNN 预测服务提供 RESTful API，支持：
- 单结构和批量结构预测
- 多种晶体性质预测
- 异步任务处理

### 基础信息

| 项目 | 值 |
|------|------|
| 基础 URL | `http://your-server:8000` |
| API 版本 | v1 |
| 文档地址 | `http://your-server:8000/docs` |
| 格式 | JSON |

---

## 认证

当前版本暂未启用认证，生产环境建议添加 API Key 或 JWT 认证。

---

## API 端点

### 健康检查

```
GET /health
```

检查服务状态。

**响应示例：**

```json
{
  "status": "healthy",
  "service": "ALIGNN Prediction Service",
  "version": "1.0.0",
  "device": "cpu",
  "loaded_models": []
}
```

---

### 获取可用模型

```
GET /api/v1/models
```

获取所有可用预测模型。

**响应示例：**

```json
{
  "models": {
    "jv_formation_energy_peratom_alignn": {
      "name": "形成能 (JARVIS)",
      "description": "预测化合物的形成能",
      "unit": "eV/atom",
      "source": "JARVIS-DFT",
      "type": "energy"
    },
    "jv_optb88vdw_bandgap_alignn": {
      "name": "带隙 (OptB88vdW)",
      "description": "使用 OptB88vdW 泛函预测带隙",
      "unit": "eV",
      "source": "JARVIS-DFT",
      "type": "electronic"
    }
  },
  "total": 7,
  "local_models": []
}
```

---

### 单结构异步预测（推荐）

```
POST /api/v1/predict/async
```

提交单结构预测任务，立即返回任务 ID。

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | ✅ | 结构文件（POSCAR/CIF/XYZ/PDB） |
| models | string | ✅ | 模型名称，逗号分隔 |
| cutoff | float | 否 | 截断半径（默认 8.0 Å） |
| max_neighbors | int | 否 | 最大近邻数（默认 12） |

**响应示例：**

```json
{
  "task_id": "abc123-def456-ghi789",
  "status": "pending",
  "message": "任务已提交，请通过 /api/v1/tasks/{task_id} 查询状态"
}
```

---

### 单结构同步预测

```
POST /api/v1/predict/sync
```

同步预测，等待结果返回（不推荐 CPU 环境）。

**请求参数：** 同异步接口

**响应示例：**

```json
{
  "status": "success",
  "task_id": "...",
  "predictions": {
    "jv_formation_energy_peratom_alignn": {
      "value": -3.4521,
      "unit": "eV/atom",
      "processing_time": 45.23
    },
    "jv_optb88vdw_bandgap_alignn": {
      "value": 0.5823,
      "unit": "eV",
      "processing_time": 48.12
    }
  },
  "structure_info": {
    "formula": "LiFePO4",
    "n_atoms": 28,
    "n_elements": 4,
    "lattice": {
      "a": 10.34,
      "b": 6.01,
      "c": 4.69
    }
  },
  "total_models": 2,
  "successful": 2,
  "total_time_seconds": 93.35
}
```

---

### 批量预测

```
POST /api/v1/predict/batch
```

批量预测多个结构文件。

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | File[] | ✅ | 文件列表（最多 100 个） |
| models | string | ✅ | 模型名称，逗号分隔 |
| cutoff | float | 否 | 截断半径 |
| max_neighbors | int | 否 | 最大近邻数 |

**响应示例：**

```json
{
  "task_id": "batch-abc123-def456",
  "status": "pending",
  "total_files": 50,
  "message": "批量任务已提交，请通过 /api/v1/tasks/{task_id} 查询状态"
}
```

---

### 查询任务状态

```
GET /api/v1/tasks/{task_id}
```

查询异步任务状态。

**响应示例：**

```json
{
  "task_id": "abc123-def456",
  "status": "PROCESSING",
  "info": {
    "status": "processing",
    "message": "正在使用 jv_optb88vdw_bandgap_alignn 预测...",
    "progress": 0.65
  },
  "ready": false,
  "successful": null
}
```

**任务状态说明：**

| 状态 | 说明 |
|------|------|
| PENDING | 任务等待中 |
| PROCESSING | 正在处理 |
| SUCCESS | 已完成 |
| FAILURE | 失败 |

---

### 获取任务结果

```
GET /api/v1/tasks/{task_id}/result
```

获取任务执行结果。

**响应示例：**

```json
{
  "status": "success",
  "task_id": "abc123-def456",
  "predictions": {
    "jv_formation_energy_peratom_alignn": {
      "value": -3.4521,
      "unit": "eV/atom",
      "processing_time": 45.23
    }
  },
  "structure_info": {
    "formula": "LiFePO4",
    "n_atoms": 28
  },
  "completed_at": "2026-04-10T12:00:00"
}
```

---

## 使用示例

### Python 示例

```python
import requests

API_BASE = "http://localhost:8000"

# 1. 提交预测任务
files = {"file": open("structure.poscar", "rb")}
data = {"models": "jv_formation_energy_peratom_alignn,jv_optb88vdw_bandgap_alignn"}

response = requests.post(f"{API_BASE}/api/v1/predict/async", files=files, data=data)
task_id = response.json()["task_id"]

# 2. 轮询任务状态
import time
while True:
    status = requests.get(f"{API_BASE}/api/v1/tasks/{task_id}").json()
    if status["status"] == "SUCCESS":
        break
    time.sleep(2)

# 3. 获取结果
result = requests.get(f"{API_BASE}/api/v1/tasks/{task_id}/result").json()
print(result)
```

### curl 示例

```bash
# 提交任务
TASK_ID=$(curl -s -X POST "http://localhost:8000/api/v1/predict/async" \
  -F "file=@structure.poscar" \
  -F "models=jv_formation_energy_peratom_alignn,jv_optb88vdw_bandgap_alignn" \
  | jq -r '.task_id')

echo "Task ID: $TASK_ID"

# 轮询状态
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/v1/tasks/$TASK_ID" | jq -r '.status')
  echo "Status: $STATUS"
  if [ "$STATUS" == "SUCCESS" ]; then
    break
  fi
  sleep 2
done

# 获取结果
curl -s "http://localhost:8000/api/v1/tasks/$TASK_ID/result" | jq .
```

### JavaScript/Node.js 示例

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function predict(filePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  form.append('models', 'jv_formation_energy_peratom_alignn,jv_optb88vdw_bandgap_alignn');

  // 提交任务
  const { data: { task_id } } = await axios.post(
    'http://localhost:8000/api/v1/predict/async',
    form,
    { headers: form.getHeaders() }
  );

  console.log('Task ID:', task_id);

  // 轮询状态
  while (true) {
    const { data: status } = await axios.get(
      `http://localhost:8000/api/v1/tasks/${task_id}`
    );
    if (status.status === 'SUCCESS') break;
    await new Promise(r => setTimeout(r, 2000));
  }

  // 获取结果
  const { data: result } = await axios.get(
    `http://localhost:8000/api/v1/tasks/${task_id}/result`
  );
  console.log(result);
}

predict('structure.poscar');
```

---

## 错误处理

### 错误响应格式

```json
{
  "detail": "错误信息描述"
}
```

### 常见错误

| HTTP 状态码 | 说明 |
|------------|------|
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

---

## 速率限制

当前版本无速率限制。生产环境建议：
- 每个 IP 每分钟 60 次请求
- 每个用户每天 1000 次预测
