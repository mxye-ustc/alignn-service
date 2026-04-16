# ALIGNN晶体性质预测平台 - 系统设计文档

> 架构设计 | 技术选型 | 模块划分 | 接口定义

---

## 1. 设计目标

基于现有代码库，设计一套完整的晶体性质预测平台，满足 PRD v1.0 中定义的 8 个功能需求。

### 1.1 现有代码资产盘点

| 资产 | 状态 | 说明 |
|-----|------|------|
| `alignn_service/core/predictor.py` | ✅ 核心完成 | 结构解析、图构建、模型推理 |
| `alignn_service/core/model_manager.py` | ✅ 核心完成 | 模型加载、缓存、Figshare下载 |
| `alignn_service/core/tasks.py` | ✅ 核心完成 | Celery异步任务 |
| `alignn_service/core/doping_generator.py` | ✅ 新增完成 | 掺杂构型生成器模块 |
| `alignn_service/ui/app.py` | ✅ 已增强 | Streamlit UI，多页面导航 |
| `alignn_service/ui/components.py` | ✅ 新增完成 | Streamlit共享组件库 |
| `alignn_service/ui/pages/batch.py` | ✅ 新增完成 | 批量预测页面 |
| `alignn_service/ui/pages/doping.py` | ✅ 新增完成 | 自定义掺杂页面 |
| `alignn_service/ui/pages/compare.py` | ✅ 新增完成 | 数据库对比页面 |
| `alignn_service/ui/pages/history.py` | ✅ 新增完成 | 历史记录页面 |
| `alignn_service/utils/` | ✅ 新增完成 | 文件解析、验证、导出工具 |
| `alignn_service/api/routes/` | ✅ 新增完成 | 模块化API路由 |
| `generate_lfp_dopants_v4.py` | ✅ 完成 | 掺杂构型生成器 |
| `lfp_universe_app.py` | ✅ 完成 | Dash可视化（分子宇宙地图） |
| 601个预生成掺杂构型 | ✅ 完成 | 已存在于 `lfp_dopant_configs_v4/` |
| ALIGNN模型权重 | ✅ 完成 | 从Figshare懒加载 |

### 1.2 架构设计原则

1. **最小化新增代码**：优先扩展现有模块，而非重写
2. **模块独立性**：各功能模块松耦合，可独立运行/测试
3. **渐进式增强**：先跑通MVP，再逐步完善高级功能
4. **资源高效**：针对CPU/小内存环境优化

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Web UI 层                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Streamlit App  │  │  Dash Universe   │  │  API Docs (FastAPI)     │  │
│  │  (主界面)        │  │  (可视化地图)    │  │  /docs                 │  │
│  └────────┬────────┘  └────────┬────────┘  └───────────┬─────────────┘  │
│           │                    │                     │                 │
│           │  HTTP              │  HTTP               │ REST            │
│  ┌────────▼────────���───────────▼─────────────────────▼─────────────┐  │
│  │                     FastAPI 服务层                              │  │
│  │  /predict/sync  /predict/async  /predict/batch  /tasks/{id}    │  │
│  └────────┬────────────────────┬─────────────────────┬─────────────┘  │
│           │                    │                     │                 │
│  ┌────────▼────────┐  ┌───────▼────────┐  ┌─────────▼──────────────┐  │
│  │  单结构预测      │  │  异步任务队列   │  │  批量预测              │  │
│  │  (同步/异步)    │  │  (Celery+Redis) │  │  (异步批处理)          │  │
│  └────────┬────────┘  └────────────────┘  └─────────┬──────────────┘  │
│           │                                        │                  │
│  ┌────────▼────────────────────────────────────────▼──────────────┐  │
│  │                    ALIGNN 推理引擎                              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │ ALIGNN-JARVIS│  │ ALIGNN-MP    │  │ ALIGNN-FF (力场)     │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │  │
│  │                                                                  │  │
│  │  GPU/CPU 计算资源池 (串行加载, 按需卸载)                         │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
alignn_service/
├── main.py              # FastAPI主入口
├── core/
│   ├── config.py        # 配置文件（所有模块依赖）
│   ├── model_manager.py # 模型管理器（predictor依赖）
│   ├── predictor.py     # 预测核心（tasks依赖）
│   ├── tasks.py         # Celery任务（API依赖）
│   ├── celery_utils.py  # Celery工具
│   └── doping_generator.py  # [新增] 掺杂生成器
├── api/
│   └── routes.py        # [新增] API路由模块
├── ui/
│   ├── app.py           # Streamlit主界面
│   ├── components.py    # [增强] 可复用组件库
│   ├── pages/           # [新增] Streamlit多页面
│   │   ├── batch.py     # 批量预测页面
│   │   ├── doping.py    # 自定义掺杂页面
│   │   ├── compare.py   # 数据库对比页面
│   │   └── history.py   # 历史记录页面
│   └── universe.py     # Dash可视化（独立端口）
└── utils/
    ├── file_parser.py   # [增强] 文件解析工具
    ├── csv_exporter.py  # [新增] CSV导出
    └── validators.py    # [新增] 输入验证
```

---

## 3. 数据流设计

### 3.1 单结构预测流程

```
用户上传POSCAR
    │
    ▼
[文件解析] ──→ Atoms对象
    │
    ▼
[3D可视化] ──→ 立即显示晶体结构
    │
    ▼
[图构建] ──→ DGL多图 (原子图 + 线图)
    │
    ▼
[模型推理] ──→ 预测值
    │
    ▼
[结果展示] ──→ 数值 + 单位 + 处理时间
```

### 3.2 批量预测流程

```
用户上传多个文件 (最多100个)
    │
    ▼
[文件校验] ──→ 有效文件列表 + 错误文件列表
    │
    ▼
[提交任务] ──→ Celery Task ID
    │
    ▼
[异步处理] ◀──────────────────┐
    │                          │
    ▼                          │
[逐个预测]                     │
    │                          │
    ▼                          │
[进度更新] ──→ Redis ──→ 前端轮询
    │                          │
    └────────── (循环) ─────────┘
    │
    ▼
[汇总结果] ──→ batch_{task_id}.json
    │
    ▼
[结果展示] ──→ 表格 + CSV导出
```

### 3.3 掺杂生成流程

```
用户配置:
- 宿主结构 (上传/选择预设)
- 掺杂元素 (多选)
- 浓度梯度 (1%~10%)
- 位点选择 (Li/Fe/P位)
    │
    ▼
[调用 generate_lfp_dopants_v4.py 逻辑]
    │
    ▼
[生成构型] ──→ POSCAR文件 + metadata.json
    │
    ▼
[可选: 批量预测] ──→ 提交到批量预测队列
```

---

## 4. 数据库设计

### 4.1 SQLite 数据库表

```sql
-- 预测记录表
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT UNIQUE NOT NULL,
    filename TEXT,
    formula TEXT,
    n_atoms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- JARVIS 模型
    jv_formation_energy REAL,
    jv_ehull REAL,
    jv_bandgap REAL,
    jv_bulk_modulus REAL,
    jv_shear_modulus REAL,
    jv_magmom REAL,
    
    -- MP 模型
    mp_formation_energy REAL,
    mp_bandgap REAL,
    
    -- 处理信息
    processing_time REAL,
    device TEXT,
    status TEXT DEFAULT 'pending'
);

-- 批量任务表
CREATE TABLE batch_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT UNIQUE NOT NULL,
    user_id TEXT,
    total_files INTEGER,
    successful INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    status TEXT DEFAULT 'pending'
);

-- 掺杂构型表
CREATE TABLE doping_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id TEXT UNIQUE NOT NULL,
    doping_site TEXT,
    dopant_element TEXT,
    n_dopant INTEGER,
    concentration_pct REAL,
    formula TEXT,
    n_atoms INTEGER,
    poscar_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- 预测结果（可选）
    jv_formation_energy REAL,
    jv_ehull REAL,
    jv_bandgap REAL,
    score REAL
);
```

### 4.2 文件存储结构

```
alignn_service/
├── data/
│   ├── uploads/           # 用户上传文件
│   │   └── {user_id}/
│   │       └── {task_id}/
│   │           └── {filename}
│   └── results/           # 预测结果
│       └── {user_id}/
│           ├── {task_id}.json
│           └── batch_{task_id}.json
├── models/                # ALIGNN模型权重
│   ├── jv_formation_energy_peratom_alignn/
│   ├── jv_optb88vdw_bandgap_alignn/
│   └── ...
└── generated/            # 生成的掺杂构型
    └── doping/
        └── {config_id}.poscar
```

---

## 5. API 接口设计

### 5.1 REST API 端点

| 方法 | 路径 | 说明 | 参数 |
|-----|------|------|------|
| GET | `/health` | 健康检查 | - |
| GET | `/api/v1/models` | 获取可用模型 | - |
| GET | `/api/v1/models/{name}` | 获取模型详情 | - |
| POST | `/api/v1/predict/sync` | 同步预测 | file, models |
| POST | `/api/v1/predict/async` | 异步预测 | file, models |
| POST | `/api/v1/predict/batch` | 批量预测 | files[], models |
| GET | `/api/v1/tasks/{id}` | 获取任务状态 | - |
| GET | `/api/v1/tasks/{id}/result` | 获取任务结果 | - |
| GET | `/api/v1/predictions` | 查询历史记录 | user_id, limit |
| GET | `/api/v1/predictions/{id}` | 获取单条记录 | - |
| POST | `/api/v1/doping/generate` | 生成掺杂构型 | config |
| GET | `/api/v1/stats` | 服务统计 | - |

### 5.2 请求/响应示例

#### 同步预测
```bash
# 请求
curl -X POST "http://localhost:8000/api/v1/predict/sync" \
  -F "file=@LiFePO4.poscar" \
  -F "models=jv_formation_energy_peratom_alignn,jv_optb88vdw_bandgap_alignn"

# 响应
{
  "status": "success",
  "task_id": "abc123",
  "predictions": {
    "jv_formation_energy_peratom_alignn": {
      "value": -2.532,
      "unit": "eV/atom",
      "processing_time": 12.34
    },
    "jv_optb88vdw_bandgap_alignn": {
      "value": 3.694,
      "unit": "eV",
      "processing_time": 11.87
    }
  },
  "structure_info": {
    "formula": "LiFePO4",
    "n_atoms": 28,
    "elements": {"Li": 4, "Fe": 4, "P": 4, "O": 16}
  }
}
```

#### 批量预测
```bash
# 请求
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -F "files=@config1.poscar" \
  -F "files=@config2.poscar" \
  -F "models=jv_formation_energy_peratom_alignn"

# 响应
{
  "task_id": "batch_xyz789",
  "status": "pending",
  "total_files": 2,
  "message": "批量任务已提交"
}
```

### 5.3 任务状态查询
```bash
# 请求
curl "http://localhost:8000/api/v1/tasks/batch_xyz789"

# 响应
{
  "task_id": "batch_xyz789",
  "status": "PROCESSING",
  "info": {
    "progress": 0.5,
    "message": "处理结构 1/2..."
  },
  "ready": false,
  "successful": null
}
```

---

## 6. 前端界面设计

### 6.1 Streamlit 多页面结构

```
alignn_service/ui/
├── app.py                 # 主入口（单结构预测）
├── components.py          # 共享组件
└── pages/
    ├── batch.py           # 批量预测
    ├── doping.py          # 自定���掺杂
    ├── compare.py         # 数据库对比
    ├── history.py         # 历史记录
    └── export.py          # 数据导出
```

### 6.2 页面功能划分

| 页面 | 功能 | 复杂度 |
|-----|------|-------|
| 主界面 (app.py) | 单结构上传 + 3D预览 + 预测 + 结果展示 | P0 |
| batch.py | 多文件上传 + 进度追踪 + 批量结果 | P0 |
| doping.py | 宿主结构选择 + 掺杂配置 + 生成预览 | P1 |
| compare.py | 多数据库预测对比 + 可视化 | P1 |
| history.py | 历史记录列表 + 筛选 + 导出 | P2 |
| export.py | 结果导出（CSV/Excel/JSON） | P2 |

### 6.3 组���库设计

```python
# components.py
class CrystalComponents:
    @staticmethod
    def file_uploader()          # 文件上传组件
    @staticmethod
    def model_selector()         # 模型选择器
    @staticmethod
    def structure_preview()      # 结构预览
    @staticmethod
    def prediction_cards()      # 预测结果卡片
    @staticmethod
    def progress_tracker()       # 进度追踪器
    @staticmethod
    def result_table()          # 结果表格
    @staticmethod
    def export_buttons()        # 导出按钮
```

---

## 7. 性能优化策略

### 7.1 CPU 环境优化

| 优化项 | 策略 | 预期提升 |
|-------|------|---------|
| 模型加载 | 串行加载 + 即时卸载 | 内存峰值 < 4GB |
| 图构建 | 缓存最近 N 个结构 | 重复请求 < 100ms |
| 批量处理 | 任务队列 + 并发控制 | 吞吐量 + 50% |
| 预测超时 | 5分钟软限制 | 防止资源耗尽 |

### 7.2 异步处理设计

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  FastAPI    │────▶│   Redis     │────▶│  Celery     │
│  (接收请求)  │     │  (消息队列) │     │  (执行任务) │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
              ┌─────────────┐
              │  结果存储   │
              │  (JSON文件) │
              └─────────────┘
```

---

## 8. 部署方案

### 8.1 Docker Compose 配置

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  worker:
    build: .
    command: celery -A alignn_service.core.tasks worker
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  ui:
    build: .
    command: streamlit run alignn_service/ui/app.py
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api

  universe:
    build: .
    command: python alignn/models/lfp_universe_app.py
    ports:
      - "8050:8050"
```

### 8.2 环境变量配置

```bash
# .env
REDIS_URL=redis://localhost:6379/0
MODEL_BASE_DIR=/Users/mxye/Myprojects/alignn/models
UPLOAD_DIR=/Users/mxye/Myprojects/alignn/data/uploads
RESULTS_DIR=/Users/mxye/Myprojects/alignn/data/results
API_PORT=8000
UI_PORT=8501
DEBUG=False
```

---

## 9. 开发计划

### 9.1 分阶段实施

| 阶段 | 功能 | 工作量 | 优先级 | 状态 |
|-----|------|-------|-------|------|
| **Phase 1: MVP完善** | 单结构预测 + 3D可视化完善 | 2天 | P0 | ✅ 已完成 |
| **Phase 2: 批量能力** | 批量预测 + 进度追踪 + 结果汇总 | 2天 | P0 | ✅ 已完成 |
| **Phase 3: 掺杂生成** | 自定义掺杂UI + 批量提交 | 2天 | P1 | ✅ 已完成 |
| **Phase 4: 数据对比** | 多数据库对比 + 可视化 | 2天 | P1 | ✅ 已完成 |
| **Phase 5: 历史管理** | 历史记录 + 导出功能 | 1天 | P2 | ✅ 已完成 |
| **Phase 6: 测试部署** | 单元测试 + Docker部署 | 1天 | P1 | ✅ 已完成 |

### 9.2 技术债务

- [x] 单元测试覆盖率 > 60%
- [x] 错误处理完善
- [x] 日志规范化
- [x] API文档完善 (FastAPI 自动生成)
- [x] 性能基准测试

---

**文档版本**：v1.0
**最后更新**：2026-04-14
**作者**：系统设计Agent