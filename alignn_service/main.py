"""FastAPI 主应用"""

import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from alignn_service.core import (
    AVAILABLE_MODELS,
    get_task_status,
    get_task_result,
    model_manager,
    predict_task,
    batch_predict_task,
)
from alignn_service.core.config import settings
from alignn_service.api.routes import (
    predictions_router,
    tasks_router,
    models_router,
    doping_router,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info(f"启动 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"设备: {model_manager.device}")

    # 确保目录存在
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # 关闭时
    logger.info("服务关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## ALIGNN 晶体性质预测服务

支持使用预训练的 ALIGNN 模型预测晶体的多种性质：
- **形成能**：预测化合物的形成能
- **带隙**：预测电子带隙
- **力学性质**：体模量、剪切模量等

### 使用方式

1. **单结构预测**：上传晶体结构文件，立即获得预测结果
2. **批量预测**：提交多个结构，系统后台处理后返回结果
3. **异步任务**：对于耗时较长的任务，系统返回任务 ID，可随时查询状态

### 支持的文件格式

- POSCAR / VASP
- CIF
- XYZ
- PDB
    """,
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(models_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(doping_router, prefix="/api/v1")


# ==================== 健康检查 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": model_manager.device,
        "loaded_models": model_manager.get_loaded_models(),
    }


# ==================== 模型管理 ====================

@app.get("/api/v1/models", tags=["模型"])
async def list_models():
    """获取可用模型列表"""
    return {
        "models": AVAILABLE_MODELS,
        "total": len(AVAILABLE_MODELS),
        "local_models": model_manager.list_local_models()
    }


@app.get("/api/v1/models/{model_name}", tags=["模型"])
async def get_model_info(model_name: str):
    """获取模型详情"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")

    return {
        "name": model_name,
        "info": AVAILABLE_MODELS[model_name],
        "is_loaded": model_name in model_manager.get_loaded_models()
    }


# ==================== 单结构预测 ====================

@app.post("/api/v1/predict/sync", tags=["预测"])
async def predict_sync(
    file: UploadFile = File(..., description="晶体结构文件"),
    models: str = Query(..., description="模型名称，逗号分隔"),
    cutoff: Optional[float] = Query(None, description="截断半径（埃）"),
    max_neighbors: Optional[int] = Query(None, description="最大近邻数"),
):
    """
    同步预测（适用于小结构、快速返回）

    注意：由于 CPU 推理较慢，建议使用异步接口
    """
    # 读取文件
    content = await file.read()
    model_list = [m.strip() for m in models.split(",")]

    # 验证模型
    invalid_models = [m for m in model_list if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型: {', '.join(invalid_models)}"
        )

    # Base64 编码
    encoded_content = base64.b64encode(content).decode("utf-8")

    # 提交异步任务
    task = predict_task.delay(
        structure_content=encoded_content,
        model_names=model_list,
        file_format=file.filename.split(".")[-1] if "." in file.filename else "poscar",
        cutoff=cutoff,
        max_neighbors=max_neighbors
    )

    # 等待结果（同步模式）
    result = task.get(timeout=settings.PREDICTION_TIMEOUT)

    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return result


@app.post("/api/v1/predict/async", tags=["预测"])
async def predict_async(
    file: UploadFile = File(..., description="晶体结构文件"),
    models: str = Query(..., description="模型名称，逗号分隔"),
    cutoff: Optional[float] = Query(None, description="截断半径（埃）"),
    max_neighbors: Optional[int] = Query(None, description="最大近邻数"),
    user_id: Optional[str] = Query(None, description="用户 ID"),
):
    """
    异步预测（推荐）

    立即返回任务 ID，通过 /api/v1/tasks/{task_id} 查询结果
    """
    # 读取文件
    content = await file.read()
    model_list = [m.strip() for m in models.split(",")]

    # 验证模型
    invalid_models = [m for m in model_list if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型: {', '.join(invalid_models)}"
        )

    # Base64 编码
    encoded_content = base64.b64encode(content).decode("utf-8")

    # 提交异步任务
    task = predict_task.delay(
        structure_content=encoded_content,
        model_names=model_list,
        file_format=file.filename.split(".")[-1] if "." in file.filename else "poscar",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        user_id=user_id
    )

    return {
        "task_id": task.id,
        "status": "pending",
        "message": "任务已提交，请通过 /api/v1/tasks/{task_id} 查询状态"
    }


# ==================== 批量预测 ====================

@app.post("/api/v1/predict/batch", tags=["批量预测"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="结构文件列表（最多 100 个）"),
    models: str = Query(..., description="模型名称，逗号分隔"),
    cutoff: Optional[float] = Query(None, description="截断半径"),
    max_neighbors: Optional[int] = Query(None, description="最大近邻数"),
    user_id: Optional[str] = Query(None, description="用户 ID"),
):
    """
    批量预测

    上传多个结构文件，系统后台处理
    """
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="最多支持 100 个文件")

    model_list = [m.strip() for m in models.split(",")]

    # 验证模型
    invalid_models = [m for m in model_list if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型: {', '.join(invalid_models)}"
        )

    # 构建结构列表
    structures = []
    for file in files:
        content = await file.read()
        encoded = base64.b64encode(content).decode("utf-8")
        structures.append({
            "content": encoded,
            "format": file.filename.split(".")[-1] if "." in file.filename else "poscar",
            "name": file.filename
        })

    # 提交批量任务
    task = batch_predict_task.delay(
        structures=structures,
        model_names=model_list,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        user_id=user_id
    )

    return {
        "task_id": task.id,
        "status": "pending",
        "total_files": len(files),
        "message": "批量任务已提交，请通过 /api/v1/tasks/{task_id} 查询状态"
    }


# ==================== 任务管理 ====================

@app.get("/api/v1/tasks/{task_id}", tags=["任务"])
async def get_task(task_id: str):
    """获取任务状态"""
    status = get_task_status(task_id)

    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="任务不存在")

    return status


@app.get("/api/v1/tasks/{task_id}/result", tags=["任务"])
async def get_task_result_endpoint(task_id: str):
    """获取任务结果"""
    result = get_task_result(task_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="任务不存在或尚未完成"
        )

    return result


# ==================== 统计信息 ====================

@app.get("/api/v1/stats", tags=["统计"])
async def get_stats():
    """获取服务统计信息"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": model_manager.device,
        "available_models": len(AVAILABLE_MODELS),
        "loaded_models": len(model_manager.get_loaded_models()),
        "max_concurrent": settings.MAX_CONCURRENT_PREDICTIONS,
    }


# ==================== 根路径 ====================

@app.get("/", tags=["信息"])
async def root():
    """服务根路径"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "alignn_service.main:app",
        host=settings.HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
