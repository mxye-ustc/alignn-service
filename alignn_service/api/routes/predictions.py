"""预测相关 API 路由"""

import base64
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from alignn_service.core import (
    AVAILABLE_MODELS,
    predict_task,
    batch_predict_task,
)
from alignn_service.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["预测"])


@router.post("/sync")
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
    content = await file.read()
    model_list = [m.strip() for m in models.split(",")]

    invalid_models = [m for m in model_list if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型: {', '.join(invalid_models)}"
        )

    encoded_content = base64.b64encode(content).decode("utf-8")

    task = predict_task.delay(
        structure_content=encoded_content,
        model_names=model_list,
        file_format=file.filename.split(".")[-1] if "." in file.filename else "poscar",
        cutoff=cutoff,
        max_neighbors=max_neighbors
    )

    result = task.get(timeout=settings.PREDICTION_TIMEOUT)

    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return result


@router.post("/async")
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
    content = await file.read()
    model_list = [m.strip() for m in models.split(",")]

    invalid_models = [m for m in model_list if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型: {', '.join(invalid_models)}"
        )

    encoded_content = base64.b64encode(content).decode("utf-8")

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


@router.post("/batch")
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

    invalid_models = [m for m in model_list if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型: {', '.join(invalid_models)}"
        )

    structures = []
    for file in files:
        content = await file.read()
        encoded = base64.b64encode(content).decode("utf-8")
        structures.append({
            "content": encoded,
            "format": file.filename.split(".")[-1] if "." in file.filename else "poscar",
            "name": file.filename
        })

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
