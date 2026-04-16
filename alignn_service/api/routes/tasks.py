"""任务管理 API 路由"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from alignn_service.core import get_task_status, get_task_result

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["任务"])


@router.get("/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """获取任务状态"""
    status = get_task_status(task_id)

    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="任务不存在")

    return status


@router.get("/{task_id}/result")
async def get_task_result_endpoint(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务结果"""
    result = get_task_result(task_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="任务不存在或尚未完成"
        )

    return result
