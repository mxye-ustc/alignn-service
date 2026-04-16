"""模型管理 API 路由"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from alignn_service.core import AVAILABLE_MODELS, model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["模型"])


@router.get("")
async def list_models() -> Dict[str, Any]:
    """获取可用模型列表"""
    return {
        "models": AVAILABLE_MODELS,
        "total": len(AVAILABLE_MODELS),
        "local_models": model_manager.list_local_models()
    }


@router.get("/{model_name}")
async def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型详情"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")

    return {
        "name": model_name,
        "info": AVAILABLE_MODELS[model_name],
        "is_loaded": model_name in model_manager.get_loaded_models()
    }


@router.post("/{model_name}/preload")
async def preload_model(model_name: str) -> Dict[str, Any]:
    """预加载模型"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")

    model = model_manager.load_model(model_name)

    if model is None:
        raise HTTPException(status_code=500, detail="模型加载失败")

    return {
        "name": model_name,
        "status": "loaded",
        "device": model_manager.device
    }


@router.post("/{model_name}/unload")
async def unload_model(model_name: str) -> Dict[str, Any]:
    """卸载模型"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")

    model_manager.unload_model(model_name)

    return {
        "name": model_name,
        "status": "unloaded"
    }
