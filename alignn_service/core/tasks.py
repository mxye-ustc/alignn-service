"""Celery 异步任务模块

处理预测任务的异步执行
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from .config import settings
from .predictor import Predictor, PredictionError, StructureParser

logger = logging.getLogger(__name__)

# 创建 Celery 应用
celery_app = Celery(
    "alignn_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["alignn_service.core.tasks"]
)

# Celery 配置
celery_app.conf.update(
    # 任务路由
    task_routes={
        "alignn_service.core.tasks.predict_task": {"queue": "prediction"},
        "alignn_service.core.tasks.batch_predict_task": {"queue": "batch"},
    },
    # 任务序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # 结果过期时间（24小时）
    result_expires=86400,
    # 任务超时（5分钟）
    task_soft_time_limit=settings.PREDICTION_TIMEOUT - 30,
    task_time_limit=settings.PREDICTION_TIMEOUT,
    # 并发数（CPU 环境下限制）
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
    # 任务重试
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Worker 就绪时的回调"""
    logger.info("预测服务 Worker 已就绪")


@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Worker 关闭时的回调"""
    logger.info("预测服务 Worker 正在关闭")


@celery_app.task(bind=True, name="alignn_service.core.tasks.predict_task")
def predict_task(
    self,
    structure_content: str,
    model_names: List[str],
    file_format: str = "poscar",
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    单结构预测任务

    Args:
        structure_content: 结构文件内容（Base64 编码）
        model_names: 模型名称列表
        file_format: 文件格式
        cutoff: 截断半径
        max_neighbors: 最大近邻数
        user_id: 用户 ID（可选）

    Returns:
        预测结果
    """
    import base64

    task_id = self.request.id
    logger.info(f"任务 {task_id} 开始执行")

    # 更新状态
    self.update_state(
        state="PROCESSING",
        meta={
            "status": "processing",
            "message": "正在解析结构...",
            "progress": 0.1
        }
    )

    try:
        # 解码结构内容
        decoded_content = base64.b64decode(structure_content).decode("utf-8")

        # 解析结构
        atoms = StructureParser.parse(decoded_content, file_format)

        self.update_state(
            state="PROCESSING",
            meta={
                "status": "processing",
                "message": f"结构已解析，共 {atoms.num_atoms} 个原子",
                "progress": 0.2
            }
        )

        # 创建预测器
        predictor = Predictor(cutoff=cutoff, max_neighbors=max_neighbors)

        # 逐个模型预测
        results = {}
        n_models = len(model_names)

        for i, model_name in enumerate(model_names):
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": "processing",
                    "message": f"正在使用 {model_name} 预测...",
                    "progress": 0.2 + 0.7 * (i / n_models)
                }
            )

            try:
                result = predictor.predict_single(atoms, model_name)
                results[model_name] = {
                    "value": result["value"],
                    "unit": result["unit"],
                    "processing_time": result["processing_time_seconds"]
                }
            except PredictionError as e:
                results[model_name] = {
                    "error": str(e)
                }

        # 计算总时间
        total_time = sum(
            r.get("processing_time", 0)
            for r in results.values()
            if "error" not in r
        )

        # 准备最终结果
        final_result = {
            "status": "success",
            "task_id": task_id,
            "predictions": results,
            "structure_info": predictor._get_structure_info(atoms),
            "total_models": n_models,
            "successful": sum(1 for r in results.values() if "error" not in r),
            "total_time_seconds": round(total_time, 2),
            "completed_at": datetime.now().isoformat()
        }

        # 保存结果到文件
        result_dir = settings.RESULTS_DIR / user_id if user_id else settings.RESULTS_DIR
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{task_id}.json"
        with open(result_file, "w") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        logger.info(f"任务 {task_id} 完成")

        return final_result

    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {e}")

        # 保存错误结果
        error_result = {
            "status": "failed",
            "task_id": task_id,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }

        return error_result


@celery_app.task(bind=True, name="alignn_service.core.tasks.batch_predict_task")
def batch_predict_task(
    self,
    structures: List[Dict[str, str]],
    model_names: List[str],
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    批量预测任务

    Args:
        structures: 结构列表 [{"content": "...", "format": "poscar", "name": "..."}]
        model_names: 模型名称列表
        cutoff: 截断半径
        max_neighbors: 最大近邻数
        user_id: 用户 ID

    Returns:
        批量预测结果
    """
    import base64

    task_id = self.request.id
    total = len(structures)
    logger.info(f"批量任务 {task_id} 开始，共 {total} 个结构")

    results = []
    errors = []
    total_time = 0

    predictor = Predictor(cutoff=cutoff, max_neighbors=max_neighbors)

    for i, struct in enumerate(structures):
        try:
            # 解码结构
            decoded_content = base64.b64decode(struct["content"]).decode("utf-8")
            atoms = StructureParser.parse(
                decoded_content,
                struct.get("format", "poscar")
            )

            self.update_state(
                state="PROCESSING",
                meta={
                    "status": "processing",
                    "message": f"处理结构 {i+1}/{total}...",
                    "progress": i / total
                }
            )

            # 预测
            predictions = {}
            for model_name in model_names:
                try:
                    result = predictor.predict_single(atoms, model_name)
                    predictions[model_name] = {
                        "value": result["value"],
                        "unit": result["unit"]
                    }
                except PredictionError as e:
                    predictions[model_name] = {"error": str(e)}

            results.append({
                "name": struct.get("name", f"structure_{i+1}"),
                "formula": atoms.composition.reduced_formula,
                "n_atoms": atoms.num_atoms,
                "predictions": predictions
            })

            total_time += sum(
                p.get("processing_time", 0)
                for p in predictions.values()
                if "error" not in p
            )

        except Exception as e:
            errors.append({
                "index": i,
                "name": struct.get("name", f"structure_{i+1}"),
                "error": str(e)
            })

    # 保存结果
    result_dir = settings.RESULTS_DIR / user_id if user_id else settings.RESULTS_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    batch_result = {
        "status": "completed",
        "task_id": task_id,
        "total_structures": total,
        "successful": len(results),
        "failed": len(errors),
        "total_time_seconds": round(total_time, 2),
        "results": results,
        "errors": errors,
        "completed_at": datetime.now().isoformat()
    }

    result_file = result_dir / f"batch_{task_id}.json"
    with open(result_file, "w") as f:
        json.dump(batch_result, f, indent=2, ensure_ascii=False)

    logger.info(f"批量任务 {task_id} 完成，成功 {len(results)}，失败 {len(errors)}")

    return batch_result


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    获取任务状态

    Args:
        task_id: 任务 ID

    Returns:
        任务状态信息
    """
    from alignn_service.core.celery_utils import get_async_result

    result = get_async_result(task_id)

    if result is None:
        return {
            "task_id": task_id,
            "status": "not_found",
            "message": "任务不存在或已过期"
        }

    return {
        "task_id": task_id,
        "status": result.state,
        "info": result.info if hasattr(result, "info") else None,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None
    }


def get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取任务结果

    Args:
        task_id: 任务 ID

    Returns:
        任务结果，失败返回 None
    """
    from alignn_service.core.celery_utils import get_async_result

    result = get_async_result(task_id)

    if result is None:
        return None

    if not result.ready():
        return None

    if result.successful():
        return result.result
    else:
        return {
            "status": "failed",
            "error": str(result.result)
        }
