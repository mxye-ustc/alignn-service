"""Celery 工具函数"""

from celery.result import AsyncResult

from .config import settings

# 获取 Celery 应用实例
_celery_app = None


def get_celery_app():
    """获取 Celery 应用实例"""
    global _celery_app
    if _celery_app is None:
        _celery_app = Celery("alignn_tasks")
        _celery_app.conf.broker_url = settings.REDIS_URL
        _celery_app.conf.result_backend = settings.REDIS_URL
    return _celery_app


def get_async_result(task_id: str) -> AsyncResult:
    """
    获取异步任务结果

    Args:
        task_id: 任务 ID

    Returns:
        AsyncResult 对象
    """
    app = get_celery_app()
    return AsyncResult(task_id, app=app)


# 延迟导入 Celery
from celery import Celery
