"""API 路由模块

提供 REST API 端点的模块化组织
"""

from .predictions import router as predictions_router
from .tasks import router as tasks_router
from .models import router as models_router
from .doping import router as doping_router

__all__ = [
    "predictions_router",
    "tasks_router",
    "models_router",
    "doping_router",
]
