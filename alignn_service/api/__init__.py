"""API 路由模块"""

from .routes import (
    predictions_router,
    tasks_router,
    models_router,
    doping_router,
)

__all__ = [
    "predictions_router",
    "tasks_router",
    "models_router",
    "doping_router",
]
