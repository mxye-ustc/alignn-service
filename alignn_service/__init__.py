"""ALIGNN 预测服务"""

__version__ = "1.0.0"
__author__ = "ALIGNN Team"

from .core import (
    settings,
    AVAILABLE_MODELS,
    model_manager,
    Predictor,
    PredictionError,
)

__all__ = [
    "settings",
    "AVAILABLE_MODELS",
    "model_manager",
    "Predictor",
    "PredictionError",
]
