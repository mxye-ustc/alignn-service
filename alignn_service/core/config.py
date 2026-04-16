"""ALIGNN 预测服务配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""

    # 应用基础配置
    APP_NAME: str = "ALIGNN Prediction Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, validation_alias="DEBUG")

    # 服务器配置
    HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    UI_PORT: int = 8501

    # 数据库配置
    DATABASE_URL: str = Field(
        default="sqlite:///./alignn_service.db",
        validation_alias="DATABASE_URL"
    )

    # Redis 配置
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="REDIS_URL"
    )

    # 模型配置
    MODEL_BASE_DIR: Path = Field(
        default=Path("/models"),
        validation_alias="MODEL_BASE_DIR"
    )

    # 预测参数
    DEFAULT_CUTOFF: float = 8.0
    DEFAULT_MAX_NEIGHBORS: int = 12
    GRAPH_DISTANCE: str = "nearest_neighbor"

    # 并发配置
    MAX_CONCURRENT_PREDICTIONS: int = 1  # 4GB 内存限制
    PREDICTION_TIMEOUT: int = 300  # 5分钟超时

    # 用户认证配置
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        validation_alias="SECRET_KEY"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24小时

    # 上传文件配置
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: Path = Path("./data/uploads")
    RESULTS_DIR: Path = Path("./data/results")

    # CORS 配置
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


# 可用的预测模型配置
AVAILABLE_MODELS = {
    # JARVIS-DFT 数据集模型（用于带隙和形成能）
    "jv_formation_energy_peratom_alignn": {
        "name": "形成能 (JARVIS)",
        "description": "预测化合物的形成能",
        "unit": "eV/atom",
        "source": "JARVIS-DFT",
        "type": "energy",
    },
    "jv_optb88vdw_bandgap_alignn": {
        "name": "带隙 (OptB88vdW)",
        "description": "使用 OptB88vdW 泛函预测带隙",
        "unit": "eV",
        "source": "JARVIS-DFT",
        "type": "electronic",
    },
    "jv_mbj_bandgap_alignn": {
        "name": "带隙 (MBJ)",
        "description": "使用 meta-GGA BLIP 方法预测带隙（更精确）",
        "unit": "eV",
        "source": "JARVIS-DFT",
        "type": "electronic",
    },
    # Materials Project 模型
    "mp_e_form_alignn": {
        "name": "形成能 (MP)",
        "description": "Materials Project 数据集形成能",
        "unit": "eV/atom",
        "source": "Materials Project",
        "type": "energy",
    },
    "mp_gappbe_alignn": {
        "name": "带隙 (PBE)",
        "description": "Materials Project PBE 带隙",
        "unit": "eV",
        "source": "Materials Project",
        "type": "electronic",
    },
    # 力学性质模型
    "jv_bulk_modulus_kv_alignn": {
        "name": "体模量",
        "description": "预测体积模量",
        "unit": "GPa",
        "source": "JARVIS-DFT",
        "type": "mechanical",
    },
    "jv_shear_modulus_gv_alignn": {
        "name": "剪切模量",
        "description": "预测剪切模量",
        "unit": "GPa",
        "source": "JARVIS-DFT",
        "type": "mechanical",
    },
}


# 获取全局配置实例
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
