"""模型管理器模块

负责模型加载、缓存、卸载
针对 CPU + 小内存环境优化：每次只加载一个模型
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import torch
import requests

from alignn.models.alignn import ALIGNN, ALIGNNConfig

from .config import AVAILABLE_MODELS, settings

logger = logging.getLogger(__name__)


class ModelInfo:
    """模型信息"""

    def __init__(self, name: str, config: Dict, checkpoint_path: str):
        self.name = name
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.model: Optional[ALIGNN] = None
        self._lock = threading.Lock()


class ModelManager:
    """
    模型管理器

    设计原则：
    1. 串行加载：同一时间只加载一个模型
    2. 及时释放：预测完成后立即释放内存
    3. 按需下载：从 Figshare 懒加载模型
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()
        self._device = self._get_device()

    def _get_device(self) -> str:
        """检测可用设备"""
        if torch.cuda.is_available():
            logger.info("检测到 GPU，将使用 CUDA")
            return "cuda"
        else:
            logger.info("使用 CPU 进行推理")
            return "cpu"

    @property
    def device(self) -> str:
        """获取当前设备"""
        return self._device

    def _get_model_base_path(self, model_name: str) -> Path:
        """获取模型本地存储路径"""
        return settings.MODEL_BASE_DIR / model_name

    def _get_checkpoint_path(self, model_name: str) -> Path:
        """获取 checkpoint 文件路径"""
        base = self._get_model_base_path(model_name)
        return base / "checkpoint_300.pt"

    def _get_config_path(self, model_name: str) -> Path:
        """获取配置文件路径"""
        base = self._get_model_base_path(model_name)
        return base / "config.json"

    def _download_model(self, model_name: str) -> bool:
        """
        从 Figshare 下载模型

        Returns:
            bool: 下载是否成功
        """
        if model_name not in AVAILABLE_MODELS:
            logger.error(f"未知模型: {model_name}")
            return False

        model_dir = self._get_model_base_path(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Figshare 模型 ID 映射
        figshare_ids = {
            "jv_formation_energy_peratom_alignn": "14758619",
            "jv_optb88vdw_bandgap_alignn": "14758620",
            "jv_mbj_bandgap_alignn": "14758621",
            "mp_e_form_alignn": "14758622",
            "mp_gappbe_alignn": "14758623",
            "jv_bulk_modulus_kv_alignn": "14758624",
            "jv_shear_modulus_gv_alignn": "14758625",
        }

        figshare_id = figshare_ids.get(model_name)
        if not figshare_id:
            logger.error(f"未找到模型 {model_name} 的 Figshare ID")
            return False

        # 下载配置
        config_url = f"https://figshare.com/ndownloader/files/{figshare_id}/config.json"
        checkpoint_url = f"https://figshare.com/ndownloader/files/{figshare_id}/checkpoint_300.pt"

        try:
            # 下载 config.json
            config_path = self._get_config_path(model_name)
            if not config_path.exists():
                logger.info(f"下载 {model_name} 配置文件...")
                response = requests.get(config_url, timeout=60)
                response.raise_for_status()
                with open(config_path, "w") as f:
                    f.write(response.text)

            # 下载 checkpoint
            checkpoint_path = self._get_checkpoint_path(model_name)
            if not checkpoint_path.exists():
                logger.info(f"下载 {model_name} 模型文件（约 50MB）...")
                response = requests.get(checkpoint_url, timeout=300)
                response.raise_for_status()
                with open(checkpoint_path, "wb") as f:
                    f.write(response.content)

            logger.info(f"模型 {model_name} 下载完成")
            return True

        except requests.RequestException as e:
            logger.error(f"下载模型失败: {e}")
            return False

    def _load_model_config(self, model_name: str) -> Optional[Dict]:
        """加载模型配置"""
        config_path = self._get_config_path(model_name)
        if not config_path.exists():
            logger.warning(f"模型配置文件不存在: {config_path}")
            if not self._download_model(model_name):
                return None

        with open(config_path) as f:
            return json.load(f)

    def load_model(self, model_name: str) -> Optional[ALIGNN]:
        """
        加载模型

        Args:
            model_name: 模型名称

        Returns:
            加载的模型，失败返回 None
        """
        if model_name not in AVAILABLE_MODELS:
            logger.error(f"未知模型: {model_name}")
            return None

        with self._lock:
            # 检查是否已加载
            if model_name in self._models and self._models[model_name].model is not None:
                logger.info(f"模型 {model_name} 已缓存")
                return self._models[model_name].model

            # 清理其他模型（释放内存）
            self._unload_all()

            # 加载配置
            config_data = self._load_model_config(model_name)
            if not config_data:
                return None

            # 创建模型
            try:
                model_cfg = ALIGNNConfig(**config_data["model"])
                model = ALIGNN(config=model_cfg)

                # 加载权重
                checkpoint_path = self._get_checkpoint_path(model_name)
                state = torch.load(
                    checkpoint_path,
                    map_location=self._device,
                    weights_only=False
                )
                model.load_state_dict(state["model"])

                # 移到设备并设置为评估模式
                model = model.to(self._device)
                model.eval()

                # 缓存模型
                model_info = ModelInfo(
                    name=model_name,
                    config=config_data,
                    checkpoint_path=str(checkpoint_path)
                )
                model_info.model = model
                self._models[model_name] = model_info

                # 记录内存使用
                memory_allocated = torch.cuda.memory_allocated() if self._device == "cuda" else 0
                logger.info(
                    f"模型 {model_name} 加载完成, "
                    f"内存使用: {memory_allocated / 1024**2:.1f} MB"
                )

                return model

            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                return None

    def unload_model(self, model_name: str):
        """卸载模型，释放内存"""
        with self._lock:
            if model_name in self._models:
                if self._models[model_name].model is not None:
                    del self._models[model_name].model
                    self._models[model_name].model = None

                    if self._device == "cuda":
                        torch.cuda.empty_cache()
                    else:
                        import gc
                        gc.collect()

                    logger.info(f"模型 {model_name} 已卸载")

    def _unload_all(self):
        """卸载所有模型"""
        model_names = list(self._models.keys())
        for name in model_names:
            self.unload_model(name)

    def get_loaded_models(self) -> list:
        """获取已加载的模型列表"""
        return [
            name for name, info in self._models.items()
            if info.model is not None
        ]

    def preload_models(self, model_names: list = None):
        """
        预加载模型

        Args:
            model_names: 要预加载的模型列表，None 表示全部
        """
        if model_names is None:
            model_names = list(AVAILABLE_MODELS.keys())

        for name in model_names:
            if name in AVAILABLE_MODELS:
                self.load_model(name)

    def list_local_models(self) -> list:
        """列出本地已有的模型"""
        if not settings.MODEL_BASE_DIR.exists():
            return []

        local_models = []
        for model_dir in settings.MODEL_BASE_DIR.iterdir():
            if model_dir.is_dir():
                checkpoint = model_dir / "checkpoint_300.pt"
                config = model_dir / "config.json"
                if checkpoint.exists() and config.exists():
                    local_models.append(model_dir.name)

        return local_models


# 全局模型管理器实例
model_manager = ModelManager()
