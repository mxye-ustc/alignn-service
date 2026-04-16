"""预测核心模块

处理晶体结构预测的核心逻辑
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph

from .config import AVAILABLE_MODELS, settings
from .model_manager import model_manager

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """预测异常"""
    pass


class StructureParser:
    """结构文件解析器"""

    SUPPORTED_FORMATS = ["poscar", "vasp", "cif", "xyz", "pdb"]

    @classmethod
    def parse(
        cls,
        content: Union[str, bytes],
        file_format: str = "poscar",
        filename: Optional[str] = None
    ) -> Atoms:
        """
        解析晶体结构文件

        Args:
            content: 文件内容或字节
            file_format: 文件格式
            filename: 文件名（用于自动检测格式）

        Returns:
            Atoms 对象
        """
        # 自动检测格式
        if file_format == "auto" and filename:
            file_format = cls._detect_format(filename)

        file_format = file_format.lower()

        if file_format not in cls.SUPPORTED_FORMATS:
            raise PredictionError(
                f"不支持的文件格式: {file_format}。"
                f"支持的格式: {', '.join(cls.SUPPORTED_FORMATS)}"
            )

        try:
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            if file_format in ("poscar", "vasp"):
                return Atoms.from_poscar(content)
            elif file_format == "cif":
                return cls._parse_cif(content)
            elif file_format == "xyz":
                return cls._parse_xyz(content)
            elif file_format == "pdb":
                return cls._parse_pdb(content)
            else:
                raise PredictionError(f"无法解析格式: {file_format}")

        except Exception as e:
            raise PredictionError(f"结构文件解析失败: {e}")

    @staticmethod
    def _detect_format(filename: str) -> str:
        """根据文件名检测格式"""
        ext = Path(filename).suffix.lower().lstrip(".")
        format_map = {
            "poscar": "poscar",
            "vasp": "vasp",
            "cif": "cif",
            "xyz": "xyz",
            "pdb": "pdb",
        }
        return format_map.get(ext, "poscar")

    @staticmethod
    def _parse_cif(content: str) -> Atoms:
        """解析 CIF 文件"""
        from pymatgen.core import Structure

        structure = Structure.from_str(content, "cif")
        return Atoms(
            coords=structure.cart_coords.tolist(),
            elements=[str(site.specie) for site in structure],
            lattice_mat=structure.lattice.matrix.tolist(),
            cartesian=True
        )

    @staticmethod
    def _parse_xyz(content: str) -> Atoms:
        """解析 XYZ 文件"""
        from ase.io import read

        atoms = read(content, format="xyz")
        return Atoms(
            coords=atoms.positions.tolist(),
            elements=atoms.get_chemical_symbols(),
            lattice_mat=atoms.cell.tolist(),
            cartesian=True
        )

    @staticmethod
    def _parse_pdb(content: str) -> Atoms:
        """解析 PDB 文件"""
        from ase.io import read

        atoms = read(content, format="pdb")
        return Atoms(
            coords=atoms.positions.tolist(),
            elements=atoms.get_chemical_symbols(),
            lattice_mat=atoms.cell.tolist(),
            cartesian=True
        )


class Predictor:
    """
    预测器核心类

    负责：
    1. 结构解析
    2. 图构建
    3. 模型推理
    4. 结果后处理
    """

    def __init__(
        self,
        cutoff: float = None,
        max_neighbors: int = None
    ):
        self.cutoff = cutoff or settings.DEFAULT_CUTOFF
        self.max_neighbors = max_neighbors or settings.DEFAULT_MAX_NEIGHBORS

    def predict_single(
        self,
        atoms: Atoms,
        model_name: str,
        return_graph: bool = False
    ) -> Dict[str, Any]:
        """
        使用单个模型预测

        Args:
            atoms: 晶体结构
            model_name: 模型名称
            return_graph: 是否返回图数据

        Returns:
            预测结果字典
        """
        start_time = time.time()

        # 验证模型
        if model_name not in AVAILABLE_MODELS:
            raise PredictionError(f"未知模型: {model_name}")

        # 加载模型
        model = model_manager.load_model(model_name)
        if model is None:
            raise PredictionError(f"模型加载失败: {model_name}")

        # 构建图
        try:
            graph_data = self._build_graph(atoms)
        except Exception as e:
            raise PredictionError(f"图构建失败: {e}")

        # 推理
        try:
            device = model_manager.device
            graph, line_graph = graph_data["graph"], graph_data["line_graph"]
            lattice = torch.tensor(
                atoms.lattice_mat,
                dtype=torch.float32,
                device=device
            )

            with torch.no_grad():
                prediction = model(
                    [graph.to(device), line_graph.to(device), lattice]
                )

            value = prediction.item()

        except Exception as e:
            raise PredictionError(f"模型推理失败: {e}")

        # 释放模型
        model_manager.unload_model(model_name)

        elapsed = time.time() - start_time

        result = {
            "model": model_name,
            "model_name_display": AVAILABLE_MODELS[model_name]["name"],
            "value": value,
            "unit": AVAILABLE_MODELS[model_name]["unit"],
            "processing_time_seconds": round(elapsed, 2),
            "structure_info": self._get_structure_info(atoms),
        }

        if return_graph:
            result["graph"] = graph_data

        return result

    def predict_multiple(
        self,
        atoms: Atoms,
        model_names: List[str]
    ) -> Dict[str, Any]:
        """
        使用多个模型预测

        Args:
            atoms: 晶体结构
            model_names: 模型名称列表

        Returns:
            包含多个模型预测结果的字典
        """
        results = {}
        errors = []

        for model_name in model_names:
            try:
                results[model_name] = self.predict_single(atoms, model_name)
            except PredictionError as e:
                errors.append({
                    "model": model_name,
                    "error": str(e)
                })

        return {
            "predictions": results,
            "errors": errors,
            "structure_info": self._get_structure_info(atoms),
            "total_models": len(model_names),
            "successful": len(results),
            "failed": len(errors)
        }

    def _build_graph(self, atoms: Atoms) -> Dict[str, Any]:
        """构建晶体图"""
        graph, line_graph = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            atom_features="cgcnn",
            compute_line_graph=True,
            use_canonize=True
        )

        return {
            "graph": graph,
            "line_graph": line_graph
        }

    @staticmethod
    def _get_structure_info(atoms: Atoms) -> Dict[str, Any]:
        """获取结构信息"""
        return {
            "formula": atoms.composition.reduced_formula,
            "n_atoms": atoms.num_atoms,
            "n_elements": len(set(atoms.elements)),
            "elements": dict(atoms.composition.fractional_composition.as_dict()),
            "lattice": {
                "a": float(atoms.lattice_mat[0][0]),
                "b": float(atoms.lattice_mat[1][1]),
                "c": float(atoms.lattice_mat[2][2]),
            }
        }


# 便捷函数
def predict_structure(
    structure_content: Union[str, bytes],
    model_names: List[str],
    file_format: str = "poscar",
    cutoff: float = None,
    max_neighbors: int = None
) -> Dict[str, Any]:
    """
    预测晶体结构性质的便捷函数

    Args:
        structure_content: 结构文件内容
        model_names: 模型名称列表
        file_format: 文件格式
        cutoff: 截断半径
        max_neighbors: 最大近邻数

    Returns:
        预测结果
    """
    predictor = Predictor(cutoff=cutoff, max_neighbors=max_neighbors)

    # 解析结构
    atoms = StructureParser.parse(structure_content, file_format)

    # 预测
    return predictor.predict_multiple(atoms, model_names)
