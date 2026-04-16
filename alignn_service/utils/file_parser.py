"""文件解析工具模块

提供统一的文件解析接口，支持多种晶体结构格式
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from jarvis.core.atoms import Atoms

logger = logging.getLogger(__name__)


class FileParseError(Exception):
    """文件解析错误"""
    pass


class FileParser:
    """文件解析器"""

    SUPPORTED_FORMATS = ["poscar", "vasp", "cif", "xyz", "pdb"]

    # 格式扩展名映射
    FORMAT_EXTENSIONS = {
        "poscar": [".poscar", ".vasp", ".POSCAR", ".VASP"],
        "cif": [".cif", ".CIF"],
        "xyz": [".xyz", ".XYZ"],
        "pdb": [".pdb", ".PDB"],
    }

    @classmethod
    def parse_file(cls, file_path: Union[str, Path]) -> Atoms:
        """解析文件，自动检测格式"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileParseError(f"文件不存在: {file_path}")

        file_format = cls.detect_format(file_path.name)
        content = file_path.read_text(encoding="utf-8")

        return cls.parse_content(content, file_format)

    @classmethod
    def parse_content(
        cls,
        content: Union[str, bytes],
        file_format: str = "poscar"
    ) -> Atoms:
        """解析文件内容"""
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        file_format = file_format.lower()

        if file_format not in cls.SUPPORTED_FORMATS:
            raise FileParseError(
                f"不支持的格式: {file_format}。"
                f"支持的格式: {', '.join(cls.SUPPORTED_FORMATS)}"
            )

        try:
            if file_format in ("poscar", "vasp"):
                return Atoms.from_poscar(content)
            elif file_format == "cif":
                return cls._parse_cif(content)
            elif file_format == "xyz":
                return cls._parse_xyz(content)
            elif file_format == "pdb":
                return cls._parse_pdb(content)
            else:
                raise FileParseError(f"无法解析格式: {file_format}")

        except FileParseError:
            raise
        except Exception as e:
            raise FileParseError(f"解析失败: {e}")

    @staticmethod
    def detect_format(filename: str) -> str:
        """根据文件名检测格式"""
        ext = Path(filename).suffix.lower().lstrip(".")

        format_map = {
            "poscar": "poscar",
            "vasp": "poscar",
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


class POSCARParser:
    """POSCAR 专用解析器，提供更细粒度的解析能力"""

    @staticmethod
    def parse_header(content: str) -> Dict[str, Any]:
        """解析 POSCAR 头部信息"""
        lines = content.strip().split("\n")

        # 第1行: 注释
        comment = lines[0].strip()

        # 第2行: 缩放因子
        scale = float(lines[1].strip())

        # 第3-5行: 晶格向量
        lattice = []
        for i in range(2, 5):
            lattice.append([float(x) for x in lines[i].split()])

        # 第6行: 元素列表
        elements = lines[5].strip().split()

        # 第7行: 原子数量
        natoms = [int(x) for x in lines[6].split()]

        return {
            "comment": comment,
            "scale": scale,
            "lattice": lattice,
            "elements": elements,
            "natoms": natoms,
            "total_atoms": sum(natoms)
        }

    @staticmethod
    def parse_coordinates(
        content: str,
        coord_type: str = "direct"
    ) -> Tuple[List[str], List[List[float]]]:
        """
        解析原子坐标

        Args:
            content: POSCAR 内容
            coord_type: 坐标系类型 ("direct" 或 "cartesian")

        Returns:
            (元素列表, 坐标列表)
        """
        lines = content.strip().split("\n")

        # 跳过前7行（到原子数量行）
        idx = 7

        # 检查是否有选择性动力学
        selective = False
        if any(c.lower() == "s" for c in lines[idx].strip()):
            if "Selective" in lines[idx] or "s" in lines[idx].lower():
                selective = True
                idx += 1

        # 确定坐标系
        if "Direct" in lines[idx] or "direct" in lines[idx]:
            coord_type = "direct"
        elif "Cartesian" in lines[idx] or "cartesian" in lines[idx]:
            coord_type = "cartesian"

        idx += 1

        # 解析元素和数量
        elements = []
        natoms = [int(x) for x in lines[6].split()]
        for elem, n in zip(lines[5].split(), natoms):
            elements.extend([elem] * n)

        # 解析坐标
        coords = []
        for i in range(idx, idx + sum(natoms)):
            parts = lines[i].split()
            coords.append([float(x) for x in parts[:3]])

        return elements, coords

    @staticmethod
    def format_poscar(
        elements: List[str],
        coords: List[List[float]],
        lattice: List[List[float]],
        coord_type: str = "Direct",
        comment: str = "Generated by ALIGNN"
    ) -> str:
        """生成 POSCAR 格式字符串"""
        # 统计元素和数量
        unique_elements = []
        element_counts = []
        for elem in elements:
            if elem not in unique_elements:
                unique_elements.append(elem)
                element_counts.append(0)
            element_counts[unique_elements.index(elem)] += 1

        # 构建字符串
        lines = [
            comment,
            "1.0",
            f"  {'  '.join([f'{v:.16f}' for v in lattice[0]])}",
            f"  {'  '.join([f'{v:.16f}' for v in lattice[1]])}",
            f"  {'  '.join([f'{v:.16f}' for v in lattice[2]])}",
            f"  {'  '.join(unique_elements)}",
            f"  {'  '.join(str(c) for c in element_counts)}",
            coord_type,
        ]

        # 添加坐标
        for coord in coords:
            lines.append(f"  {'  '.join([f'{v:.16f}' for v in coord])}")

        return "\n".join(lines)


class BatchFileParser:
    """批量文件解析器"""

    def __init__(self, max_files: int = 100):
        self.max_files = max_files

    def parse_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.poscar"
    ) -> Dict[str, Atoms]:
        """解析目录中的所有结构文件"""
        directory = Path(directory)

        files = list(directory.glob(pattern))
        if len(files) > self.max_files:
            files = files[:self.max_files]
            logger.warning(f"文件数量超过限制，只解析前 {self.max_files} 个")

        results = {}
        errors = []

        for file_path in files:
            try:
                atoms = FileParser.parse_file(file_path)
                results[file_path.name] = atoms
            except Exception as e:
                errors.append({"file": file_path.name, "error": str(e)})

        return {
            "success": results,
            "errors": errors,
            "total": len(files),
            "parsed": len(results),
            "failed": len(errors)
        }

    def parse_files(
        self,
        file_paths: List[Union[str, Path]]
    ) -> Dict[str, Atoms]:
        """解析指定的文件列表"""
        results = {}
        errors = []

        for file_path in file_paths:
            try:
                atoms = FileParser.parse_file(file_path)
                results[str(file_path)] = atoms
            except Exception as e:
                errors.append({"file": str(file_path), "error": str(e)})

        return {
            "success": results,
            "errors": errors
        }


class CSVExporter:
    """结果导出器"""

    @staticmethod
    def export_predictions(
        predictions: List[Dict[str, Any]],
        output_path: Union[str, Path],
        format: str = "csv"
    ) -> str:
        """
        导出预测结果

        Args:
            predictions: 预测结果列表
            output_path: 输出路径
            format: 导出格式 ("csv", "excel", "json")

        Returns:
            输出文件路径
        """
        output_path = Path(output_path)
        format = format.lower()

        if format == "csv":
            return CSVExporter._export_csv(predictions, output_path)
        elif format in ("excel", "xlsx"):
            return CSVExporter._export_excel(predictions, output_path)
        elif format == "json":
            return CSVExporter._export_json(predictions, output_path)
        else:
            raise ValueError(f"不支持的格式: {format}")

    @staticmethod
    def _export_csv(predictions: List[Dict], output_path: Path) -> str:
        """导出为 CSV 格式"""
        rows = []

        for pred in predictions:
            row = {
                "filename": pred.get("filename", ""),
                "formula": pred.get("formula", ""),
                "n_atoms": pred.get("n_atoms", ""),
            }

            # 添加预测值
            for model_name, result in pred.get("predictions", {}).items():
                if isinstance(result, dict) and "value" in result:
                    row[f"{model_name}_value"] = result["value"]
                    row[f"{model_name}_unit"] = result.get("unit", "")

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        return str(output_path)

    @staticmethod
    def _export_excel(predictions: List[Dict], output_path: Path) -> str:
        """导出为 Excel 格式"""
        rows = []

        for pred in predictions:
            row = {
                "文件名": pred.get("filename", ""),
                "化学式": pred.get("formula", ""),
                "原子数": pred.get("n_atoms", ""),
            }

            for model_name, result in pred.get("predictions", {}).items():
                if isinstance(result, dict) and "value" in result:
                    row[f"{model_name}_值"] = result["value"]
                    row[f"{model_name}_单位"] = result.get("unit", "")

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False, engine="openpyxl")

        return str(output_path)

    @staticmethod
    def _export_json(predictions: List[Dict], output_path: Path) -> str:
        """导出为 JSON 格式"""
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        return str(output_path)


def quick_parse(file_path: Union[str, Path]) -> Atoms:
    """快速解析文件的便捷函数"""
    return FileParser.parse_file(file_path)


def quick_export(
    predictions: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = "csv"
) -> str:
    """快速导出预测结果的便捷函数"""
    return CSVExporter.export_predictions(predictions, output_path, format)