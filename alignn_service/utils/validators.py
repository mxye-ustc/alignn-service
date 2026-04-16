"""输入验证器模块

提供各种输入验证功能，确保数据有效性
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """验证错误"""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"[{field}] {message}")


class Validator:
    """通用验证器"""

    # 元素周期表（用于验证元素符号）
    VALID_ELEMENTS = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    }

    @classmethod
    def validate_file_exists(
        cls,
        file_path: Union[str, Path],
        field: str = "file"
    ) -> Path:
        """验证文件是否存在"""
        path = Path(file_path)

        if not path.exists():
            raise ValidationError(field, f"文件不存在: {file_path}")

        if not path.is_file():
            raise ValidationError(field, f"路径不是文件: {file_path}")

        return path

    @classmethod
    def validate_file_extension(
        cls,
        file_path: Union[str, Path],
        allowed_extensions: List[str],
        field: str = "file"
    ) -> Path:
        """验证文件扩展名"""
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")

        if ext not in [e.lower().lstrip(".") for e in allowed_extensions]:
            raise ValidationError(
                field,
                f"不支持的文件格式: {ext}。"
                f"支持的格式: {', '.join(allowed_extensions)}"
            )

        return path

    @classmethod
    def validate_file_size(
        cls,
        file_path: Union[str, Path],
        max_size_mb: float = 10,
        field: str = "file"
    ) -> Path:
        """验证文件大小"""
        path = Path(file_path)

        if not path.exists():
            raise ValidationError(field, f"文件不存在: {file_path}")

        size_mb = path.stat().st_size / (1024 * 1024)

        if size_mb > max_size_mb:
            raise ValidationError(
                field,
                f"文件过大: {size_mb:.2f}MB。"
                f"最大允许: {max_size_mb}MB"
            )

        return path

    @classmethod
    def validate_element(
        cls,
        element: str,
        field: str = "element"
    ) -> str:
        """验证元素符号"""
        element = element.strip().capitalize()

        if element not in cls.VALID_ELEMENTS:
            raise ValidationError(
                field,
                f"无效的元素符号: {element}"
            )

        return element

    @classmethod
    def validate_elements(
        cls,
        elements: List[str],
        field: str = "elements"
    ) -> List[str]:
        """验证元素列表"""
        if not elements:
            raise ValidationError(field, "元素列表不能为空")

        validated = []
        for elem in elements:
            validated.append(cls.validate_element(elem, field))

        return validated


class StructureValidator:
    """晶体结构验证器"""

    @classmethod
    def validate_poscar(cls, content: str) -> Tuple[bool, Optional[str]]:
        """
        验证 POSCAR 内容

        Returns:
            (是否有效, 错误信息)
        """
        lines = content.strip().split("\n")

        if len(lines) < 7:
            return False, "POSCAR 文件至少需要 7 行"

        try:
            # 第2行：缩放因子
            scale = float(lines[1].strip())
            if scale <= 0:
                return False, "缩放因子必须大于 0"

            # 第3-5行：晶格向量
            for i in range(2, 5):
                parts = lines[i].split()
                if len(parts) != 3:
                    return False, f"第 {i+1} 行晶格向量格式错误"
                for val in parts:
                    float(val)

            # 第6行：元素
            elements = lines[5].strip().split()
            if not elements:
                return False, "第 6 行元素列表为空"

            # 验证元素符号
            for elem in elements:
                elem_clean = elem.strip().capitalize()
                if elem_clean not in Validator.VALID_ELEMENTS:
                    return False, f"无效的元素符号: {elem}"

            # 第7行：原子数量
            natoms = lines[6].strip().split()
            if len(natoms) != len(elements):
                return False, "元素数量与原子数量不匹配"

            for n in natoms:
                if not n.isdigit() or int(n) <= 0:
                    return False, f"原子数量无效: {n}"

            total_atoms = sum(int(n) for n in natoms)

            # 坐标行数检查
            coord_start = 7
            if "Selective" in lines[7] or "s" in lines[7].lower():
                if "Selective" in lines[7]:
                    coord_start = 8

            # 确定是 Direct 还是 Cartesian
            coord_line = lines[coord_start].strip()
            is_direct = "Direct" in coord_line or "direct" in coord_line

            coord_start += 1

            if len(lines) < coord_start + total_atoms:
                return False, f"坐标行数不足，需要 {total_atoms} 行，实际 {len(lines) - coord_start} 行"

            # 验证坐标格式
            for i in range(total_atoms):
                line_idx = coord_start + i
                parts = lines[line_idx].split()[:3]  # 只取前3列
                if len(parts) != 3:
                    return False, f"第 {line_idx+1} 行坐标格式错误"

                for val in parts:
                    float(val)

            return True, None

        except (ValueError, IndexError) as e:
            return False, f"POSCAR 格式错误: {e}"

    @classmethod
    def validate_formula(cls, formula: str) -> Tuple[bool, Optional[str]]:
        """验证化学式格式"""
        # 简单的化学式验证
        pattern = r"^([A-Z][a-z]?\d*)+$"

        if not re.match(pattern, formula):
            return False, f"化学式格式无效: {formula}"

        return True, None

    @classmethod
    def validate_lattice_vectors(
        cls,
        lattice: List[List[float]]
    ) -> Tuple[bool, Optional[str]]:
        """验证晶格向量"""
        if len(lattice) != 3:
            return False, "晶格向量需要 3 行"

        for i, vec in enumerate(lattice):
            if len(vec) != 3:
                return False, f"第 {i+1} 行晶格向量需要 3 列"
            for val in vec:
                if not isinstance(val, (int, float)) or val < 0:
                    return False, f"晶格向量值无效: {val}"

        return True, None


class ModelValidator:
    """模型参数验证器"""

    # 可用的模型列表
    AVAILABLE_MODELS = {
        "jv_formation_energy_peratom_alignn",
        "jv_optb88vdw_bandgap_alignn",
        "jv_mbj_bandgap_alignn",
        "mp_e_form_alignn",
        "mp_gappbe_alignn",
        "jv_bulk_modulus_kv_alignn",
        "jv_shear_modulus_gv_alignn",
    }

    @classmethod
    def validate_model_name(
        cls,
        model_name: str,
        field: str = "model"
    ) -> str:
        """验证模型名称"""
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValidationError(
                field,
                f"未知模型: {model_name}。"
                f"可用模型: {', '.join(cls.AVAILABLE_MODELS)}"
            )

        return model_name

    @classmethod
    def validate_model_list(
        cls,
        model_names: List[str],
        field: str = "models"
    ) -> List[str]:
        """验证模型列表"""
        if not model_names:
            raise ValidationError(field, "模型列表不能为空")

        validated = []
        for name in model_names:
            validated.append(cls.validate_model_name(name, field))

        return validated

    @classmethod
    def validate_cutoff(
        cls,
        cutoff: float,
        min_val: float = 1.0,
        max_val: float = 20.0,
        field: str = "cutoff"
    ) -> float:
        """验证截断半径"""
        if not isinstance(cutoff, (int, float)):
            raise ValidationError(field, "截断半径必须是数字")

        if cutoff < min_val or cutoff > max_val:
            raise ValidationError(
                field,
                f"截断半径超出范围: {cutoff}。"
                f"有效范围: {min_val} - {max_val} 埃"
            )

        return float(cutoff)

    @classmethod
    def validate_max_neighbors(
        cls,
        max_neighbors: int,
        min_val: int = 1,
        max_val: int = 50,
        field: str = "max_neighbors"
    ) -> int:
        """验证最大近邻数"""
        if not isinstance(max_neighbors, int):
            raise ValidationError(field, "最大近邻数必须是整数")

        if max_neighbors < min_val or max_neighbors > max_val:
            raise ValidationError(
                field,
                f"最大近邻数超出范围: {max_neighbors}。"
                f"有效范围: {min_val} - {max_val}"
            )

        return max_neighbors


class DopingConfigValidator:
    """掺杂配置验证器"""

    # 可用的掺杂位点
    VALID_SITES = ["Li", "Fe", "P", "O", "Mn", "Co", "Ni"]

    # 可选的掺杂元素（常见的电池材料掺杂元素）
    COMMON_DOPANTS = [
        "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn",
        "Zr", "Nb", "Mo", "W", "Al", "Mg", "Si", "Ce"
    ]

    @classmethod
    def validate_dopant_element(
        cls,
        element: str,
        field: str = "dopant_element"
    ) -> str:
        """验证掺杂元素"""
        element = Validator.validate_element(element, field)
        return element

    @classmethod
    def validate_doping_site(
        cls,
        site: str,
        field: str = "doping_site"
    ) -> str:
        """验证掺杂位点"""
        site = site.strip()

        if site not in cls.VALID_SITES:
            logger.warning(
                f"非标准掺杂位点: {site}。"
                f"常用位点: {', '.join(cls.VALID_SITES)}"
            )

        return site

    @classmethod
    def validate_concentration(
        cls,
        concentration: float,
        field: str = "concentration"
    ) -> float:
        """验证掺杂浓度"""
        if not isinstance(concentration, (int, float)):
            raise ValidationError(field, "浓度必须是数字")

        if concentration < 0 or concentration > 100:
            raise ValidationError(
                field,
                f"浓度超出范围: {concentration}%。"
                f"有效范围: 0 - 100%"
            )

        return float(concentration)

    @classmethod
    def validate_concentration_range(
        cls,
        min_conc: float,
        max_conc: float,
        step: float = None,
        field: str = "concentration_range"
    ) -> Tuple[float, float, Optional[float]]:
        """验证浓度范围"""
        if min_conc < 0 or min_conc > 100:
            raise ValidationError(field, f"最小浓度超出范围: {min_conc}%")

        if max_conc < 0 or max_conc > 100:
            raise ValidationError(field, f"最大浓度超出范围: {max_conc}%")

        if min_conc > max_conc:
            raise ValidationError(field, "最小浓度不能大于最大浓度")

        if step is not None:
            if step <= 0:
                raise ValidationError(field, "步长必须大于 0")
            if step > (max_conc - min_conc):
                raise ValidationError(field, "步长不能大于浓度范围")

        return float(min_conc), float(max_conc), float(step) if step else None

    @classmethod
    def validate_doping_config(
        cls,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证完整的掺杂配置

        Returns:
            验证后的配置字典
        """
        validated = {}

        # 宿主结构
        if "host_structure" not in config:
            raise ValidationError("host_structure", "缺少宿主结构")

        # 掺杂元素
        if "dopant_element" not in config:
            raise ValidationError("dopant_element", "缺少掺杂元素")
        validated["dopant_element"] = cls.validate_dopant_element(
            config["dopant_element"]
        )

        # 掺杂位点
        if "doping_site" in config:
            validated["doping_site"] = cls.validate_doping_site(
                config["doping_site"]
            )

        # 浓度
        if "concentration" in config:
            validated["concentration"] = cls.validate_concentration(
                config["concentration"]
            )
        elif "min_concentration" in config and "max_concentration" in config:
            min_c, max_c, step = cls.validate_concentration_range(
                config["min_concentration"],
                config["max_concentration"],
                config.get("concentration_step")
            )
            validated["min_concentration"] = min_c
            validated["max_concentration"] = max_c
            validated["concentration_step"] = step

        return validated


class BatchValidator:
    """批量任务验证器"""

    MAX_FILES = 100
    MAX_TOTAL_SIZE_MB = 100

    @classmethod
    def validate_file_count(
        cls,
        file_count: int,
        max_files: int = None,
        field: str = "files"
    ) -> int:
        """验证文件数量"""
        max_files = max_files or cls.MAX_FILES

        if file_count <= 0:
            raise ValidationError(field, "文件列表不能为空")

        if file_count > max_files:
            raise ValidationError(
                field,
                f"文件数量超出限制: {file_count}。"
                f"最大允许: {max_files}"
            )

        return file_count

    @classmethod
    def validate_total_size(
        cls,
        total_size_bytes: int,
        max_size_mb: float = None,
        field: str = "total_size"
    ) -> int:
        """验证总文件大小"""
        max_size_mb = max_size_mb or cls.MAX_TOTAL_SIZE_MB
        total_size_mb = total_size_bytes / (1024 * 1024)

        if total_size_mb > max_size_mb:
            raise ValidationError(
                field,
                f"文件总大小超出限制: {total_size_mb:.2f}MB。"
                f"最大允许: {max_size_mb}MB"
            )

        return total_size_bytes


def validate_structure_file(
    file_path: Union[str, Path],
    max_size_mb: float = 10
) -> Tuple[bool, Optional[str]]:
    """便捷函数：验证结构文件"""
    try:
        Validator.validate_file_exists(file_path)
        Validator.validate_file_extension(
            file_path,
            ["poscar", "vasp", "cif", "xyz", "pdb"]
        )
        Validator.validate_file_size(file_path, max_size_mb)
        return True, None
    except ValidationError as e:
        return False, str(e)


def validate_doping_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """便捷函数：验证掺杂配置"""
    return DopingConfigValidator.validate_doping_config(config)