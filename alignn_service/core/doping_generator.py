"""掺杂构型生成器模块

基于宿主结构生成各种掺杂构型
支持多种掺杂策略：随机掺杂、指定位点掺杂、浓度梯度扫描等
"""

import json
import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from jarvis.core.atoms import Atoms

logger = logging.getLogger(__name__)


class DopingError(Exception):
    """掺杂生成错误"""
    pass


class DopingGenerator:
    """
    掺杂构型生成器

    基于现有的 generate_lfp_dopants_v4.py 逻辑重构，
    提供更灵活的接口和更好的扩展性
    """

    # 宿主结构类型
    HOST_STRUCTURES = {
        "lfp": {
            "formula": "LiFePO4",
            "elements": ["Li", "Fe", "P", "O"],
            "sites": {
                "Li": {"element": "Li", "count": 4, "Wyckoff": "4a"},
                "Fe": {"element": "Fe", "count": 4, "Wyckoff": "4c"},
                "P": {"element": "P", "count": 4, "Wyckoff": "4c"},
                "O": {"element": "O", "count": 16, "Wyckoff": "8d + 4c"},
            }
        },
        "ncm": {
            "formula": "LiNiCoMnO2",
            "elements": ["Li", "Ni", "Co", "Mn", "O"],
            "sites": {
                "Li": {"element": "Li", "count": 1, "Wyckoff": "3a"},
                "Ni": {"element": "Ni", "count": 1/3, "Wyckoff": "3a"},
                "Co": {"element": "Co", "count": 1/3, "Wyckoff": "3a"},
                "Mn": {"element": "Mn", "count": 1/3, "Wyckoff": "3a"},
                "O": {"element": "O", "count": 2, "Wyckoff": "6e"},
            }
        },
        "lco": {
            "formula": "LiCoO2",
            "elements": ["Li", "Co", "O"],
            "sites": {
                "Li": {"element": "Li", "count": 1, "Wyckoff": "3a"},
                "Co": {"element": "Co", "count": 1, "Wyckoff": "3a"},
                "O": {"element": "O", "count": 2, "Wyckoff": "6e"},
            }
        },
        "lmo": {
            "formula": "LiMn2O4",
            "elements": ["Li", "Mn", "O"],
            "sites": {
                "Li": {"element": "Li", "count": 1, "Wyckoff": "8a"},
                "Mn": {"element": "Mn", "count": 2, "Wyckoff": "16d"},
                "O": {"element": "O", "count": 4, "Wyckoff": "8a + 16c"},
            }
        }
    }

    # 常见掺杂元素
    COMMON_DOPANTS = [
        "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn",
        "Zr", "Nb", "Mo", "W", "Al", "Mg", "Si", "Ce", "Y"
    ]

    def __init__(
        self,
        output_dir: Union[str, Path] = None,
        poscar_dir: Union[str, Path] = None
    ):
        """
        初始化掺杂生成器

        Args:
            output_dir: 输出目录
            poscar_dir: POSCAR 文件存储目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path("alignn_service/generated/doping")
        self.poscar_dir = Path(poscar_dir) if poscar_dir else Path("alignn_service/generated/doping/poscar_files")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.poscar_dir.mkdir(parents=True, exist_ok=True)

    def generate_random_doping(
        self,
        host_atoms: Atoms,
        dopant: str,
        doping_site: str,
        n_dopant: int,
        seed: int = None
    ) -> Atoms:
        """
        随机掺杂：随机选择 n_dopant 个原子替换为掺杂元素

        Args:
            host_atoms: 宿主结构
            dopant: 掺杂元素
            doping_site: 掺杂位点（如 "Fe", "Li", "P"）
            n_dopant: 掺杂原子数量
            seed: 随机种子（用于可重复性）

        Returns:
            掺杂后的 Atoms 对象
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 获取指定位点的原子索引
        site_indices = self._get_site_indices(host_atoms, doping_site)

        if len(site_indices) == 0:
            raise DopingError(f"未找到掺杂位点: {doping_site}")

        if n_dopant > len(site_indices):
            raise DopingError(
                f"掺杂数量 {n_dopant} 超过可用位点数 {len(site_indices)}"
            )

        # 随机选择掺杂位置
        dopant_indices = random.sample(site_indices, n_dopant)

        # 替换元素
        new_elements = list(host_atoms.elements)
        for idx in dopant_indices:
            new_elements[idx] = dopant

        # 创建新结构
        doped_atoms = Atoms(
            coords=host_atoms.coords,
            elements=new_elements,
            lattice_mat=host_atoms.lattice_mat,
            cartesian=host_atoms.cartesian
        )

        return doped_atoms

    def generate_concentration_series(
        self,
        host_atoms: Atoms,
        dopant: str,
        doping_site: str,
        concentrations: List[float],
        total_sites: int = None
    ) -> List[Atoms]:
        """
        生成浓度系列：生成不同浓度的掺杂构型

        Args:
            host_atoms: 宿主结构
            dopant: 掺杂元素
            doping_site: 掺杂位点
            concentrations: 浓度列表（百分比）
            total_sites: 位点总数（如果为 None，自动从结构中获取）

        Returns:
            掺杂构型列表
        """
        if total_sites is None:
            total_sites = len(self._get_site_indices(host_atoms, doping_site))

        if total_sites == 0:
            raise DopingError(f"未找到掺杂位点: {doping_site}")

        doped_structures = []

        for conc in concentrations:
            n_dopant = round(total_sites * conc / 100)
            n_dopant = max(1, min(n_dopant, total_sites - 1))

            try:
                doped = self.generate_random_doping(
                    host_atoms,
                    dopant,
                    doping_site,
                    n_dopant
                )
                doped_structures.append(doped)
            except DopingError as e:
                logger.warning(f"生成浓度 {conc}% 失败: {e}")
                continue

        return doped_structures

    def generate_specific_sites(
        self,
        host_atoms: Atoms,
        dopant: str,
        site_indices: List[int]
    ) -> Atoms:
        """
        指定位点掺杂：替换特定的原子位置

        Args:
            host_atoms: 宿主结构
            dopant: 掺杂元素
            site_indices: 要替换的原子索引列表

        Returns:
            掺杂后的 Atoms 对象
        """
        max_idx = host_atoms.num_atoms - 1
        for idx in site_indices:
            if idx < 0 or idx > max_idx:
                raise DopingError(f"原子索引超出范围: {idx}")

        new_elements = list(host_atoms.elements)
        for idx in site_indices:
            new_elements[idx] = dopant

        doped_atoms = Atoms(
            coords=host_atoms.coords,
            elements=new_elements,
            lattice_mat=host_atoms.lattice_mat,
            cartesian=host_atoms.cartesian
        )

        return doped_atoms

    def generate_multiple_dopants(
        self,
        host_atoms: Atoms,
        dopants: Dict[str, float],
        doping_site: str,
        total_dopant_ratio: float = 10.0
    ) -> Atoms:
        """
        多元素共掺杂

        Args:
            host_atoms: 宿主结构
            dopants: 掺杂元素及权重 {元素: 权重}，权重越高被选中的概率越大
            doping_site: 掺杂位点
            total_dopant_ratio: 总掺杂比例（百分比）

        Returns:
            掺杂后的 Atoms 对象
        """
        site_indices = self._get_site_indices(host_atoms, doping_site)
        if len(site_indices) == 0:
            raise DopingError(f"未找到掺杂位点: {doping_site}")

        # 计算各元素分配数量
        n_total = round(len(site_indices) * total_dopant_ratio / 100)
        n_total = max(1, min(n_total, len(site_indices) - 1))

        # 按权重分配
        elements = list(dopants.keys())
        weights = list(dopants.values())
        selected_elements = random.choices(elements, weights=weights, k=n_total)

        # 替换
        new_elements = list(host_atoms.elements)
        for i, idx in enumerate(random.sample(site_indices, n_total)):
            new_elements[idx] = selected_elements[i]

        doped_atoms = Atoms(
            coords=host_atoms.coords,
            elements=new_elements,
            lattice_mat=host_atoms.lattice_mat,
            cartesian=host_atoms.cartesian
        )

        return doped_atoms

    def save_config(
        self,
        atoms: Atoms,
        config_id: str,
        dopant: str,
        doping_site: str,
        n_dopant: int,
        concentration: float = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        保存掺杂构型到文件

        Args:
            atoms: 掺杂后的结构
            config_id: 配置 ID
            dopant: 掺杂元素
            doping_site: 掺杂位点
            n_dopant: 掺杂原子数
            concentration: 浓度百分比
            metadata: 额外元数据

        Returns:
            配置信息字典
        """
        # 生成 POSCAR 文件
        poscar_path = self.poscar_dir / f"{config_id}.poscar"
        poscar_content = atoms.to_poscar()

        with open(poscar_path, "w") as f:
            f.write(poscar_content)

        # 构建配置信息
        config_info = {
            "config_id": config_id,
            "dopant_element": dopant,
            "doping_site": doping_site,
            "n_dopant": n_dopant,
            "concentration_pct": concentration,
            "formula": atoms.composition.reduced_formula,
            "n_atoms": atoms.num_atoms,
            "poscar_path": str(poscar_path),
            "created_at": datetime.now().isoformat(),
        }

        if metadata:
            config_info.update(metadata)

        # 保存元数据
        meta_path = self.output_dir / f"{config_id}.json"
        with open(meta_path, "w") as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)

        return config_info

    def generate_batch(
        self,
        host_atoms: Atoms,
        dopants: List[str],
        doping_site: str,
        concentrations: List[float],
        n_configs_per_combination: int = 3,
        save: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量生成掺杂构型

        Args:
            host_atoms: 宿主结构
            dopants: 掺杂元素列表
            doping_site: 掺杂位点
            concentrations: 浓度列表
            n_configs_per_combination: 每种组合生成的构型数
            save: 是否保存到文件

        Returns:
            配置信息列表
        """
        configs = []
        total_sites = len(self._get_site_indices(host_atoms, doping_site))

        for dopant in dopants:
            for conc in concentrations:
                n_dopant = max(1, round(total_sites * conc / 100))

                for i in range(n_configs_per_combination):
                    try:
                        doped = self.generate_random_doping(
                            host_atoms,
                            dopant,
                            doping_site,
                            n_dopant,
                            seed=hash((dopant, conc, i)) % (2**32)
                        )

                        config_id = f"{host_atoms.composition.reduced_formula}_{dopant}_{int(conc)}pct_{i+1}"

                        if save:
                            config = self.save_config(
                                atoms=doped,
                                config_id=config_id,
                                dopant=dopant,
                                doping_site=doping_site,
                                n_dopant=n_dopant,
                                concentration=conc,
                                metadata={"batch_index": i + 1}
                            )
                        else:
                            config = {
                                "config_id": config_id,
                                "atoms": doped,
                                "dopant_element": dopant,
                                "doping_site": doping_site,
                                "n_dopant": n_dopant,
                                "concentration_pct": conc,
                                "formula": doped.composition.reduced_formula,
                            }

                        configs.append(config)

                    except DopingError as e:
                        logger.warning(f"生成失败: {dopant}@{conc}% config {i+1}: {e}")
                        continue

        logger.info(f"批量生成完成: {len(configs)} 个构型")
        return configs

    def _get_site_indices(self, atoms: Atoms, site: str) -> List[int]:
        """获取指定位点的原子索引"""
        indices = []
        for i, elem in enumerate(atoms.elements):
            if elem.strip().capitalize() == site.strip().capitalize():
                indices.append(i)
        return indices

    def get_available_sites(self, atoms: Atoms) -> Dict[str, int]:
        """获取结构中可用的掺杂位点及其数量"""
        sites = {}
        for elem in atoms.elements:
            elem = elem.strip().capitalize()
            sites[elem] = sites.get(elem, 0) + 1
        return sites


class LFPGenerator(DopingGenerator):
    """
    LiFePO4 专用掺杂生成器

    继承自 DopingGenerator，提供 LFP 特定的便捷方法
    """

    # LFP 结构参数（标准橄榄石结构）
    LFP_LATTICE = {
        "a": 10.329,
        "b": 6.007,
        "c": 4.692,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.host_structure = "lfp"

    def generate_ti_doping(
        self,
        host_atoms: Atoms,
        concentration: float,
        n_configs: int = 1
    ) -> List[Dict[str, Any]]:
        """生成 Ti 掺杂 LFP"""
        return self.generate_batch(
            host_atoms=host_atoms,
            dopants=["Ti"],
            doping_site="Fe",
            concentrations=[concentration],
            n_configs_per_combination=n_configs
        )

    def generate_v_doping(
        self,
        host_atoms: Atoms,
        concentration: float,
        n_configs: int = 1
    ) -> List[Dict[str, Any]]:
        """生成 V 掺杂 LFP"""
        return self.generate_batch(
            host_atoms=host_atoms,
            dopants=["V"],
            doping_site="Fe",
            concentrations=[concentration],
            n_configs_per_combination=n_configs
        )

    def generate_mn_doping(
        self,
        host_atoms: Atoms,
        concentration: float,
        n_configs: int = 1
    ) -> List[Dict[str, Any]]:
        """生成 Mn 掺杂 LFP（替代 Fe 位）"""
        return self.generate_batch(
            host_atoms=host_atoms,
            dopants=["Mn"],
            doping_site="Fe",
            concentrations=[concentration],
            n_configs_per_combination=n_configs
        )

    def generate_co_doping(
        self,
        host_atoms: Atoms,
        concentration: float,
        n_configs: int = 1
    ) -> List[Dict[str, Any]]:
        """生成 Co 掺杂 LFP"""
        return self.generate_batch(
            host_atoms=host_atoms,
            dopants=["Co"],
            doping_site="Fe",
            concentrations=[concentration],
            n_configs_per_combination=n_configs
        )

    def generate_multi_element_doping(
        self,
        host_atoms: Atoms,
        dopants: Dict[str, float],
        total_concentration: float
    ) -> Dict[str, Any]:
        """
        生成多元素共掺杂 LFP

        Args:
            host_atoms: 宿主 LFP 结构
            dopants: 掺杂元素及权重 {元素: 权重}
            total_concentration: 总掺杂浓度（百分比）

        Returns:
            配置信息
        """
        doped = self.generate_multiple_dopants(
            host_atoms,
            dopants,
            "Fe",
            total_concentration
        )

        config_id = f"LFP_{'-'.join(dopants.keys())}_{int(total_concentration)}pct"

        return self.save_config(
            atoms=doped,
            config_id=config_id,
            dopant="+".join(dopants.keys()),
            doping_site="Fe",
            n_dopant=round(host_atoms.num_atoms * total_concentration / 100),
            concentration=total_concentration,
            metadata={"type": "co_doping", "dopants": dopants}
        )


def generate_lfp_dopants(
    host_poscar_path: Union[str, Path],
    dopants: List[str],
    site: str = "Fe",
    concentrations: List[float] = None,
    output_dir: Union[str, Path] = None
) -> List[Dict[str, Any]]:
    """
    便捷函数：生成 LFP 掺杂构型

    Args:
        host_poscar_path: 宿主 POSCAR 路径
        dopants: 掺杂元素列表
        site: 掺杂位点
        concentrations: 浓度列表
        output_dir: 输出目录

    Returns:
        配置信息列表
    """
    from alignn_service.utils.file_parser import FileParser

    # 读取宿主结构
    atoms = FileParser.parse_file(host_poscar_path)

    # 默认浓度
    if concentrations is None:
        concentrations = [1, 2, 5, 10]

    # 创建生成器
    generator = LFPGenerator(output_dir=output_dir)

    # 生成
    return generator.generate_batch(
        host_atoms=atoms,
        dopants=dopants,
        doping_site=site,
        concentrations=concentrations
    )
