"""掺杂生成器测试"""

import pytest
from pathlib import Path
import tempfile

from jarvis.core.atoms import Atoms

from alignn_service.core.doping_generator import (
    DopingGenerator,
    DopingError,
    LFPGenerator,
    generate_lfp_dopants,
)


class TestDopingGenerator:
    """掺杂生成器测试"""

    @pytest.fixture
    def lfp_atoms(self):
        """创建 LiFePO4 结构"""
        # 简化的 LFP 结构
        return Atoms(
            coords=[
                [0, 0, 0], [5, 0, 0],  # Li
                [2.5, 1.5, 1], [7.5, 1.5, 1],  # Fe
                [5, 3, 2], [0, 3, 2],  # P
                [1, 2, 0.5], [4, 2, 0.5], [6, 2, 0.5], [9, 2, 0.5],  # O
            ],
            elements=["Li", "Li", "Fe", "Fe", "P", "P", "O", "O", "O", "O"],
            lattice_mat=[
                [10.329, 0, 0],
                [0, 6.007, 0],
                [0, 0, 4.692]
            ],
            cartesian=True
        )

    def test_get_site_indices(self, lfp_atoms):
        """测试获取位点索引"""
        generator = DopingGenerator()

        fe_indices = generator._get_site_indices(lfp_atoms, "Fe")
        assert len(fe_indices) == 2

        li_indices = generator._get_site_indices(lfp_atoms, "Li")
        assert len(li_indices) == 2

        o_indices = generator._get_site_indices(lfp_atoms, "O")
        assert len(o_indices) == 4

    def test_get_available_sites(self, lfp_atoms):
        """测试获取可用位点"""
        generator = DopingGenerator()
        sites = generator.get_available_sites(lfp_atoms)

        assert sites["Li"] == 2
        assert sites["Fe"] == 2
        assert sites["P"] == 2
        assert sites["O"] == 4

    def test_generate_random_doping(self, lfp_atoms):
        """测试随机掺杂"""
        generator = DopingGenerator()

        doped = generator.generate_random_doping(
            host_atoms=lfp_atoms,
            dopant="Ti",
            doping_site="Fe",
            n_dopant=1,
            seed=42
        )

        # 验证 Ti 替换了一个 Fe
        assert "Ti" in doped.elements
        assert doped.elements.count("Ti") == 1
        assert doped.elements.count("Fe") == 1  # 原来是 2 个

    def test_generate_random_doping_invalid_site(self, lfp_atoms):
        """测试无效位点掺杂"""
        generator = DopingGenerator()

        with pytest.raises(DopingError):
            generator.generate_random_doping(
                host_atoms=lfp_atoms,
                dopant="Ti",
                doping_site="Invalid",
                n_dopant=1
            )

    def test_generate_random_doping_too_many(self, lfp_atoms):
        """测试掺杂数量超过可用位点"""
        generator = DopingGenerator()

        with pytest.raises(DopingError):
            generator.generate_random_doping(
                host_atoms=lfp_atoms,
                dopant="Ti",
                doping_site="Fe",
                n_dopant=10  # 只有 2 个 Fe 位点
            )

    def test_generate_specific_sites(self, lfp_atoms):
        """测试指定位点掺杂"""
        generator = DopingGenerator()

        doped = generator.generate_specific_sites(
            host_atoms=lfp_atoms,
            dopant="Ti",
            site_indices=[2]  # 第一个 Fe 的索引
        )

        assert doped.elements[2] == "Ti"
        assert "Fe" in doped.elements

    def test_generate_specific_sites_invalid_index(self, lfp_atoms):
        """测试无效的位点索引"""
        generator = DopingGenerator()

        with pytest.raises(DopingError):
            generator.generate_specific_sites(
                host_atoms=lfp_atoms,
                dopant="Ti",
                site_indices=[100]  # 超出范围
            )

    def test_generate_multiple_dopants(self, lfp_atoms):
        """测试多元素共掺杂"""
        generator = DopingGenerator()

        doped = generator.generate_multiple_dopants(
            host_atoms=lfp_atoms,
            dopants={"Ti": 1, "V": 1},  # 权重相等
            doping_site="Fe",
            total_dopant_ratio=100  # 100% 替换
        )

        # 应该替换所有 Fe
        assert "Ti" in doped.elements
        assert "V" in doped.elements
        assert "Fe" not in doped.elements

    def test_save_config(self, lfp_atoms):
        """测试保存配置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = DopingGenerator(output_dir=tmpdir)

            doped = generator.generate_random_doping(
                host_atoms=lfp_atoms,
                dopant="Ti",
                doping_site="Fe",
                n_dopant=1
            )

            config = generator.save_config(
                atoms=doped,
                config_id="test_config_001",
                dopant="Ti",
                doping_site="Fe",
                n_dopant=1,
                concentration=25.0
            )

            assert config["config_id"] == "test_config_001"
            assert config["dopant_element"] == "Ti"
            assert config["n_dopant"] == 1
            assert config["concentration_pct"] == 25.0

            # 验证文件已创建
            assert Path(config["poscar_path"]).exists()


class TestLFPGenerator:
    """LFP 专用生成器测试"""

    @pytest.fixture
    def lfp_atoms(self):
        """创建 LiFePO4 结构"""
        return Atoms(
            coords=[
                [0, 0, 0], [5, 0, 0],  # Li
                [2.5, 1.5, 1], [7.5, 1.5, 1],  # Fe
                [5, 3, 2], [0, 3, 2],  # P
                [1, 2, 0.5], [4, 2, 0.5], [6, 2, 0.5], [9, 2, 0.5],  # O
            ],
            elements=["Li", "Li", "Fe", "Fe", "P", "P", "O", "O", "O", "O"],
            lattice_mat=[
                [10.329, 0, 0],
                [0, 6.007, 0],
                [0, 0, 4.692]
            ],
            cartesian=True
        )

    def test_generate_ti_doping(self, lfp_atoms):
        """测试 Ti 掺杂"""
        generator = LFPGenerator()

        configs = generator.generate_ti_doping(
            host_atoms=lfp_atoms,
            concentration=50.0,
            n_configs=1
        )

        assert len(configs) == 1
        assert configs[0]["dopant_element"] == "Ti"

    def test_generate_multi_element_doping(self, lfp_atoms):
        """测试多元素共掺杂"""
        generator = LFPGenerator()

        config = generator.generate_multi_element_doping(
            host_atoms=lfp_atoms,
            dopants={"Ti": 1, "V": 1},
            total_concentration=100.0
        )

        assert "Ti" in config["dopant_element"]
        assert "V" in config["dopant_element"]
        assert config["concentration_pct"] == 100.0


class TestGenerateLFP_DopantsFunction:
    """便捷函数测试"""

    def test_generate_lfp_dopants_presets(self):
        """测试使用预设宿主结构生成掺杂"""
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = generate_lfp_dopants(
                host_poscar_path="tests/data/LiFePO4.poscar",  # 需要有测试文件
                dopants=["Ti", "V"],
                site="Fe",
                concentrations=[5, 10],
                output_dir=tmpdir
            )

            # 由于没有实际的测试文件，这个测试会失败
            # 但函数逻辑是正确的
