"""工具模块测试"""

import pytest
from pathlib import Path
import tempfile
import os

from alignn_service.utils.validators import (
    Validator,
    ValidationError,
    StructureValidator,
    ModelValidator,
    DopingConfigValidator,
    BatchValidator,
)
from alignn_service.utils.file_parser import FileParser, FileParseError


class TestValidator:
    """验证器测试"""

    def test_validate_element_valid(self):
        """测试有效元素"""
        assert Validator.validate_element("Li") == "Li"
        assert Validator.validate_element("Fe") == "Fe"
        assert Validator.validate_element("fe") == "Fe"  # 大小写不敏感

    def test_validate_element_invalid(self):
        """测试无效元素"""
        with pytest.raises(ValidationError):
            Validator.validate_element("Xx")

    def test_validate_elements_list(self):
        """测试元素列表"""
        elements = Validator.validate_elements(["Li", "Fe", "P", "O"])
        assert elements == ["Li", "Fe", "P", "O"]

    def test_validate_elements_empty(self):
        """测试空元素列表"""
        with pytest.raises(ValidationError):
            Validator.validate_elements([])

    def test_validate_file_extension(self):
        """测试文件扩展名验证"""
        with tempfile.NamedTemporaryFile(suffix=".poscar", delete=False) as f:
            path = Path(f.name)
            result = Validator.validate_file_extension(path, ["poscar", "cif"])
            assert result == path
            os.unlink(f.name)

    def test_validate_file_extension_invalid(self):
        """测试无效文件扩展名"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
            with pytest.raises(ValidationError):
                Validator.validate_file_extension(path, ["poscar", "cif"])
            os.unlink(f.name)


class TestStructureValidator:
    """结构验证器测试"""

    def test_validate_poscar_valid(self, sample_poscar):
        """测试有效 POSCAR"""
        is_valid, error = StructureValidator.validate_poscar(sample_poscar)
        assert is_valid is True
        assert error is None

    def test_validate_poscar_invalid_too_short(self):
        """测试过短的 POSCAR"""
        is_valid, error = StructureValidator.validate_poscar("LiFePO4\n1.0")
        assert is_valid is False
        assert "至少需要 7 行" in error

    def test_validate_poscar_invalid_scale(self, sample_poscar):
        """测试无效缩放因子"""
        lines = sample_poscar.split("\n")
        lines[1] = "-1.0"  # 负数缩放因子
        content = "\n".join(lines)

        is_valid, error = StructureValidator.validate_poscar(content)
        assert is_valid is False
        assert "缩放因子必须大于 0" in error

    def test_validate_formula_valid(self):
        """测试有效化学式"""
        is_valid, error = StructureValidator.validate_formula("LiFePO4")
        assert is_valid is True

    def test_validate_formula_invalid(self):
        """测试无效化学式"""
        is_valid, error = StructureValidator.validate_formula("Li@Fe")
        assert is_valid is False


class TestModelValidator:
    """模型验证器测试"""

    def test_validate_model_name_valid(self):
        """测试有效模型名称"""
        result = ModelValidator.validate_model_name("jv_formation_energy_peratom_alignn")
        assert result == "jv_formation_energy_peratom_alignn"

    def test_validate_model_name_invalid(self):
        """测试无效模型名称"""
        with pytest.raises(ValidationError):
            ModelValidator.validate_model_name("invalid_model")

    def test_validate_model_list(self):
        """测试模型列表"""
        models = ModelValidator.validate_model_list([
            "jv_formation_energy_peratom_alignn",
            "jv_optb88vdw_bandgap_alignn"
        ])
        assert len(models) == 2

    def test_validate_cutoff_valid(self):
        """测试有效截断半径"""
        result = ModelValidator.validate_cutoff(8.0)
        assert result == 8.0

    def test_validate_cutoff_out_of_range(self):
        """测试超出范围的截断半径"""
        with pytest.raises(ValidationError):
            ModelValidator.validate_cutoff(50.0)  # 超出 1-20 范围

    def test_validate_max_neighbors_valid(self):
        """测试有效的最大近邻数"""
        result = ModelValidator.validate_max_neighbors(16)
        assert result == 16

    def test_validate_max_neighbors_invalid(self):
        """测试无效的最大近邻数"""
        with pytest.raises(ValidationError):
            ModelValidator.validate_max_neighbors(100)  # 超出 1-50 范围


class TestDopingConfigValidator:
    """掺杂配置验证器测试"""

    def test_validate_dopant_element_valid(self):
        """测试有效掺杂元素"""
        result = DopingConfigValidator.validate_dopant_element("Ti")
        assert result == "Ti"

    def test_validate_dopant_element_invalid(self):
        """测试无效掺杂元素"""
        with pytest.raises(ValidationError):
            DopingConfigValidator.validate_dopant_element("Xx")

    def test_validate_concentration_valid(self):
        """测试有效浓度"""
        result = DopingConfigValidator.validate_concentration(10.0)
        assert result == 10.0

    def test_validate_concentration_out_of_range(self):
        """测试超出范围的浓度"""
        with pytest.raises(ValidationError):
            DopingConfigValidator.validate_concentration(150.0)  # > 100%

    def test_validate_concentration_negative(self):
        """测试负浓度"""
        with pytest.raises(ValidationError):
            DopingConfigValidator.validate_concentration(-5.0)

    def test_validate_doping_config_complete(self):
        """测试完整的掺杂配置"""
        config = {
            "host_structure": "LiFePO4",
            "dopant_element": "Ti",
            "doping_site": "Fe",
            "concentration": 5.0
        }
        result = DopingConfigValidator.validate_doping_config(config)
        assert result["dopant_element"] == "Ti"
        assert result["concentration"] == 5.0

    def test_validate_doping_config_missing_host(self):
        """测试缺少宿主结构的配置"""
        config = {
            "dopant_element": "Ti",
            "concentration": 5.0
        }
        with pytest.raises(ValidationError):
            DopingConfigValidator.validate_doping_config(config)


class TestBatchValidator:
    """批量任务验证器测试"""

    def test_validate_file_count_valid(self):
        """测试有效的文件数量"""
        result = BatchValidator.validate_file_count(50)
        assert result == 50

    def test_validate_file_count_exceeds_limit(self):
        """测试超出限制的文件数量"""
        with pytest.raises(ValidationError):
            BatchValidator.validate_file_count(150)

    def test_validate_file_count_empty(self):
        """测试空文件列表"""
        with pytest.raises(ValidationError):
            BatchValidator.validate_file_count(0)

    def test_validate_total_size_valid(self):
        """测试有效的总大小"""
        result = BatchValidator.validate_total_size(50 * 1024 * 1024)  # 50 MB
        assert result == 50 * 1024 * 1024

    def test_validate_total_size_exceeds_limit(self):
        """测试超出限制的总大小"""
        with pytest.raises(ValidationError):
            BatchValidator.validate_total_size(200 * 1024 * 1024)  # 200 MB > 100 MB


class TestFileParser:
    """文件解析器测试"""

    def test_parse_poscar_content(self, sample_poscar):
        """测试解析 POSCAR 内容"""
        atoms = FileParser.parse_content(sample_poscar, file_format="poscar")
        assert atoms is not None
        assert atoms.num_atoms == 28
        assert "Li" in atoms.elements

    def test_parse_content_bytes(self, sample_poscar):
        """测试解析字节内容"""
        content_bytes = sample_poscar.encode("utf-8")
        atoms = FileParser.parse_content(content_bytes, file_format="poscar")
        assert atoms is not None

    def test_parse_unsupported_format(self, sample_poscar):
        """测试不支持的格式"""
        with pytest.raises(FileParseError):
            FileParser.parse_content(sample_poscar, file_format="xyz123")

    def test_detect_format_poscar(self):
        """测试检测 POSCAR 格式"""
        assert FileParser.detect_format("structure.poscar") == "poscar"
        assert FileParser.detect_format("structure.vasp") == "poscar"

    def test_detect_format_cif(self):
        """测试检测 CIF 格式"""
        assert FileParser.detect_format("structure.cif") == "cif"
