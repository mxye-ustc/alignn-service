"""CSV 导出工具测试"""

import pytest
import tempfile
import os
import json
from pathlib import Path

from alignn_service.utils.csv_exporter import (
    PredictionExporter,
    BatchResultExporter,
    DopingConfigExporter,
    CSVExportError,
)


class TestPredictionExporter:
    """预测结果导出器测试"""

    @pytest.fixture
    def sample_predictions(self):
        """示例预测结果"""
        return [
            {
                "filename": "LiFePO4.poscar",
                "formula": "LiFePO4",
                "n_atoms": 28,
                "task_id": "task_001",
                "predictions": {
                    "jv_formation_energy_peratom_alignn": {
                        "value": -2.532,
                        "unit": "eV/atom",
                        "processing_time": 12.34
                    },
                    "jv_optb88vdw_bandgap_alignn": {
                        "value": 3.694,
                        "unit": "eV",
                        "processing_time": 11.87
                    }
                }
            },
            {
                "filename": "LiTiPO4.poscar",
                "formula": "LiTiPO4",
                "n_atoms": 28,
                "task_id": "task_002",
                "predictions": {
                    "jv_formation_energy_peratom_alignn": {
                        "value": -2.123,
                        "unit": "eV/atom",
                        "processing_time": 13.45
                    }
                }
            }
        ]

    def test_export_csv(self, sample_predictions):
        """测试 CSV 导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.csv"

            result = PredictionExporter.export(
                sample_predictions,
                output_path,
                format="csv"
            )

            assert output_path.exists()
            assert result == str(output_path)

            # 验证内容
            content = output_path.read_text()
            assert "filename" in content
            assert "LiFePO4" in content
            assert "-2.532" in content

    def test_export_json(self, sample_predictions):
        """测试 JSON 导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"

            result = PredictionExporter.export(
                sample_predictions,
                output_path,
                format="json"
            )

            assert output_path.exists()

            # 验证内容
            with open(output_path) as f:
                data = json.load(f)
                assert len(data) == 2
                assert data[0]["formula"] == "LiFePO4"

    def test_export_unsupported_format(self, sample_predictions):
        """测试不支持的格式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.xyz"

            with pytest.raises(CSVExportError):
                PredictionExporter.export(
                    sample_predictions,
                    output_path,
                    format="xyz"
                )

    def test_export_empty_data(self):
        """测试导出空数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty.csv"

            with pytest.raises(CSVExportError):
                PredictionExporter.export([], output_path, format="csv")

    def test_export_to_string_csv(self, sample_predictions):
        """测试导出为 CSV 字符串"""
        csv_string = PredictionExporter.export_to_string(
            sample_predictions,
            format="csv"
        )

        assert "filename" in csv_string
        assert "LiFePO4" in csv_string

    def test_export_to_string_json(self, sample_predictions):
        """测试导出为 JSON 字符串"""
        json_string = PredictionExporter.export_to_string(
            sample_predictions,
            format="json"
        )

        data = json.loads(json_string)
        assert len(data) == 2


class TestBatchResultExporter:
    """批量结果导出器测试"""

    @pytest.fixture
    def sample_batch_result(self):
        """示例批量结果"""
        return {
            "task_id": "batch_001",
            "status": "completed",
            "total_structures": 10,
            "successful": 9,
            "failed": 1,
            "total_time_seconds": 120.5,
            "results": [
                {
                    "filename": "config1.poscar",
                    "formula": "LiFePO4",
                    "predictions": {
                        "jv_formation_energy_peratom_alignn": {
                            "value": -2.5,
                            "unit": "eV/atom"
                        }
                    }
                }
            ],
            "errors": [
                {
                    "filename": "config10.poscar",
                    "error": "Invalid format"
                }
            ]
        }

    def test_export_batch_summary(self, sample_batch_result):
        """测试批量结果汇总导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            files = BatchResultExporter.export_batch_summary(
                sample_batch_result,
                output_dir
            )

            assert "csv" in files
            assert "json" in files
            assert Path(files["csv"]).exists()
            assert Path(files["json"]).exists()

    def test_export_errors(self):
        """测试错误报告导出"""
        errors = [
            {"filename": "test1.poscar", "error": "Parse error"},
            {"filename": "test2.poscar", "error": "Invalid lattice"}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "errors.csv"

            result = BatchResultExporter.export_errors(errors, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "test1.poscar" in content
            assert "Parse error" in content


class TestDopingConfigExporter:
    """掺杂配置导出器测试"""

    @pytest.fixture
    def sample_configs(self):
        """示例掺杂配置"""
        return [
            {
                "config_id": "LFP_Ti_5pct_1",
                "formula": "LiFe0.9Ti0.1PO4",
                "dopant_element": "Ti",
                "doping_site": "Fe",
                "concentration_pct": 5.0,
                "n_atoms": 28,
                "poscar_content": """LiFePO4
1.0
10.329 0 0
0 6.007 0
0 0 4.692
Li Fe P O
4 4 4 16
Direct
0 0 0
"""
            }
        ]

    def test_export_configs_with_poscar(self, sample_configs):
        """测试导出含 POSCAR 的配置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "configs.txt"

            result = DopingConfigExporter.export_configs(
                sample_configs,
                output_path,
                include_poscar=True
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert "LiFe0.9Ti0.1PO4" in content
            assert "Ti" in content

    def test_export_configs_metadata_only(self, sample_configs):
        """测试仅导出配置元数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "configs.csv"

            result = DopingConfigExporter.export_configs(
                sample_configs,
                output_path,
                include_poscar=False
            )

            assert output_path.exists()
