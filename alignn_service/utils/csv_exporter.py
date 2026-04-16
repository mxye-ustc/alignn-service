"""CSV 导出工具模块

提供预测结果的多种格式导出功能
"""

import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CSVExportError(Exception):
    """导出错误"""
    pass


class PredictionExporter:
    """预测结果导出器"""

    SUPPORTED_FORMATS = ["csv", "xlsx", "json", "tsv"]

    # 列名映射（英文 -> 中文）
    COLUMN_NAMES = {
        "filename": "文件名",
        "formula": "化学式",
        "n_atoms": "原子数",
        "task_id": "任务ID",
        "created_at": "创建时间",
        "model": "模型",
        "value": "预测值",
        "unit": "单位",
        "processing_time": "处理时间(秒)",
        "dopant_element": "掺杂元素",
        "doping_site": "掺杂位点",
        "concentration": "浓度(%)",
    }

    @classmethod
    def export(
        cls,
        predictions: List[Dict[str, Any]],
        output_path: Union[str, Path],
        format: str = "csv",
        include_metadata: bool = True
    ) -> str:
        """
        导出预测结果

        Args:
            predictions: 预测结果列表
            output_path: 输出路径
            format: 导出格式 ("csv", "xlsx", "json", "tsv")
            include_metadata: 是否包含元数据

        Returns:
            输出文件路径
        """
        output_path = Path(output_path)
        format = format.lower()

        if format not in cls.SUPPORTED_FORMATS:
            raise CSVExportError(
                f"不支持的格式: {format}。"
                f"支持的格式: {', '.join(cls.SUPPORTED_FORMATS)}"
            )

        # 展平预测结果
        flat_data = cls._flatten_predictions(predictions, include_metadata)

        if format == "csv":
            return cls._to_csv(flat_data, output_path)
        elif format == "tsv":
            return cls._to_tsv(flat_data, output_path)
        elif format == "xlsx":
            return cls._to_excel(flat_data, output_path)
        elif format == "json":
            return cls._to_json(predictions, output_path)

    @classmethod
    def export_to_string(
        cls,
        predictions: List[Dict[str, Any]],
        format: str = "csv",
        include_metadata: bool = True
    ) -> str:
        """
        导出为字符串（用于下载）

        Args:
            predictions: 预测结果列表
            format: 导出格式
            include_metadata: 是否包含元数据

        Returns:
            导出内容的字符串
        """
        flat_data = cls._flatten_predictions(predictions, include_metadata)

        if format == "csv":
            return cls._to_csv_string(flat_data)
        elif format == "tsv":
            return cls._to_tsv_string(flat_data)
        elif format == "json":
            return json.dumps(predictions, indent=2, ensure_ascii=False)
        else:
            raise CSVExportError(f"不支持的字符串格式: {format}")

    @staticmethod
    def _flatten_predictions(
        predictions: List[Dict[str, Any]],
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """将嵌套的预测结果展平为行"""
        rows = []

        for pred in predictions:
            # 基础信息
            row = {}

            if include_metadata:
                row["filename"] = pred.get("filename", "")
                row["task_id"] = pred.get("task_id", "")
                row["created_at"] = pred.get("created_at", "")

            row["formula"] = pred.get("formula", "")
            row["n_atoms"] = pred.get("n_atoms", "")

            # 掺杂信息（如果有）
            if "dopant_element" in pred:
                row["dopant_element"] = pred["dopant_element"]
            if "doping_site" in pred:
                row["doping_site"] = pred["doping_site"]
            if "concentration" in pred:
                row["concentration"] = pred["concentration"]

            # 处理时间
            if "processing_time_seconds" in pred:
                row["processing_time"] = pred["processing_time_seconds"]

            # 预测值
            preds = pred.get("predictions", {})
            if isinstance(preds, dict):
                for model_name, result in preds.items():
                    if isinstance(result, dict):
                        if "value" in result:
                            row[f"{model_name}_value"] = result["value"]
                        if "unit" in result:
                            row[f"{model_name}_unit"] = result["unit"]
                        if "processing_time" in result:
                            row[f"{model_name}_time"] = result["processing_time"]
                    elif isinstance(result, (int, float)):
                        row[f"{model_name}_value"] = result
                    elif isinstance(result, dict) and "error" in result:
                        row[f"{model_name}_error"] = result["error"]

            rows.append(row)

        return rows

    @staticmethod
    def _to_csv(flat_data: List[Dict], output_path: Path) -> str:
        """导出为 CSV 文件"""
        if not flat_data:
            raise CSVExportError("没有数据可导出")

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=flat_data[0].keys())
            writer.writeheader()
            writer.writerows(flat_data)

        logger.info(f"CSV 导出完成: {output_path}")
        return str(output_path)

    @staticmethod
    def _to_csv_string(flat_data: List[Dict]) -> str:
        """导出为 CSV 字符串"""
        if not flat_data:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=flat_data[0].keys())
        writer.writeheader()
        writer.writerows(flat_data)

        return output.getvalue()

    @staticmethod
    def _to_tsv(flat_data: List[Dict], output_path: Path) -> str:
        """导出为 TSV 文件"""
        if not flat_data:
            raise CSVExportError("没有数据可导出")

        with open(output_path, "w", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=flat_data[0].keys(),
                delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(flat_data)

        logger.info(f"TSV 导出完成: {output_path}")
        return str(output_path)

    @staticmethod
    def _to_tsv_string(flat_data: List[Dict]) -> str:
        """导出为 TSV 字符串"""
        if not flat_data:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=flat_data[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(flat_data)

        return output.getvalue()

    @staticmethod
    def _to_excel(flat_data: List[Dict], output_path: Path) -> str:
        """导出为 Excel 文件"""
        try:
            import pandas as pd
        except ImportError:
            raise CSVExportError("需要安装 pandas 和 openpyxl 来导出 Excel")

        if not flat_data:
            raise CSVExportError("没有数据可导出")

        df = pd.DataFrame(flat_data)
        df.to_excel(output_path, index=False, engine="openpyxl")

        logger.info(f"Excel 导出完成: {output_path}")
        return str(output_path)

    @staticmethod
    def _to_json(predictions: List[Dict], output_path: Path) -> str:
        """导出为 JSON 文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON 导出完成: {output_path}")
        return str(output_path)


class BatchResultExporter:
    """批量结果导出器"""

    @staticmethod
    def export_batch_summary(
        batch_result: Dict[str, Any],
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """
        导出批量预测汇总结果

        Args:
            batch_result: 批量预测结果
            output_dir: 输出目录

        Returns:
            各格式文件路径的字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = batch_result.get("task_id", "unknown")
        base_name = f"batch_{task_id}_{timestamp}"

        results = batch_result.get("results", [])

        output_files = {}

        # CSV
        csv_path = output_dir / f"{base_name}.csv"
        try:
            PredictionExporter.export(results, csv_path, format="csv")
            output_files["csv"] = str(csv_path)
        except Exception as e:
            logger.error(f"CSV 导出失败: {e}")

        # Excel
        xlsx_path = output_dir / f"{base_name}.xlsx"
        try:
            PredictionExporter.export(results, xlsx_path, format="xlsx")
            output_files["xlsx"] = str(xlsx_path)
        except Exception as e:
            logger.error(f"Excel 导出失败: {e}")

        # JSON
        json_path = output_dir / f"{base_name}.json"
        try:
            PredictionExporter.export(batch_result, json_path, format="json")
            output_files["json"] = str(json_path)
        except Exception as e:
            logger.error(f"JSON 导出失败: {e}")

        return output_files

    @staticmethod
    def export_errors(
        errors: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ) -> str:
        """导出错误报告"""
        output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "error"],
                extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(errors)

        return str(output_path)


class DopingConfigExporter:
    """掺杂配置导出器"""

    @staticmethod
    def export_configs(
        configs: List[Dict[str, Any]],
        output_path: Union[str, Path],
        include_poscar: bool = True
    ) -> str:
        """
        导出掺杂配置

        Args:
            configs: 掺杂配置列表
            output_path: 输出路径
            include_poscar: 是否包含 POSCAR 内容

        Returns:
            输出文件路径
        """
        output_path = Path(output_path)

        if include_poscar:
            return DopingConfigExporter._export_with_poscar(configs, output_path)
        else:
            return DopingConfigExporter._export_metadata_only(configs, output_path)

    @staticmethod
    def _export_with_poscar(
        configs: List[Dict[str, Any]],
        output_path: Path
    ) -> str:
        """导出配置和 POSCAR 内容"""
        lines = []

        for config in configs:
            lines.append(f"# {config.get('formula', 'Unknown')}")
            lines.append(f"# Config ID: {config.get('config_id', 'N/A')}")
            lines.append(f"# Dopant: {config.get('dopant_element', 'N/A')}")
            lines.append(f"# Site: {config.get('doping_site', 'N/A')}")
            lines.append(f"# Concentration: {config.get('concentration_pct', 0):.2f}%")
            lines.append("")

            if "poscar_content" in config:
                lines.append(config["poscar_content"])
            elif "poscar_path" in config:
                poscar_path = Path(config["poscar_path"])
                if poscar_path.exists():
                    lines.append(poscar_path.read_text())

            lines.append("")
            lines.append("=" * 80)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return str(output_path)

    @staticmethod
    def _export_metadata_only(
        configs: List[Dict[str, Any]],
        output_path: Path
    ) -> str:
        """仅导出配置元数据"""
        return PredictionExporter.export(configs, output_path, format="csv")


def export_predictions(
    predictions: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = "csv"
) -> str:
    """便捷函数：导出预测结果"""
    return PredictionExporter.export(predictions, output_path, format)


def export_batch_results(
    batch_result: Dict[str, Any],
    output_dir: Union[str, Path]
) -> Dict[str, str]:
    """便捷函数：导出批量结果"""
    return BatchResultExporter.export_batch_summary(batch_result, output_dir)