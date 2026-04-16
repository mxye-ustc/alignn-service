"""Streamlit 共享组件库

提供可复用的 UI 组件
"""

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# ==================== 常量定义 ====================

API_BASE_URL = "http://localhost:8000"

# 可用模型列表
DEFAULT_MODELS = {
    "jv_formation_energy_peratom_alignn": {
        "name": "形成能 (JARVIS)",
        "unit": "eV/atom",
        "description": "JARVIS-DFT 计算的形成能"
    },
    "jv_optb88vdw_bandgap_alignn": {
        "name": "带隙 (JARVIS-vdw)",
        "unit": "eV",
        "description": "JARVIS vdW-DFT 计算的带隙"
    },
    "jv_mbj_bandgap_alignn": {
        "name": "带隙 (JARVIS-MBJ)",
        "unit": "eV",
        "description": "JARVIS MBJ 计算的带隙"
    },
    "mp_e_form_alignn": {
        "name": "形成能 (MP)",
        "unit": "eV/atom",
        "description": "Materials Project 计算的形成能"
    },
    "mp_gappbe_alignn": {
        "name": "带隙 (MP-GGA)",
        "unit": "eV",
        "description": "Materials Project GGA-PBE 计算的带隙"
    },
    "jv_bulk_modulus_kv_alignn": {
        "name": "体模量 (JARVIS)",
        "unit": "GPa",
        "description": "JARVIS 计算的体模量"
    },
    "jv_shear_modulus_gv_alignn": {
        "name": "剪切模量 (JARVIS)",
        "unit": "GPa",
        "description": "JARVIS 计算的剪切模量"
    },
}


# ==================== 组件类 ====================

class CrystalComponents:
    """晶体相关组件"""

    @staticmethod
    def file_uploader(
        label: str = "上传晶体结构文件",
        accepted_formats: List[str] = ["poscar", "vasp", "cif", "xyz", "pdb"],
        help_text: str = "支持 POSCAR, VASP, CIF, XYZ, PDB 格式"
    ) -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
        """
        文件上传组件

        Returns:
            上传的文件对象，None 表示未上传
        """
        formats_str = ",".join([f".{f}" for f in accepted_formats])

        uploaded_file = st.file_uploader(
            label,
            type=accepted_formats,
            help=help_text,
            accept_multiple_files=False
        )

        return uploaded_file

    @staticmethod
    def multi_file_uploader(
        label: str = "上传多个结构文件",
        max_files: int = 100,
        help_text: str = "最多支持 100 个文件"
    ) -> List[st.runtime.uploaded_file_manager.UploadedFile]:
        """
        多文件上传组件

        Returns:
            上传的文件列表
        """
        uploaded_files = st.file_uploader(
            label,
            type=["poscar", "vasp", "cif", "xyz", "pdb"],
            accept_multiple_files=True,
            help=f"{help_text} (最多 {max_files} 个)"
        )

        if len(uploaded_files) > max_files:
            st.warning(f"文件数量超过限制，只处理前 {max_files} 个文件")
            return uploaded_files[:max_files]

        return uploaded_files

    @staticmethod
    def model_selector(
        label: str = "选择预测模型",
        default_models: List[str] = None,
        show_description: bool = True
    ) -> List[str]:
        """
        模型选择器

        Args:
            label: 标签
            default_models: 默认选中的模型
            show_description: 是否显示模型描述

        Returns:
            选中的模型名称列表
        """
        if default_models is None:
            default_models = ["jv_formation_energy_peratom_alignn"]

        selected = []

        if show_description:
            cols = st.columns(len(DEFAULT_MODELS))
            for i, (model_key, model_info) in enumerate(DEFAULT_MODELS.items()):
                with cols[i]:
                    if st.checkbox(
                        model_info["name"],
                        value=model_key in default_models,
                        help=f"{model_info['description']} (单位: {model_info['unit']})"
                    ):
                        selected.append(model_key)
        else:
            selected = st.multiselect(
                label,
                options=list(DEFAULT_MODELS.keys()),
                default=default_models
            )

        return selected

    @staticmethod
    def structure_preview_html(poscar_content: str, dopant_elem: str = "") -> str:
        """
        生成 3D 结构预览 HTML

        Args:
            poscar_content: POSCAR 文件内容
            dopant_elem: 要高亮的掺杂元素

        Returns:
            HTML 字符串
        """
        encoded_content = base64.b64encode(poscar_content.encode()).decode()

        highlight_style = ""
        if dopant_elem:
            highlight_style = f"""
            viewer.setStyle({{elem: '{dopant_elem}'}}, {{sphere: {{colorscheme: 'Jmol', radius: 0.7}}}});
            """

        html = f"""
        <div id="viewer_3d" style="width: 100%; height: 400px; position: relative; border-radius: 8px; overflow: hidden;"></div>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <script>
        $(document).ready(function() {{
            var viewer = $3Dmol.createViewer("viewer_3d", {{
                backgroundColor: "rgba(5,5,20,1)"
            }});

            var poscarData = atob("{encoded_content}");
            viewer.addModel(poscarData, "vasp");

            viewer.setStyle({{}}, {{stick: {{}}, sphere: {{radius: 0.4}}}});
            {highlight_style}
            viewer.zoomTo();
            viewer.render();
        }});
        </script>
        """
        return html

    @staticmethod
    def render_3d_viewer(poscar_content: str, dopant_elem: str = "", height: int = 400):
        """渲染 3D 晶体结构"""
        html = CrystalComponents.structure_preview_html(poscar_content, dopant_elem)
        st.components.v1.html(html, height=height, scrolling=False)

    @staticmethod
    def structure_info_card(structure_info: Dict[str, Any]):
        """显示结构信息卡片"""
        if not structure_info:
            return

        cols = st.columns(4)

        with cols[0]:
            st.metric("化学式", structure_info.get("formula", "N/A"))

        with cols[1]:
            st.metric("原子数", structure_info.get("n_atoms", "N/A"))

        with cols[2]:
            n_elements = structure_info.get("n_elements", "N/A")
            st.metric("元素种类", n_elements)

        with cols[3]:
            lattice = structure_info.get("lattice", {})
            a = lattice.get("a", "N/A")
            st.metric("晶格常数 a", f"{a:.3f}" if isinstance(a, (int, float)) else a)


class PredictionComponents:
    """预测结果相关组件"""

    @staticmethod
    def prediction_card(
        model_name: str,
        value: float,
        unit: str,
        processing_time: float = None,
        model_display_name: str = None
    ):
        """显示单个预测结果卡片"""
        if model_display_name is None:
            model_display_name = DEFAULT_MODELS.get(model_name, {}).get("name", model_name)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(model_display_name)
            st.metric("预测值", f"{value:.4f}", unit)

        with col2:
            st.markdown("&nbsp;")
            if processing_time:
                st.caption(f"⏱ {processing_time:.2f}s")

    @staticmethod
    def prediction_results(predictions: Dict[str, Any]):
        """显示所有预测结果"""
        if not predictions:
            st.info("暂无预测结果")
            return

        # 结构信息
        if "structure_info" in predictions:
            st.subheader("📊 结构信息")
            CrystalComponents.structure_info_card(predictions["structure_info"])
            st.divider()

        # 预测结果
        st.subheader("🔬 预测结果")

        if "predictions" in predictions:
            for model_name, result in predictions["predictions"].items():
                if isinstance(result, dict):
                    if "error" in result:
                        st.error(f"**{model_name}**: {result['error']}")
                    else:
                        CrystalComponents.prediction_card(
                            model_name=model_name,
                            value=result.get("value", 0),
                            unit=result.get("unit", ""),
                            processing_time=result.get("processing_time")
                        )
                elif isinstance(result, (int, float)):
                    unit = DEFAULT_MODELS.get(model_name, {}).get("unit", "")
                    CrystalComponents.prediction_card(
                        model_name=model_name,
                        value=result,
                        unit=unit
                    )

    @staticmethod
    def results_table(results: List[Dict[str, Any]], models: List[str] = None):
        """显示结果表格"""
        if not results:
            st.info("暂无结果")
            return

        # 构建表格数据
        rows = []
        for result in results:
            row = {
                "文件名": result.get("filename", ""),
                "化学式": result.get("formula", ""),
                "原子数": result.get("n_atoms", ""),
            }

            # 添加预测值
            preds = result.get("predictions", {})
            if isinstance(preds, dict):
                for model_name, value in preds.items():
                    if isinstance(value, dict):
                        row[model_name] = value.get("value", "")
                    elif isinstance(value, (int, float)):
                        row[model_name] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        return df

    @staticmethod
    def export_buttons(predictions: List[Dict[str, Any]], task_id: str = None):
        """显示导出按钮"""
        if not predictions:
            return

        cols = st.columns(3)

        # CSV
        with cols[0]:
            csv_data = pd.DataFrame(predictions).to_csv(index=False)
            st.download_button(
                "📥 下载 CSV",
                csv_data,
                file_name=f"predictions_{task_id or 'export'}.csv",
                mime="text/csv"
            )

        # JSON
        with cols[1]:
            import json
            json_data = json.dumps(predictions, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 下载 JSON",
                json_data,
                file_name=f"predictions_{task_id or 'export'}.json",
                mime="application/json"
            )

        # Excel
        with cols[2]:
            try:
                to_excel = pd.DataFrame(predictions).to_excel
                buffer = io.BytesIO()
                pd.DataFrame(predictions).to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)
                st.download_button(
                    "📥 下载 Excel",
                    buffer,
                    file_name=f"predictions_{task_id or 'export'}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception:
                pass


class ProgressComponents:
    """进度显示组件"""

    @staticmethod
    def task_progress(
        task_id: str,
        api_url: str = API_BASE_URL,
        poll_interval: float = 1.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        显示任务进度

        Args:
            task_id: 任务 ID
            api_url: API 基础 URL
            poll_interval: 轮询间隔（秒）

        Returns:
            (最终状态, 任务结果)
        """
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            try:
                response = requests.get(
                    f"{api_url}/api/v1/tasks/{task_id}",
                    timeout=10
                )

                if response.status_code != 200:
                    status_placeholder.error(f"请求失败: {response.status_code}")
                    break

                data = response.json()
                status = data.get("status", "UNKNOWN")

                # 更新进度
                if status == "PROCESSING":
                    info = data.get("info", {})
                    progress = info.get("progress", 0)
                    message = info.get("message", "处理中...")

                    progress_bar.progress(progress)
                    status_text.text(message)
                else:
                    progress_bar.progress(1.0 if status == "SUCCESS" else 0)

                # 检查是否完成
                if data.get("ready", False):
                    if data.get("successful", False):
                        status_placeholder.success("✅ 任务完成!")
                    else:
                        status_placeholder.error("❌ 任务失败")
                    break

                time.sleep(poll_interval)

            except requests.RequestException as e:
                status_placeholder.error(f"连接错误: {e}")
                break

        # 获取结果
        try:
            result_response = requests.get(
                f"{api_url}/api/v1/tasks/{task_id}/result",
                timeout=10
            )
            if result_response.status_code == 200:
                return "SUCCESS", result_response.json()
            else:
                return status, None
        except:
            return status, None

    @staticmethod
    def batch_progress(
        task_id: str,
        total: int,
        api_url: str = API_BASE_URL,
        poll_interval: float = 2.0
    ) -> Dict[str, Any]:
        """
        显示批量任务进度

        Returns:
            批量结果
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()

        all_results = []
        start_time = time.time()

        while True:
            try:
                response = requests.get(
                    f"{api_url}/api/v1/tasks/{task_id}",
                    timeout=10
                )

                if response.status_code != 200:
                    break

                data = response.json()
                status = data.get("status", "UNKNOWN")

                if status == "PROCESSING":
                    info = data.get("info", {})
                    progress = info.get("progress", 0)
                    message = info.get("message", "处理中...")

                    progress_bar.progress(progress)
                    status_text.text(message)
                else:
                    progress_bar.progress(1.0)
                    break

                time.sleep(poll_interval)

            except requests.RequestException:
                break

        # 获取最终结果
        try:
            result_response = requests.get(
                f"{api_url}/api/v1/tasks/{task_id}/result",
                timeout=30
            )
            if result_response.status_code == 200:
                result = result_response.json()

                elapsed = time.time() - start_time

                with results_placeholder.container():
                    st.success(f"✅ 批量预测完成! 耗时: {elapsed:.1f}秒")

                    if "results" in result:
                        st.subheader("📋 结果预览")
                        PredictionComponents.results_table(result["results"])

                return result
        except Exception as e:
            st.error(f"获取结果失败: {e}")

        return {"status": "failed", "error": "无法获取结果"}


class ComparisonComponents:
    """对比分析组件"""

    @staticmethod
    def model_comparison_table(predictions: Dict[str, Dict[str, Any]]):
        """显示多模型对比表格"""
        rows = []

        for model_name, result in predictions.items():
            if isinstance(result, dict) and "value" in result:
                model_info = DEFAULT_MODELS.get(model_name, {})
                rows.append({
                    "模型": model_info.get("name", model_name),
                    "预测值": f"{result['value']:.4f}",
                    "单位": result.get("unit", ""),
                    "数据源": model_info.get("description", "")
                })

        if rows:
            df = pd.DataFrame(rows)
            st.table(df)
            return df

        return None

    @staticmethod
    def correlation_indicator(values: List[float]) -> str:
        """相关性指示器"""
        if len(values) < 2:
            return "数据不足"

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)

        if variance < 0.01:
            return "🟢 高度一致"
        elif variance < 0.1:
            return "🟡 中等一致"
        else:
            return "🔴 差异较大"


class DopingComponents:
    """掺杂相关组件"""

    @staticmethod
    def dopant_selector(
        label: str = "选择掺杂元素",
        common_dopants: List[str] = None
    ) -> List[str]:
        """掺杂元素选择器"""
        if common_dopants is None:
            common_dopants = ["Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn", "Al", "Mg", "Si", "Zr", "Nb", "Mo"]

        selected = st.multiselect(
            label,
            options=common_dopants,
            default=["Ti", "V", "Mn"]
        )

        return selected

    @staticmethod
    def site_selector(
        label: str = "选择掺杂位点",
        available_sites: Dict[str, int] = None
    ) -> str:
        """掺杂位点选择器"""
        if available_sites is None:
            available_sites = {"Li": 4, "Fe": 4, "P": 4, "O": 16}

        options = list(available_sites.keys())
        labels = [f"{site} ({count}个原子)" for site, count in available_sites.items()]

        selected_idx = st.selectbox(
            label,
            options=range(len(options)),
            format_func=lambda i: labels[i]
        )

        return options[selected_idx]

    @staticmethod
    def concentration_slider(
        label: str = "掺杂浓度范围",
        min_val: float = 0.5,
        max_val: float = 20.0
    ) -> Tuple[float, float]:
        """浓度滑块"""
        min_col, max_col, step_col = st.columns(3)

        with min_col:
            min_val_input = st.number_input(
                "最小浓度 (%)",
                min_value=0.0,
                max_value=100.0,
                value=min_val,
                step=0.5
            )

        with max_col:
            max_val_input = st.number_input(
                "最大浓度 (%)",
                min_value=0.0,
                max_value=100.0,
                value=max_val,
                step=0.5
            )

        with step_col:
            step_input = st.number_input(
                "步长 (%)",
                min_value=0.1,
                max_value=50.0,
                value=1.0,
                step=0.5
            )

        return min_val_input, max_val_input

    @staticmethod
    def doping_preview_table(configs: List[Dict[str, Any]]):
        """显示掺杂配置预览表格"""
        if not configs:
            st.info("暂无生成的构型")
            return

        rows = []
        for config in configs:
            rows.append({
                "配置ID": config.get("config_id", ""),
                "化学式": config.get("formula", ""),
                "掺杂元素": config.get("dopant_element", ""),
                "掺杂位点": config.get("doping_site", ""),
                "浓度 (%)": f"{config.get('concentration_pct', 0):.2f}",
                "原子数": config.get("n_atoms", ""),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        return df


# ==================== 便捷函数 ====================

def get_models() -> Dict[str, Any]:
    """获取可用模型列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"models": DEFAULT_MODELS, "total": len(DEFAULT_MODELS)}


def check_service_health() -> bool:
    """检查服务健康状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def submit_prediction(
    file_content: bytes,
    filename: str,
    models: List[str],
    api_url: str = API_BASE_URL
) -> Optional[str]:
    """
    提交预测任务

    Returns:
        任务 ID，失败返回 None
    """
    try:
        files = {"file": (filename, file_content)}
        data = {"models": ",".join(models)}

        response = requests.post(
            f"{api_url}/api/v1/predict/async",
            files=files,
            data=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("task_id")

    except requests.RequestException as e:
        st.error(f"提交任务失败: {e}")

    return None


def submit_batch_prediction(
    files: List[Tuple[str, bytes]],
    models: List[str],
    api_url: str = API_BASE_URL
) -> Optional[str]:
    """
    提交批量预测任务

    Returns:
        任务 ID，失败返回 None
    """
    try:
        file_tuples = [("files", (name, content)) for name, content in files]
        data = {"models": ",".join(models)}

        response = requests.post(
            f"{api_url}/api/v1/predict/batch",
            files=file_tuples,
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("task_id")

    except requests.RequestException as e:
        st.error(f"提交批量任务失败: {e}")

    return None
