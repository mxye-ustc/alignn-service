"""数据库对比页面

对比不同数据库（JARVIS/MP）模型的预测结果
"""

import json
from typing import Dict, Any, List

import numpy as np
import requests
import streamlit as st

from alignn_service.ui.components import (
    CrystalComponents,
    ComparisonComponents,
    DEFAULT_MODELS,
    check_service_health,
)


# JARVIS 模型列表
JARVIS_MODELS = {
    "jv_formation_energy_peratom_alignn": {
        "name": "形成能 (JARVIS)",
        "property": "formation_energy",
        "category": "热力学"
    },
    "jv_optb88vdw_bandgap_alignn": {
        "name": "带隙 (JARVIS-vdw)",
        "property": "bandgap",
        "category": "电子结构"
    },
    "jv_mbj_bandgap_alignn": {
        "name": "带隙 (JARVIS-MBJ)",
        "property": "bandgap",
        "category": "电子结构"
    },
    "jv_bulk_modulus_kv_alignn": {
        "name": "体模量 (JARVIS)",
        "property": "bulk_modulus",
        "category": "力学性质"
    },
    "jv_shear_modulus_gv_alignn": {
        "name": "剪切模量 (JARVIS)",
        "property": "shear_modulus",
        "category": "力学性质"
    },
}

# Materials Project 模型列表
MP_MODELS = {
    "mp_e_form_alignn": {
        "name": "形成能 (MP)",
        "property": "formation_energy",
        "category": "热力学"
    },
    "mp_gappbe_alignn": {
        "name": "带隙 (MP-GGA)",
        "property": "bandgap",
        "category": "电子结构"
    },
}


def analyze_correlation(results: Dict[str, Any]) -> Dict[str, Any]:
    """分析不同模型预测结果的相关性"""
    predictions = results.get("predictions", {})

    # 按性质分组
    properties = {}

    for model_key, model_info in list(JARVIS_MODELS.items()) | list(MP_MODELS.items()):
        if model_key in predictions:
            result = predictions[model_key]
            if isinstance(result, dict) and "value" in result:
                prop = model_info["property"]
                if prop not in properties:
                    properties[prop] = {}
                properties[prop][model_key] = result["value"]

    # 计算相关性
    correlations = {}
    for prop, values in properties.items():
        if len(values) >= 2:
            val_list = list(values.values())
            mean_val = np.mean(val_list)
            std_val = np.std(val_list)

            # 计算一致性指标
            if std_val == 0:
                consistency = "高度一致"
                color = "green"
            elif std_val < abs(mean_val) * 0.1:
                consistency = "较好一致"
                color = "yellow"
            else:
                consistency = "差异较大"
                color = "red"

            correlations[prop] = {
                "values": values,
                "mean": mean_val,
                "std": std_val,
                "consistency": consistency,
                "color": color,
                "n_models": len(values)
            }

    return correlations


def calculate_reliability(correlations: Dict[str, Any]) -> Dict[str, str]:
    """计算各性质的可靠性评估"""
    recommendations = {}

    for prop, data in correlations.items():
        std_pct = abs(data["std"] / data["mean"] * 100) if data["mean"] != 0 else 0

        if std_pct < 5:
            reliability = "🟢 高"
            note = "两数据库预测值高度一致"
        elif std_pct < 15:
            reliability = "🟡 中"
            note = "预测值有一定差异，建议参考多个数据源"
        else:
            reliability = "🔴 低"
            note = "预测值差异较大，建议参考实验数据"

        recommendations[prop] = {
            "reliability": reliability,
            "note": note,
            "std_pct": std_pct
        }

    return recommendations


def render_comparison_chart(results: Dict[str, Any]):
    """渲染对比图表"""
    predictions = results.get("predictions", {})

    if not predictions:
        return

    # 准备数据
    labels = []
    values = []
    colors = []

    for model_key, result in predictions.items():
        if isinstance(result, dict) and "value" in result:
            model_info = JARVIS_MODELS.get(model_key) or MP_MODELS.get(model_key)
            if model_info:
                labels.append(model_info["name"])
                values.append(result["value"])
                colors.append("blue" if model_key.startswith("jv_") else "orange")

    if labels:
        import pandas as pd

        df = pd.DataFrame({
            "模型": labels,
            "预测值": values,
        })

        st.bar_chart(
            df.set_index("模型"),
            horizontal=False
        )


def main():
    st.set_page_config(
        page_title="数据库对比 - ALIGNN",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 多数据库预测对比")
    st.markdown("同时使用 JARVIS 和 Materials Project 模型进行预测，对比分析结果可靠性")

    # 检查服务状态
    if not check_service_health():
        st.error("⚠️ 无法连接到预测服务，请确保服务正在运行。")
        return

    # 初始化会话状态
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None

    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ 对比设置")

        # 选择对比的模型
        st.subheader("JARVIS 模型")
        jarvis_selected = []
        for model_key, model_info in JARVIS_MODELS.items():
            if st.checkbox(model_info["name"], value=True):
                jarvis_selected.append(model_key)

        st.subheader("Materials Project 模型")
        mp_selected = []
        for model_key, model_info in MP_MODELS.items():
            if st.checkbox(model_info["name"], value=True):
                mp_selected.append(model_key)

        selected_models = jarvis_selected + mp_selected

        if not selected_models:
            st.warning("请至少选择一个模型")
            return

        st.divider()

        # 参考数据输入
        st.subheader("📊 参考数据（可选）")
        st.caption("输入实验值或文献值进行对比")

        with st.expander("添加参考数据"):
            ref_formation_energy = st.number_input(
                "实验形成能 (eV/atom)",
                value=None,
                format="%.4f",
                help="文献或实验测得的形成能"
            )

            ref_bandgap = st.number_input(
                "实验带隙 (eV)",
                value=None,
                format="%.4f",
                help="文献或实验测得的带隙"
            )

            ref_bulk_modulus = st.number_input(
                "实验体模量 (GPa)",
                value=None,
                format="%.2f",
                help="文献或实验测得的体模量"
            )

    # 主内容区
    st.subheader("📤 上传结构文件")

    col_upload1, col_upload2 = st.columns([1, 1])

    with col_upload1:
        uploaded_file = st.file_uploader(
            "拖拽或点击上传晶体结构文件",
            type=["poscar", "vasp", "cif", "xyz", "pdb"]
        )

    with col_upload2:
        if uploaded_file:
            st.success(f"✅ 已上传: {uploaded_file.name}")

            # 显示文件信息
            file_content = uploaded_file.getvalue().decode("utf-8")
            st.text(f"文件大小: {len(file_content)} bytes")

    # 3D 预览
    if uploaded_file:
        st.divider()
        st.subheader("🔮 3D 晶体结构预览")

        file_content = uploaded_file.getvalue().decode("utf-8")
        CrystalComponents.render_3d_viewer(file_content, height=350)

    # 开始预测
    st.divider()

    predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])

    with predict_col1:
        predict_button = st.button(
            "🔍 开始对比预测",
            type="primary",
            disabled=not uploaded_file,
            use_container_width=True
        )

    with predict_col2:
        if st.button("🗑️ 清空结果", use_container_width=True):
            st.session_state.comparison_results = None
            st.rerun()

    with predict_col3:
        pass

    # 处理预测
    if predict_button and uploaded_file:
        with st.spinner("正在进行多模型预测..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"models": ",".join(selected_models)}

                response = requests.post(
                    "http://localhost:8000/api/v1/predict/sync",
                    files=files,
                    data=data,
                    timeout=300
                )

                if response.status_code == 200:
                    results = response.json()
                    st.session_state.comparison_results = results
                    st.success("✅ 预测完成!")
                else:
                    st.error(f"预测失败: {response.status_code}")

            except Exception as e:
                st.error(f"预测出错: {e}")

    # 显示对比结果
    if st.session_state.comparison_results:
        results = st.session_state.comparison_results

        st.divider()
        st.subheader("📊 对比预测结果")

        # 结构信息
        if "structure_info" in results:
            CrystalComponents.structure_info_card(results["structure_info"])

        # 预测结果表格
        ComparisonComponents.model_comparison_table(results.get("predictions", {}))

        # 可视化对比
        st.divider()
        st.subheader("📈 预测值对比")

        render_comparison_chart(results)

        # 相关性分析
        st.divider()
        st.subheader("🔍 相关性分析")

        correlations = analyze_correlation(results)

        if correlations:
            for prop, data in correlations.items():
                with st.expander(f"📌 {prop.upper()} 相关性详情"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("均值", f"{data['mean']:.4f}")

                    with col2:
                        st.metric("标准差", f"{data['std']:.4f}")

                    with col3:
                        consistency = data["consistency"]
                        if consistency == "高度一致":
                            st.success(consistency)
                        elif consistency == "较好一致":
                            st.warning(consistency)
                        else:
                            st.error(consistency)

                    # 显示各模型值
                    st.text("各模型预测值:")
                    for model, value in data["values"].items():
                        model_info = JARVIS_MODELS.get(model) or MP_MODELS.get(model)
                        model_name = model_info["name"] if model_info else model
                        st.text(f"  • {model_name}: {value:.4f}")

        # 可靠性评估
        st.divider()
        st.subheader("✅ 可靠性评估")

        recommendations = calculate_reliability(correlations)

        for prop, rec in recommendations.items():
            col_r1, col_r2 = st.columns([1, 3])

            with col_r1:
                st.metric(f"{prop.upper()} 可靠性", rec["reliability"])

            with col_r2:
                st.caption(rec["note"])

        # 参考数据对比
        st.divider()
        st.subheader("📚 与参考数据对比")

        predictions = results.get("predictions", {})

        # 形成能对比
        if ref_formation_energy is not None:
            st.subheader("形成能对比")

            jv_fe = predictions.get("jv_formation_energy_peratom_alignn", {}).get("value")
            mp_fe = predictions.get("mp_e_form_alignn", {}).get("value")

            if jv_fe is not None or mp_fe is not None:
                col_fe1, col_fe2, col_fe3 = st.columns(3)

                with col_fe1:
                    st.metric("实验值", f"{ref_formation_energy:.4f} eV/atom")

                with col_fe2:
                    if jv_fe is not None:
                        diff = abs(jv_fe - ref_formation_energy)
                        st.metric("JARVIS 预测", f"{jv_fe:.4f} eV/atom", delta=f"{diff:.4f}")

                with col_fe3:
                    if mp_fe is not None:
                        diff = abs(mp_fe - ref_formation_energy)
                        st.metric("MP 预测", f"{mp_fe:.4f} eV/atom", delta=f"{diff:.4f}")

        # 带隙对比
        if ref_bandgap is not None:
            st.subheader("带隙对比")

            jv_bg = predictions.get("jv_optb88vdw_bandgap_alignn", {}).get("value")
            mp_bg = predictions.get("mp_gappbe_alignn", {}).get("value")

            if jv_bg is not None or mp_bg is not None:
                col_bg1, col_bg2, col_bg3 = st.columns(3)

                with col_bg1:
                    st.metric("实验值", f"{ref_bandgap:.4f} eV")

                with col_bg2:
                    if jv_bg is not None:
                        diff = abs(jv_bg - ref_bandgap)
                        st.metric("JARVIS 预测", f"{jv_bg:.4f} eV", delta=f"{diff:.4f}")

                with col_bg3:
                    if mp_bg is not None:
                        diff = abs(mp_bg - ref_bandgap)
                        st.metric("MP 预测", f"{mp_bg:.4f} eV", delta=f"{diff:.4f}")

        # 总结建议
        st.divider()
        st.subheader("💡 综合建议")

        if recommendations:
            for prop, rec in recommendations.items():
                reliability = rec["reliability"]

                if "高" in reliability:
                    st.success(f"**{prop.upper()}**: {rec['note']}")
                elif "中" in reliability:
                    st.warning(f"**{prop.upper()}**: {rec['note']}")
                else:
                    st.error(f"**{prop.upper()}**: {rec['note']}")


if __name__ == "__main__":
    main()
