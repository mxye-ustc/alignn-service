"""自定义掺杂页面

允许用户配置并生成自定义掺杂构型
"""

import io
import time
from pathlib import Path

import streamlit as st

from alignn_service.ui.components import (
    CrystalComponents,
    DopingComponents,
    PredictionComponents,
    ProgressComponents,
    DEFAULT_MODELS,
    check_service_health,
    submit_batch_prediction,
)


# 预设宿主结构
PRESET_HOSTS = {
    "LiFePO4 (LFP)": {
        "formula": "LiFePO4",
        "elements": {"Li": 4, "Fe": 4, "P": 4, "O": 16},
        "doping_sites": ["Fe", "Li", "P"],
    },
    "LiCoO2 (LCO)": {
        "formula": "LiCoO2",
        "elements": {"Li": 1, "Co": 1, "O": 2},
        "doping_sites": ["Li", "Co"],
    },
    "LiMn2O4 (LMO)": {
        "formula": "LiMn2O4",
        "elements": {"Li": 1, "Mn": 2, "O": 4},
        "doping_sites": ["Mn", "Li"],
    },
    "LiNiCoMnO2 (NCM)": {
        "formula": "LiNiCoMnO2",
        "elements": {"Li": 1, "Ni": 1/3, "Co": 1/3, "Mn": 1/3, "O": 2},
        "doping_sites": ["Ni", "Co", "Mn", "Li"],
    },
}

# 常用掺杂元素
COMMON_DOPANTS = ["Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn", "Al", "Mg", "Si", "Zr", "Nb", "Mo", "W", "Ce"]


def main():
    st.set_page_config(
        page_title="自定义掺杂 - ALIGNN",
        page_icon="⚗️",
        layout="wide"
    )

    st.title("⚗️ 自定义掺杂构型生成")
    st.markdown("配置掺杂参数，生成特定浓度和位点的掺杂构型")

    # 检查服务状态
    if not check_service_health():
        st.error("⚠️ 无法连接到预测服务，请确保服务正在运行。")
        return

    # 初始化会话状态
    if "generated_configs" not in st.session_state:
        st.session_state.generated_configs = []
    if "doping_task_id" not in st.session_state:
        st.session_state.doping_task_id = None

    # 侧边栏配置
    with st.sidebar:
        st.header("🔧 掺杂配置")

        # 宿主结构选择
        st.subheader("宿主结构")
        host_option = st.radio(
            "选择方式",
            ["使用预设", "上传文件"],
            captions=["使用内置宿主结构", "上传您的 POSCAR/CIF 文件"]
        )

        if host_option == "使用预设":
            selected_host = st.selectbox(
                "选择宿主材料",
                options=list(PRESET_HOSTS.keys())
            )
            host_info = PRESET_HOSTS[selected_host]
            st.info(f"化学式: {host_info['formula']}")
        else:
            uploaded_host = st.file_uploader(
                "上传宿主结构文件",
                type=["poscar", "vasp", "cif"],
                help="支持 POSCAR、VASP、CIF 格式"
            )
            if uploaded_host:
                st.success(f"已上传: {uploaded_host.name}")

        st.divider()

        # 掺杂配置
        st.subheader("掺杂设置")

        # 掺杂元素
        selected_dopants = DopingComponents.dopant_selector(
            "选择掺杂元素",
            common_dopants=COMMON_DOPANTS
        )

        if not selected_dopants:
            st.warning("请至少选择一个掺杂元素")
            return

        # 掺杂位点
        if host_option == "使用预设":
            available_sites = {site: host_info["elements"].get(site, 0) for site in host_info["doping_sites"]}
        else:
            available_sites = {"Fe": 4, "Li": 4, "P": 4}  # 默认

        selected_site = DopingComponents.site_selector(
            "选择掺杂位点",
            available_sites=available_sites
        )

        # 浓度配置
        min_conc, max_conc = DopingComponents.concentration_slider(
            "掺杂浓度范围 (%)",
            min_val=0.5,
            max_val=20.0
        )

        # 生成参数
        with st.expander("生成参数"):
            n_configs = st.number_input(
                "每个组合生成构型数",
                min_value=1,
                max_value=10,
                value=3,
                help="每个元素/浓度组合生成多个随机构型"
            )

            generate_button = st.button(
                "🎲 生成掺杂构型",
                type="primary",
                use_container_width=True
            )

    # 主内容区
    tab1, tab2, tab3 = st.tabs(["📊 生成预览", "🔬 结构预览", "🚀 批量预测"])

    with tab1:
        st.subheader("📊 掺杂构型预览")

        if selected_dopants and min_conc and max_conc:
            # 计算将要生成的组合
            concentrations = list(range(int(min_conc), int(max_conc) + 1))
            total_combinations = len(selected_dopants) * len(concentrations) * n_configs

            st.info(f"预计生成: {len(selected_dopants)} 种元素 × {len(concentrations)} 种浓度 × {n_configs} 个构型 = {total_combinations} 个构型")

            # 预览表格
            preview_data = []
            for dopant in selected_dopants:
                for conc in concentrations:
                    preview_data.append({
                        "掺杂元素": dopant,
                        "掺杂位点": selected_site,
                        "浓度 (%)": f"{conc}%",
                        "预计构型数": n_configs,
                    })

            import pandas as pd
            df_preview = pd.DataFrame(preview_data)
            st.dataframe(df_preview, use_container_width=True)

        # 显示已生成的构型
        if st.session_state.generated_configs:
            st.divider()
            st.subheader("✅ 已生成的构型")

            DopingComponents.doping_preview_table(st.session_state.generated_configs)

            # 统计
            unique_dopants = set(c.get("dopant_element") for c in st.session_state.generated_configs)
            unique_concs = set(c.get("concentration_pct") for c in st.session_state.generated_configs)

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("总构型数", len(st.session_state.generated_configs))
            with col_stat2:
                st.metric("掺杂元素", len(unique_dopants))
            with col_stat3:
                st.metric("浓度点数", len(unique_concs))

    with tab2:
        st.subheader("🔬 3D 结构预览")

        if st.session_state.generated_configs:
            # 选择要预览的构型
            config_options = [c.get("config_id", f"Config {i}")
                             for i, c in enumerate(st.session_state.generated_configs)]
            selected_config_idx = st.selectbox(
                "选择构型",
                options=range(len(config_options)),
                format_func=lambda i: config_options[i]
            )

            selected_config = st.session_state.generated_configs[selected_config_idx]

            # 显示构型信息
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("化学式", selected_config.get("formula", "N/A"))
            with col_info2:
                st.metric("掺杂元素", selected_config.get("dopant_element", "N/A"))
            with col_info3:
                st.metric("浓度", f"{selected_config.get('concentration_pct', 0):.2f}%")

            # 3D 可视化
            st.subheader("3D 晶体结构")
            dopant_elem = selected_config.get("dopant_element", "")

            # 从 POSCAR 内容显示（如果有）
            if "poscar_content" in selected_config:
                CrystalComponents.render_3d_viewer(
                    selected_config["poscar_content"],
                    dopant_elem=dopant_elem
                )
            elif "poscar_path" in selected_config:
                poscar_path = Path(selected_config["poscar_path"])
                if poscar_path.exists():
                    content = poscar_path.read_text()
                    CrystalComponents.render_3d_viewer(content, dopant_elem=dopant_elem)
                else:
                    st.warning("POSCAR 文件不存在")
            else:
                st.info("请先生成掺杂构型以查看 3D 结构")
        else:
            st.info("请先生成掺杂构型")

    with tab3:
        st.subheader("🚀 批量预测")

        if not st.session_state.generated_configs:
            st.info("请先生成掺杂构型")
            return

        # 选择预测模型
        selected_models = CrystalComponents.model_selector(
            "选择预测模型",
            default_models=["jv_formation_energy_peratom_alignn"],
            show_description=False
        )

        if not selected_models:
            st.warning("请至少选择一个预测模型")
            return

        st.divider()

        # 预测设置预览
        col_prev1, col_prev2, col_prev3 = st.columns(3)
        with col_prev1:
            st.metric("构型数量", len(st.session_state.generated_configs))
        with col_prev2:
            st.metric("预测模型", len(selected_models))
        with col_prev3:
            estimated_time = len(st.session_state.generated_configs) * len(selected_models) * 15
            st.metric("预计耗时", f"约 {estimated_time // 60} 分钟" if estimated_time > 60 else f"约 {estimated_time} 秒")

        st.divider()

        # 提交预测
        predict_col1, predict_col2 = st.columns([1, 1])

        with predict_col1:
            predict_button = st.button(
                "🔮 开始批量预测",
                type="primary",
                use_container_width=True
            )

        with predict_col2:
            clear_configs_button = st.button(
                "🗑️ 清空构型",
                use_container_width=True
            )

        if clear_configs_button:
            st.session_state.generated_configs = []
            st.session_state.doping_task_id = None
            st.rerun()

        if predict_button:
            # 准备预测数据
            with st.spinner("准备预测数据..."):
                files_data = []
                for config in st.session_state.generated_configs:
                    if "poscar_content" in config:
                        content = config["poscar_content"]
                    elif "poscar_path" in config:
                        poscar_path = Path(config["poscar_path"])
                        if poscar_path.exists():
                            content = poscar_path.read_text()
                        else:
                            continue
                    else:
                        continue

                    files_data.append((
                        f"{config.get('config_id', 'unknown')}.poscar",
                        content.encode()
                    ))

            if files_data:
                with st.spinner("提交批量预测任务..."):
                    task_id = submit_batch_prediction(
                        files=files_data,
                        models=selected_models
                    )

                    if task_id:
                        st.session_state.doping_task_id = task_id
                        st.success(f"✅ 任务已提交! Task ID: `{task_id}`")
                    else:
                        st.error("❌ 任务提交失败")
            else:
                st.error("没有可用的构型数据")

    # 显示预测进度
    if st.session_state.doping_task_id:
        st.divider()
        st.subheader("📈 预测进度")

        result = ProgressComponents.batch_progress(
            task_id=st.session_state.doping_task_id,
            total=len(st.session_state.generated_configs)
        )

        if result and result.get("status") == "completed":
            st.success("✅ 批量预测完成!")

            # 显示结果
            if "results" in result:
                st.subheader("📊 预测结果")

                # 合并构型信息和预测结果
                enriched_results = []
                for pred_result in result["results"]:
                    config_id = pred_result.get("name", "").replace(".poscar", "")
                    matching_config = next(
                        (c for c in st.session_state.generated_configs if c.get("config_id") == config_id),
                        {}
                    )

                    enriched = {**pred_result, **matching_config}
                    enriched_results.append(enriched)

                PredictionComponents.results_table(enriched_results, selected_models)

                # 导出按钮
                st.divider()
                PredictionComponents.export_buttons(
                    enriched_results,
                    task_id=st.session_state.doping_task_id
                )


if __name__ == "__main__":
    main()
