"""批量预测页面

支持多文件上传、批量处理和进度追踪
"""

import streamlit as st
import time
from pathlib import Path

from alignn_service.ui.components import (
    CrystalComponents,
    PredictionComponents,
    ProgressComponents,
    DEFAULT_MODELS,
    submit_batch_prediction,
    check_service_health,
)


def main():
    st.set_page_config(
        page_title="批量预测 - ALIGNN",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 批量结构预测")
    st.markdown("一次上传多个结构文件，批量获得性质预测结果")

    # 检查服务状态
    if not check_service_health():
        st.error("⚠️ 无法连接到预测服务，请确保服务正在运行。")
        st.info("启动服务命令: `uvicorn alignn_service.main:app --reload`")
        return

    # 初始化会话状态
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None
    if "batch_task_id" not in st.session_state:
        st.session_state.batch_task_id = None

    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ 预测设置")

        selected_models = CrystalComponents.model_selector(
            "选择预测模型",
            default_models=["jv_formation_energy_peratom_alignn"],
            show_description=False
        )

        st.divider()

        # 高级设置
        with st.expander("高级设置"):
            cutoff = st.number_input(
                "截断半径 (Å)",
                min_value=1.0,
                max_value=20.0,
                value=8.0,
                step=0.5,
                help="图构建的截断半径"
            )

            max_neighbors = st.number_input(
                "最大近邻数",
                min_value=1,
                max_value=50,
                value=16,
                step=1,
                help="每个原子的最大近邻数"
            )

    # 主内容区
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 上传结构文件")

        uploaded_files = st.file_uploader(
            "拖拽或点击上传文件",
            type=["poscar", "vasp", "cif", "xyz", "pdb"],
            accept_multiple_files=True,
            help="最多支持 100 个文件"
        )

        if uploaded_files:
            st.success(f"✅ 已选择 {len(uploaded_files)} 个文件")

            # 文件列表
            with st.expander("📋 查看文件列表"):
                for i, f in enumerate(uploaded_files[:20]):
                    st.text(f"{i+1}. {f.name} ({f.size} bytes)")
                if len(uploaded_files) > 20:
                    st.info(f"... 还有 {len(uploaded_files) - 20} 个文件")

    with col2:
        st.subheader("📋 预测设置")

        if not selected_models:
            st.warning("请至少选择一个预测模型")
            return

        st.info(f"已选择 {len(selected_models)} 个模型")

        # 显示选择的模型
        for model_key in selected_models:
            model_info = DEFAULT_MODELS.get(model_key, {})
            st.text(f"• {model_info.get('name', model_key)}")

        st.divider()

        # 预测统计
        n_files = len(uploaded_files) if uploaded_files else 0
        n_models = len(selected_models)
        estimated_time = n_files * n_models * 15  # 假设每个模型每文件15秒

        st.metric("文件数量", n_files)
        st.metric("模型数量", n_models)
        st.metric("预计耗时", f"约 {estimated_time // 60} 分钟" if estimated_time > 60 else f"约 {estimated_time} 秒")

    st.divider()

    # 提交预测
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

    with col_btn1:
        submit_button = st.button(
            "🚀 开始批量预测",
            type="primary",
            disabled=not uploaded_files or not selected_models,
            use_container_width=True
        )

    with col_btn2:
        clear_button = st.button(
            "🗑️ 清空选择",
            use_container_width=True
        )

    with col_btn3:
        pass

    if clear_button:
        st.session_state.batch_results = None
        st.session_state.batch_task_id = None
        st.rerun()

    # 处理提交
    if submit_button and uploaded_files:
        with st.spinner("正在提交批量任务..."):
            files_data = [(f.name, f.getvalue()) for f in uploaded_files]

            task_id = submit_batch_prediction(
                files=files_data,
                models=selected_models
            )

            if task_id:
                st.session_state.batch_task_id = task_id
                st.success(f"✅ 任务已提交! Task ID: `{task_id}`")
            else:
                st.error("❌ 任务提交失败")

    # 显示进度和结果
    if st.session_state.batch_task_id:
        st.divider()
        st.subheader("📈 预测进度")

        result = ProgressComponents.batch_progress(
            task_id=st.session_state.batch_task_id,
            total=len(uploaded_files) if uploaded_files else 0
        )

        if result and result.get("status") == "completed":
            st.session_state.batch_results = result

    # 显示历史结果
    if st.session_state.batch_results:
        st.divider()
        st.subheader("📊 预测结果")

        result = st.session_state.batch_results

        # 统计信息
        col_stat1, col_stat2, col_stat3 = st.columns(3)

        with col_stat1:
            st.metric("总结构数", result.get("total_structures", 0))

        with col_stat2:
            st.metric("成功预测", result.get("successful", 0), delta_color="normal")

        with col_stat3:
            failed = result.get("failed", 0)
            if failed > 0:
                st.metric("失败", failed, delta_color="inverse")
            else:
                st.metric("失败", 0)

        # 结果表格
        if "results" in result and result["results"]:
            st.subheader("📋 结果详情")

            df = PredictionComponents.results_table(result["results"], selected_models)

            # 导出功能
            st.divider()
            st.subheader("💾 导出结果")

            PredictionComponents.export_buttons(
                result["results"],
                task_id=st.session_state.batch_task_id
            )

        # 错误详情
        if "errors" in result and result["errors"]:
            with st.expander("⚠️ 查看失败文件"):
                for err in result["errors"]:
                    st.error(f"**{err.get('name', 'Unknown')}**: {err.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
