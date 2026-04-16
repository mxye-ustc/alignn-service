"""ALIGNN 预测服务 Web UI

基于 Streamlit 的交互式界面
支持多页面：单结构预测、批量预测、自定义掺杂、数据库对比、历史记录
"""

import io
import time
import requests
import streamlit as st
from pathlib import Path
import base64

# 页面配置
st.set_page_config(
    page_title="ALIGNN 预测服务",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 配置
API_BASE_URL = "http://localhost:8000"

# POSCAR 文件目录
POSCAR_DIR = "/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/poscar_files"


def get_poscar_content(filename: str) -> str:
    """获取 POSCAR 文件内容"""
    import os
    poscar_path = os.path.join(POSCAR_DIR, filename)
    if os.path.exists(poscar_path):
        with open(poscar_path, 'r') as f:
            return f.read()
    return None


def render_3dmol_html(poscar_content: str, dopant_elem: str = "") -> str:
    """生成 3Dmol.js HTML 代码"""
    # 对 POSCAR 内容进行 base64 编码以避免转义问题
    encoded_content = base64.b64encode(poscar_content.encode()).decode()

    highlight_style = ""
    if dopant_elem:
        highlight_style = f"""
        // 高亮掺杂元素
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

        // 解析 POSCAR 内容
        var poscarData = atob("{encoded_content}");
        viewer.addModel(poscarData, "vasp");

        // 默认球棍样式
        viewer.setStyle({{}}, {{stick: {{}}, sphere: {{radius: 0.4}}}});
        {highlight_style}
        viewer.zoomTo();
        viewer.render();
    }});
    </script>
    """
    return html


def render_3dmol_component(filename: str, dopant_elem: str = ""):
    """Streamlit 组件：3D 晶体可视化"""
    poscar_content = get_poscar_content(filename)
    if not poscar_content:
        st.error(f"找不到文件: {filename}")
        return

    html = render_3dmol_html(poscar_content, dopant_elem)
    st.components.v1.html(html, height=420, scrolling=False)


def init_session_state():
    """初始化会话状态"""
    if "task_id" not in st.session_state:
        st.session_state.task_id = None
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []


def get_available_models() -> dict:
    """获取可用模型列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {}


def get_task_status(task_id: str) -> dict:
    """获取任务状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/tasks/{task_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {"status": "error", "message": "无法连接到服务器"}


def get_task_result(task_id: str) -> dict:
    """获取任务结果"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/tasks/{task_id}/result", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def submit_prediction(file_content: bytes, filename: str, models: list) -> dict:
    """提交预测任务"""
    files = {"file": (filename, file_content)}
    data = {"models": ",".join(models)}

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict/async",
            files=files,
            data=data,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
    return {"error": "提交失败"}


def main():
    init_session_state()

    # 标题
    st.title("🔬 ALIGNN 晶体性质预测服务")
    st.markdown("**A**tomistic **L**ine **G**raph **N**eural **N**etwork - 晶体性质预测")

    # 侧边栏导航
    with st.sidebar:
        st.header("🧭 功能导航")

        import os
        st.page_link("ui/app.py", label="🏠 首页 - 单结构预测", icon="🏠")

        # 检查是否有其他页面
        pages_dir = Path(__file__).parent / "pages"
        if pages_dir.exists():
            for page_file in sorted(pages_dir.glob("*.py")):
                if page_file.stem not in ["__init__", "__pycache__"]:
                    page_name = page_file.stem.replace("_", " ").title()
                    icon_map = {
                        "batch": "📊",
                        "doping": "⚗️",
                        "compare": "📈",
                        "history": "📁",
                    }
                    icon = icon_map.get(page_file.stem, "📄")
                    st.page_link(f"ui/pages/{page_file.name}", label=f"{icon} {page_name}", icon=icon)

        st.divider()

        # 检查 API 连接
        st.header("服务状态")

    # 侧边栏
    with st.sidebar:
        st.header("服务状态")

        # 检查 API 连接
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
            st.success("✅ 服务正常")
            st.info(f"设备: {health.get('device', 'Unknown')}")
        except:
            st.error("❌ 无法连接到 API 服务")
            st.info("请确保 API 服务正在运行（端口 8000）")
            st.stop()

        st.divider()

        # 可用模型
        st.subheader("可用模型")
        models_info = get_available_models()

        if models_info.get("models"):
            for model_id, model_data in models_info["models"].items():
                with st.expander(f"📊 {model_data['name']}", expanded=False):
                    st.caption(f"**ID:** `{model_id}`")
                    st.write(f"描述: {model_data.get('description', 'N/A')}")
                    st.write(f"单位: {model_data.get('unit', 'N/A')}")
                    st.write(f"来源: {model_data.get('source', 'N/A')}")
        else:
            st.info("加载模型信息中...")

    # 主内容区
    tab1, tab2, tab3 = st.tabs(["📤 单结构预测", "📋 批量预测", "📊 历史结果"])

    # ========== Tab 1: 单结构预测 ==========
    with tab1:
        st.header("上传晶体结构文件进行预测")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "选择文件",
                type=["poscar", "vasp", "cif", "xyz", "pdb"],
                help="支持的格式: POSCAR, VASP, CIF, XYZ, PDB"
            )

            if uploaded_file:
                st.success(f"已上传: {uploaded_file.name}")
                st.info(f"文件大小: {uploaded_file.size / 1024:.1f} KB")

                # 预览文件内容
                with st.expander("📄 预览文件内容"):
                    content = uploaded_file.getvalue().decode("utf-8")
                    st.text(content[:2000] + "..." if len(content) > 2000 else content)

                # ── 3D 晶体可视化（在上传区下方） ──
                st.divider()
                st.subheader("🔬 3D 晶体结构预览")

                # 尝试从文件名提取掺杂元素
                import re
                dopant_elem = ""
                match = re.search(r'_([A-Z][a-z]?)_x\d+', uploaded_file.name)
                if match:
                    dopant_elem = match.group(1)
                    st.caption(f"🎯 检测到掺杂元素: {dopant_elem}")

                # 直接使用上传的文件内容渲染 3D 结构
                file_content = uploaded_file.getvalue().decode("utf-8")
                encoded_content = base64.b64encode(file_content.encode()).decode()

                highlight_js = f"""
                // 高亮掺杂元素
                viewer.setStyle({{elem: '{dopant_elem}'}}, {{sphere: {{colorscheme: 'Jmol', radius: 0.7}}}});
                """ if dopant_elem else ""

                html_3d = f"""
                <div id="viewer_upload" style="width: 100%; height: 350px; position: relative; border-radius: 8px; overflow: hidden; border: 1px solid rgba(100,150,255,0.2);"></div>
                <script src="https://3dmol.org/build/3Dmol-min.js"></script>
                <script>
                $(document).ready(function() {{
                    var viewer = $3Dmol.createViewer("viewer_upload", {{
                        backgroundColor: "rgba(5,5,20,1)"
                    }});
                    var poscarData = atob("{encoded_content}");
                    viewer.addModel(poscarData, "vasp");
                    viewer.setStyle({{}}, {{stick: {{}}, sphere: {{radius: 0.35}}}});
                    {highlight_js}
                    viewer.zoomTo();
                    viewer.render();
                }});
                </script>
                """
                st.components.v1.html(html_3d, height=370)
                st.caption("💡 拖拽旋转 | 滚轮缩放 | 右键平移")

        with col2:
            st.subheader("选择预测模型")
            models_info = get_available_models()

            if models_info.get("models"):
                selected_models = []
                for model_id, model_data in models_info["models"].items():
                    if st.checkbox(
                        model_data["name"],
                        value=True if model_id in ["jv_formation_energy_peratom_alignn",
                                                    "jv_optb88vdw_bandgap_alignn"] else False,
                        help=f"{model_data.get('description', '')}\n单位: {model_data.get('unit', '')}"
                    ):
                        selected_models.append(model_id)

                st.session_state.selected_models = selected_models

                if selected_models:
                    st.info(f"已选择 {len(selected_models)} 个模型")
                else:
                    st.warning("请至少选择一个模型")
            else:
                st.error("无法加载模型列表")

        # 提交预测
        if uploaded_file and st.session_state.selected_models and st.button("🚀 开始预测", type="primary"):
            with st.spinner("正在提交预测任务..."):
                result = submit_prediction(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    st.session_state.selected_models
                )

                if "task_id" in result:
                    st.session_state.task_id = result["task_id"]
                    st.success(f"任务已提交！Task ID: `{result['task_id']}`")
                else:
                    st.error(f"提交失败: {result.get('error', '未知错误')}")

        # 显示任务状态和结果
        if st.session_state.task_id:
            task_id = st.session_state.task_id
            st.divider()
            st.subheader(f"📈 任务状态: `{task_id[:8]}...`")

            # 轮询任务状态
            progress_bar = st.progress(0)
            status_text = st.empty()

            while True:
                status = get_task_status(task_id)

                if status.get("status") == "PENDING":
                    progress_bar.progress(0.1)
                    status_text.info("⏳ 任务等待中...")
                elif status.get("status") == "PROCESSING":
                    info = status.get("info", {})
                    progress = info.get("progress", 0.5)
                    message = info.get("message", "处理中...")
                    progress_bar.progress(progress)
                    status_text.info(f"🔄 {message}")
                elif status.get("status") == "SUCCESS":
                    progress_bar.progress(1.0)
                    status_text.success("✅ 预测完成！")
                    break
                elif status.get("status") == "FAILURE":
                    progress_bar.progress(1.0)
                    status_text.error(f"❌ 任务失败: {status.get('info', '未知错误')}")
                    break
                else:
                    status_text.warning(f"未知状态: {status}")
                    break

                time.sleep(2)

            # 获取并显示结果
            if status.get("status") == "SUCCESS":
                result = get_task_result(task_id)

                if result:
                    st.session_state.prediction_results = result
                    st.success("结果已获取！")
                else:
                    st.error("无法获取结果")

    # ========== Tab 2: 批量预测 ==========
    with tab2:
        st.header("批量上传多个结构文件")

        uploaded_files = st.file_uploader(
            "选择多个文件",
            type=["poscar", "vasp", "cif", "xyz", "pdb"],
            accept_multiple_files=True,
            help="最多支持 100 个文件"
        )

        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个文件")

            # 预览文件列表
            with st.expander("📋 文件列表"):
                for i, f in enumerate(uploaded_files[:20]):  # 只显示前20个
                    st.write(f"{i+1}. {f.name} ({f.size/1024:.1f} KB)")
                if len(uploaded_files) > 20:
                    st.info(f"... 还有 {len(uploaded_files) - 20} 个文件")

            # 提交批量任务
            if st.button("📤 提交批量预测", type="primary"):
                st.info("批量预测功能开发中，请使用单结构预测或 API 接口")

    # ========== Tab 3: 历史结果 ==========
    with tab3:
        st.header("预测结果")

        if st.session_state.prediction_results:
            result = st.session_state.prediction_results

            # 结构信息
            if "structure_info" in result:
                st.subheader("结构信息")
                info = result["structure_info"]
                col1, col2, col3 = st.columns(3)
                col1.metric("化学式", info.get("formula", "N/A"))
                col2.metric("原子数", info.get("n_atoms", "N/A"))
                col3.metric("元素种类", info.get("n_elements", "N/A"))

                if "elements" in info:
                    st.write("元素组成:")
                    for elem, frac in info["elements"].items():
                        st.write(f"  - {elem}: {frac:.2%}")

            st.divider()

            # 预测结果
            st.subheader("预测结果")

            if "predictions" in result:
                predictions = result["predictions"]

                # 显示每个模型的预测值
                for model_id, pred in predictions.items():
                    if "error" not in pred:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**{pred.get('model_name_display', model_id)}**")
                            st.caption(f"`{model_id}`")
                        with col2:
                            st.metric(
                                "预测值",
                                f"{pred.get('value', 0):.4f}",
                                delta=None
                            )
                        with col3:
                            st.write(f"单位: {pred.get('unit', '')}")
                            st.write(f"耗时: {pred.get('processing_time', 'N/A')}s")

                        st.divider()
                    else:
                        st.error(f"{model_id}: {pred['error']}")

            # ── 3D 晶体结构可视化 ──
            st.divider()
            st.subheader("🔬 3D 晶体结构可视化")

            # 尝试从文件名推断掺杂元素
            dopant_elem = ""
            filename_for_viewer = None

            # 如果有上传的文件名
            if "structure_info" in result and "filename" in result["structure_info"]:
                fname = result["structure_info"]["filename"]
                # 尝试从文件名提取元素
                import re
                match = re.search(r'_([A-Z][a-z]?)_x\d+', fname)
                if match:
                    dopant_elem = match.group(1)
                filename_for_viewer = fname

            # 或者查找对应的 POSCAR 文件
            if not filename_for_viewer:
                import os
                if os.path.exists(POSCAR_DIR):
                    # 尝试匹配化学式
                    if "structure_info" in result:
                        formula = result["structure_info"].get("formula", "")
                        for f in os.listdir(POSCAR_DIR):
                            if f.endswith(".poscar"):
                                # 简单匹配
                                if any(elem in f for elem in ["Ti", "Co", "V", "Mn", "Ni", "Cu", "Zn", "Cr", "Nb", "Mo", "Er", "Y", "Nd", "Na", "Mg", "Al", "W", "Si", "S"]):
                                    if formula and formula in f or not formula:
                                        filename_for_viewer = f
                                        break

            col1, col2 = st.columns([3, 1])
            with col1:
                if filename_for_viewer:
                    st.info(f"📁 检测到结构文件: {filename_for_viewer}")
                    if dopant_elem:
                        st.caption(f"🎯 掺杂元素: {dopant_elem}")
                else:
                    st.info("💡 上传 POSCAR/CIF 文件可查看 3D 结构")

            with col2:
                view_3d = st.button("🔬 查看 3D 结构", type="primary", disabled=not filename_for_viewer)

            if view_3d and filename_for_viewer:
                st.divider()
                st.markdown("**鼠标操作:** 拖拽旋转 | 滚轮缩放 | 右键平移")
                render_3dmol_component(filename_for_viewer, dopant_elem)

            # 原始 JSON
            with st.expander("📄 原始 JSON 数据"):
                st.json(result)
        else:
            st.info("暂无预测结果，请先进行预测")

    # 页脚
    st.divider()
    st.caption("ALIGNN Prediction Service v1.0 | 基于 ALIGNN 深度学习模型")


if __name__ == "__main__":
    main()
