"""历史记录页面

查看和管理历史预测记录
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

from alignn_service.ui.components import DEFAULT_MODELS


# 结果存储目录
RESULTS_DIR = Path("alignn_service/results")
UPLOAD_DIR = Path("alignn_service/data/uploads")


def load_results() -> List[Dict[str, Any]]:
    """加载所有预测结果"""
    results = []

    if not RESULTS_DIR.exists():
        return results

    # 遍历所有结果文件
    for file_path in RESULTS_DIR.rglob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 添加文件路径
            data["_file_path"] = str(file_path)

            # 解析文件名获取类型
            filename = file_path.name
            if filename.startswith("batch_"):
                data["_type"] = "batch"
            else:
                data["_type"] = "single"

            results.append(data)

        except Exception as e:
            st.warning(f"加载文件失败 {file_path}: {e}")

    # 按时间排序
    results.sort(
        key=lambda x: x.get("completed_at", x.get("created_at", "")),
        reverse=True
    )

    return results


def load_batch_result(task_id: str) -> Optional[Dict[str, Any]]:
    """加载指定批量任务的结果"""
    if not RESULTS_DIR.exists():
        return None

    # 查找批量结果文件
    for file_path in RESULTS_DIR.rglob(f"batch_{task_id}.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass

    return None


def filter_results(
    results: List[Dict[str, Any]],
    search_query: str = "",
    type_filter: str = "all",
    date_range: tuple = None,
    model_filter: List[str] = None
) -> List[Dict[str, Any]]:
    """过滤结果"""
    filtered = results

    # 搜索过滤
    if search_query:
        query = search_query.lower()
        filtered = [
            r for r in filtered
            if query in str(r.get("formula", "")).lower()
            or query in str(r.get("task_id", "")).lower()
            or query in str(r.get("filename", "")).lower()
        ]

    # 类型过滤
    if type_filter != "all":
        filtered = [r for r in filtered if r.get("_type") == type_filter]

    # 日期范围过滤
    if date_range:
        start_date, end_date = date_range
        filtered = [
            r for r in filtered
            if start_date <= datetime.fromisoformat(
                r.get("completed_at", r.get("created_at", "1970-01-01"))
            ) <= end_date
        ]

    # 模型过滤
    if model_filter:
        filtered = [
            r for r in filtered
            if any(
                model in str(r.get("predictions", {}).keys())
                for model in model_filter
            )
        ]

    return filtered


def render_results_table(results: List[Dict[str, Any]], show_details: bool = False):
    """渲染结果表格"""
    if not results:
        st.info("暂无历史记录")
        return

    # 构建表格数据
    rows = []
    for result in results:
        row = {
            "任务ID": result.get("task_id", "")[:20] + "..." if len(result.get("task_id", "")) > 20 else result.get("task_id", ""),
            "类型": "批量" if result.get("_type") == "batch" else "单结构",
            "化学式": result.get("formula", result.get("structure_info", {}).get("formula", "")),
            "原子数": result.get("n_atoms", result.get("structure_info", {}).get("n_atoms", "")),
            "时间": result.get("completed_at", result.get("created_at", ""))[:19],
        }

        # 添加第一个预测值
        predictions = result.get("predictions", {})
        if isinstance(predictions, dict):
            first_model = list(predictions.keys())[0] if predictions else None
            if first_model:
                first_value = predictions[first_model]
                if isinstance(first_value, dict):
                    row["预测值"] = f"{first_value.get('value', 'N/A'):.4f}" if isinstance(first_value.get('value'), (int, float)) else "N/A"
                else:
                    row["预测值"] = str(first_value)[:10]

        rows.append(row)

    df = pd.DataFrame(rows)

    # 显示表格
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    return df


def render_single_result_detail(result: Dict[str, Any]):
    """渲染单结构结果详情"""
    st.subheader("📊 结构信息")

    structure_info = result.get("structure_info", {})
    if structure_info:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("化学式", structure_info.get("formula", "N/A"))

        with col2:
            st.metric("原子数", structure_info.get("n_atoms", "N/A"))

        with col3:
            st.metric("元素种类", structure_info.get("n_elements", "N/A"))

        with col4:
            lattice = structure_info.get("lattice", {})
            a = lattice.get("a", "N/A")
            st.metric("晶格常数 a", f"{a:.3f}" if isinstance(a, (int, float)) else a)

    st.divider()

    # 预测结果
    st.subheader("🔬 预测结果")

    predictions = result.get("predictions", {})
    if isinstance(predictions, dict):
        for model_key, pred_value in predictions.items():
            model_info = DEFAULT_MODELS.get(model_key, {})
            model_name = model_info.get("name", model_key)

            if isinstance(pred_value, dict):
                with st.container():
                    col_pred1, col_pred2 = st.columns([3, 1])

                    with col_pred1:
                        st.markdown(f"**{model_name}**")
                        value = pred_value.get("value")
                        unit = pred_value.get("unit", "")

                        if isinstance(value, (int, float)):
                            st.metric("预测值", f"{value:.4f}", unit)
                        else:
                            st.text(f"值: {value}")

                    with col_pred2:
                        processing_time = pred_value.get("processing_time", pred_value.get("processing_time_seconds", 0))
                        if processing_time:
                            st.caption(f"⏱ {processing_time:.2f}s")

                        if "error" in pred_value:
                            st.error("预测失败")

                    st.divider()
            elif isinstance(pred_value, (int, float)):
                unit = model_info.get("unit", "")
                st.metric(model_name, f"{pred_value:.4f}", unit)


def render_batch_result_detail(result: Dict[str, Any]):
    """渲染批量结果详情"""
    # 统计
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("总结构数", result.get("total_structures", 0))

    with col_stat2:
        st.metric("成功", result.get("successful", 0), delta_color="normal")

    with col_stat3:
        failed = result.get("failed", 0)
        if failed > 0:
            st.metric("失败", failed, delta_color="inverse")
        else:
            st.metric("失败", 0)

    with col_stat4:
        total_time = result.get("total_time_seconds", 0)
        st.metric("总耗时", f"{total_time:.1f}s")

    st.divider()

    # 结果表格
    st.subheader("📋 结果详情")

    results_list = result.get("results", [])
    if results_list:
        # 构建表格
        rows = []
        for r in results_list:
            row = {
                "文件名": r.get("name", ""),
                "化学式": r.get("formula", ""),
                "原子数": r.get("n_atoms", ""),
            }

            # 添加预测值
            preds = r.get("predictions", {})
            if isinstance(preds, dict):
                for model_key, pred_value in preds.items():
                    if isinstance(pred_value, dict):
                        row[model_key] = f"{pred_value.get('value', 'N/A'):.4f}" if isinstance(pred_value.get('value'), (int, float)) else pred_value.get('error', 'N/A')
                    elif isinstance(pred_value, (int, float)):
                        row[model_key] = f"{pred_value:.4f}"

            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 导出按钮
        st.divider()
        st.subheader("💾 导出结果")

        col_export1, col_export2, col_export3 = st.columns(3)

        with col_export1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 下载 CSV",
                csv_data,
                file_name=f"batch_{result.get('task_id', 'export')}.csv",
                mime="text/csv"
            )

        with col_export2:
            json_data = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 下载 JSON",
                json_data,
                file_name=f"batch_{result.get('task_id', 'export')}.json",
                mime="application/json"
            )

    # 错误详情
    errors = result.get("errors", [])
    if errors:
        with st.expander("⚠️ 查看失败文件详情"):
            for err in errors:
                st.error(f"**{err.get('name', 'Unknown')}**: {err.get('error', 'Unknown error')}")


def main():
    st.set_page_config(
        page_title="历史记录 - ALIGNN",
        page_icon="📁",
        layout="wide"
    )

    st.title("📁 历史预测记录")
    st.markdown("查看和管理所有历史预测结果")

    # 加载结果
    results = load_results()

    if not results:
        st.info("暂无历史记录")
        st.caption("预测结果将自动保存在 alignn_service/results/ 目录下")
        return

    # 筛选控件
    st.sidebar.header("🔍 筛选条件")

    # 搜索框
    search_query = st.sidebar.text_input(
        "搜索",
        placeholder="输入化学式、任务ID或文件名...",
        label_visibility="collapsed"
    )

    # 类型筛选
    type_options = ["all", "single", "batch"]
    type_labels = ["全部", "单结构", "批量"]
    type_filter = st.sidebar.radio(
        "任务类型",
        type_options,
        format_func=lambda x: type_labels[type_options.index(x)],
        horizontal=True
    )

    # 模型筛选
    st.sidebar.subheader("模型筛选")
    selected_models = []
    for model_key, model_info in DEFAULT_MODELS.items():
        if st.sidebar.checkbox(model_info["name"], value=False):
            selected_models.append(model_key)

    # 应用筛选
    filtered_results = filter_results(
        results,
        search_query=search_query,
        type_filter=type_filter,
        model_filter=selected_models if selected_models else None
    )

    # 统计信息
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("总记录数", len(results))
    with col_stat2:
        st.metric("单结构预测", len([r for r in results if r.get("_type") == "single"]))
    with col_stat3:
        st.metric("批量任务", len([r for r in results if r.get("_type") == "batch"]))

    st.divider()

    # 显示筛选结果
    st.subheader(f"📋 记录列表 (共 {len(filtered_results)} 条)")

    # 选择要查看的记录
    if filtered_results:
        # 创建选择框选项
        options = []
        for i, r in enumerate(filtered_results):
            task_id = r.get("task_id", "unknown")[:15]
            formula = r.get("formula", r.get("structure_info", {}).get("formula", "N/A"))
            record_type = "批量" if r.get("_type") == "batch" else "单结构"
            time_str = r.get("completed_at", r.get("created_at", ""))[:10]

            options.append(f"{time_str} | {record_type} | {formula} | {task_id}...")

        selected_idx = st.selectbox(
            "选择记录查看详情",
            options=range(len(options)),
            format_func=lambda i: options[i]
        )

        selected_result = filtered_results[selected_idx]

        # 显示详情
        st.divider()

        # 结果详情标签页
        tab_detail, tab_raw = st.tabs(["📊 详情", "📄 原始数据"])

        with tab_detail:
            if selected_result.get("_type") == "batch":
                render_batch_result_detail(selected_result)
            else:
                render_single_result_detail(selected_result)

        with tab_raw:
            st.json(selected_result, expanded=False)

    else:
        st.info("没有符合条件的记录")

    # 批量操作
    st.divider()
    st.subheader("🔧 批量操作")

    col_batch1, col_batch2, col_batch3 = st.columns(3)

    with col_batch1:
        if st.button("📊 导出所有为 CSV", use_container_width=True):
            if filtered_results:
                # 简化数据用于导出
                export_data = []
                for r in filtered_results:
                    row = {
                        "task_id": r.get("task_id", ""),
                        "type": r.get("_type", ""),
                        "formula": r.get("formula", ""),
                        "n_atoms": r.get("n_atoms", ""),
                        "created_at": r.get("completed_at", r.get("created_at", "")),
                    }

                    # 添加预测值
                    preds = r.get("predictions", {})
                    if isinstance(preds, dict):
                        for model_key, pred_value in preds.items():
                            if isinstance(pred_value, dict) and "value" in pred_value:
                                row[model_key] = pred_value["value"]

                    export_data.append(row)

                df_export = pd.DataFrame(export_data)
                csv_data = df_export.to_csv(index=False)

                st.download_button(
                    "📥 下载 CSV",
                    csv_data,
                    file_name=f"alignn_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    with col_batch2:
        if st.button("📝 导出所有为 JSON", use_container_width=True):
            if filtered_results:
                json_data = json.dumps(filtered_results, indent=2, ensure_ascii=False)

                st.download_button(
                    "📥 下载 JSON",
                    json_data,
                    file_name=f"alignn_history_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

    with col_batch3:
        pass


if __name__ == "__main__":
    main()
