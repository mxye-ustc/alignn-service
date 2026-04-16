#!/usr/bin/env python3
"""
LFP ALIGNN 批量预测脚本 (GPU 优化版)
====================================
自动检测 NVIDIA GPU (CUDA) 或回退到 CPU

用法:
    # 自动检测 GPU
    python predict_lfp_gpu.py

    # 指定运行模型
    python predict_lfp_gpu.py --models all

    # 只运行 JARVIS 模型（6个）
    python predict_lfp_gpu.py --models jarvis

    # 只运行 MP 模型（2个）
    python predict_lfp_gpu.py --models mp

    # 测试模式（3个样本）
    python predict_lfp_gpu.py --limit 3

    # 指定输出目录
    python predict_lfp_gpu.py --output-dir ./results
"""

import os
import sys
import json
import time
import warnings
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from pymatgen.io.vasp import Poscar
from jarvis.core.atoms import Atoms as JAtoms
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

ALIGNN_ROOT = Path("/root/autodl-tmp")
JARVIS_DIR = ALIGNN_ROOT / "alignn" / "models" / "ALIGNN models on JARVIS-DFT dataset"
MP_DIR = ALIGNN_ROOT / "alignn" / "models" / "ALIGNN models on MP dataset"

POSCAR_DIR = ALIGNN_ROOT / "lfp_dopant_configs_v4" / "poscar_files"
OUTPUT_DIR = ALIGNN_ROOT / "lfp_dopant_configs_v4" / "predictions"

CUTOFF = 8.0
MAX_NEIGHBORS = 12

# 模型定义: (存储key, 模型目录, 单位, 标签)
JARVIS_MODELS = [
    ("jv_total_energy",      "jv_optb88vdw_total_energy_alignn",    "eV/atom",   "Etot"),
    ("jv_formation_energy",  "jv_formation_energy_peratom_alignn",  "eV/atom",   "ΔEf"),
    ("jv_ehull",             "jv_ehull_alignn",                     "meV/atom",  "Ehull"),
    ("jv_bandgap",           "jv_optb88vdw_bandgap_alignn",        "eV",        "Eg"),
    ("jv_bulk_modulus",      "jv_bulk_modulus_kv_alignn",           "GPa",       "Kv"),
    ("jv_magmom",            "jv_magmom_oszicar_alignn",            "μB",        "μ"),
]

MP_MODELS = [
    ("mp_formation_energy", "mp_e_form_alignnn", "eV/atom", "ΔEf(MP)"),
    ("mp_bandgap",          "mp_gappbe_alignnn",  "eV",       "Eg(MP)"),
]

# ============================================================================
# 设备检测
# ============================================================================

def get_device():
    """自动检测可用设备"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n{'='*60}")
        print(f"  GPU 检测到: {gpu_name}")
        print(f"  显存: {gpu_mem:.1f} GB")
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"{'='*60}\n")
    else:
        device = "cpu"
        print(f"\n{'='*60}")
        print(f"  未检测到 NVIDIA GPU，使用 CPU")
        print(f"{'='*60}\n")
    return device

# ============================================================================
# 模型加载
# ============================================================================

def load_model(model_dir: Path, device: str):
    """加载单个模型到指定设备"""
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "checkpoint_300.pt"

    with open(config_path) as f:
        cfg = json.load(f)

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
        mmap=True
    )

    model = ALIGNN(ALIGNNConfig(**cfg["model"]))
    model.load_state_dict(state["model"])
    model.to(device).eval()

    del state
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    return model

# ============================================================================
# 图构建
# ============================================================================

def pmg_to_jarvis(structure) -> JAtoms:
    """pymatgen Structure -> JARVIS Atoms"""
    return JAtoms(
        elements=[s.symbol for s in structure.species],
        coords=structure.cart_coords.tolist(),
        lattice_mat=structure.lattice.matrix.tolist(),
        cartesian=True,
    )


def build_graph(jatoms, device: str):
    """构建 DGL 图并移到设备"""
    g, lg = Graph.atom_dgl_multigraph(
        jatoms,
        cutoff=CUTOFF,
        atom_features="cgcnn",
        max_neighbors=MAX_NEIGHBORS,
        compute_line_graph=True,
        use_canonize=True
    )
    lat = torch.tensor(jatoms.lattice_mat, dtype=torch.float32)
    return g.to(device), lg.to(device), lat.to(device)


# ============================================================================
# 预测函数
# ============================================================================

def predict_one(g, lg, lat, model):
    """使用模型预测单个图"""
    try:
        with torch.no_grad():
            out = model([g, lg, lat])
        return round(float(out.detach().cpu().numpy().flatten()[0]), 6)
    except Exception as e:
        return None


def print_progress(current, total, start_time, model_name=""):
    """打印进度信息"""
    elapsed = time.time() - start_time
    avg_time = elapsed / current
    remaining = avg_time * (total - current)
    percent = current / total * 100

    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"\r  [{bar}] {percent:5.1f}%  "
          f"[{current:4d}/{total}]  "
          f"avg: {avg_time:.1f}s  "
          f"remaining: {remaining/3600:.1f}h    ",
          end="", flush=True)


# ============================================================================
# 批量预测
# ============================================================================

def run_prediction(poscar_files, model_list, base_dir, device,
                   model_group, output_prefix):
    """执行批量预测"""
    global OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_files = len(poscar_files)
    prop_names = [m[0] for m in model_list]

    print(f"\n{'='*60}")
    print(f"  模型组: {model_group}")
    print(f"  构型数:  {n_files}")
    print(f"  模型数:  {len(model_list)}")
    print(f"  设备:    {device.upper()}")
    print(f"{'='*60}\n")

    # 阶段1: 预读取所有结构
    print(f"[1/2] 预读取结构...")
    structures = {}
    for i, (poscar_path, config_id) in enumerate(poscar_files):
        try:
            p = Poscar.from_file(poscar_path, check_for_potcar=False)
            structures[config_id] = pmg_to_jarvis(p.structure)
        except Exception:
            structures[config_id] = None

        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{n_files}")

    valid_count = sum(1 for v in structures.values() if v is not None)
    print(f"      有效结构: {valid_count}/{n_files}\n")

    # 阶段2: 逐模型预测
    all_results = {}
    total_start = time.time()

    for model_idx, (name, local_dir, unit, label) in enumerate(model_list):
        model_path = base_dir / local_dir

        print(f"\n[{model_idx+1}/{len(model_list)}] {label}")
        print(f"    模型: {local_dir}")

        # 加载模型
        load_start = time.time()
        model = load_model(model_path, device)
        print(f"    加载: {time.time() - load_start:.1f}s")

        # 预测
        model_preds = {}
        pred_start = time.time()

        for i, (config_id, jatoms) in enumerate(structures.items()):
            if jatoms is None:
                model_preds[config_id] = None
                continue

            try:
                g, lg, lat = build_graph(jatoms, device)
                val = predict_one(g, lg, lat, model)
                model_preds[config_id] = val
                del g, lg
            except Exception:
                model_preds[config_id] = None

            # 进度
            if (i + 1) % 50 == 0 or (i + 1) == len(structures):
                print_progress(i + 1, len(structures), pred_start, label)

        print()  # 换行

        # 保存单个模型结果
        model_elapsed = time.time() - pred_start
        ok_count = sum(1 for v in model_preds.values() if v is not None)
        speed = ok_count / model_elapsed if model_elapsed > 0 else 0

        print(f"    完成: {ok_count}/{n_files}  耗时: {model_elapsed:.0f}s  速度: {speed:.2f}/s")

        # 保存文件
        model_json = OUTPUT_DIR / f"{output_prefix}_{name}.json"
        model_csv = OUTPUT_DIR / f"{output_prefix}_{name}.csv"

        with open(model_json, "w") as f:
            json.dump({
                "model": {
                    "name": name, "label": label, "unit": unit,
                    "local_dir": local_dir
                },
                "predictions": model_preds,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": device,
            }, f, indent=2)

        with open(model_csv, "w") as f:
            f.write("config_id,prediction\n")
            for cid, val in model_preds.items():
                f.write(f"{cid},{val if val is not None else ''}\n")

        print(f"    写入: {model_json.name}")

        # 累积结果
        for cid, val in model_preds.items():
            if cid not in all_results:
                all_results[cid] = {"config_id": cid}
            all_results[cid][name] = val

        # 释放模型
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # 合并结果
    results = [all_results[cid] for cid in sorted(all_results.keys())]
    total_elapsed = time.time() - total_start

    # 保存合并文件
    final_json = OUTPUT_DIR / f"{output_prefix}_predictions.json"
    final_csv = OUTPUT_DIR / f"{output_prefix}_predictions.csv"

    ok_total = sum(
        1 for r in results
        if all(r.get(p) is not None for p in prop_names)
    )

    with open(final_json, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_group": model_group,
                "n_structures": n_files,
                "n_complete": ok_total,
                "models": [(m[0], m[1], m[2], m[3]) for m in model_list],
                "cutoff": CUTOFF,
                "max_neighbors": MAX_NEIGHBORS,
                "device": device,
                "total_time_seconds": round(total_elapsed, 1),
            },
            "predictions": results,
        }, f, indent=2, ensure_ascii=False)

    with open(final_csv, "w") as f:
        header = ["config_id"] + prop_names
        f.write(",".join(header) + "\n")
        for r in results:
            row = [r.get("config_id", "")]
            for p in prop_names:
                v = r.get(p)
                if v is not None and not np.isnan(v):
                    row.append(f"{v:.6f}")
                else:
                    row.append("")
            f.write(",".join(row) + "\n")

    # 统计信息
    print(f"\n{'='*60}")
    print(f"  {model_group} 完成!")
    print(f"  总耗时: {total_elapsed/3600:.2f}h")
    print(f"  全部成功: {ok_total}/{n_files}")
    print(f"{'='*60}")

    print(f"\n  输出文件:")
    print(f"    JSON: {final_json}")
    print(f"    CSV:  {final_csv}")

    print(f"\n  属性统计:")
    print(f"  {'-'*55}")
    print(f"  {'属性':<10} {'单位':<10} {'最小':>12} {'最大':>12} {'均值':>12}")
    print(f"  {'-'*55}")
    for m in model_list:
        name, _, unit, label = m
        vals = [r[name] for r in results
                if r.get(name) is not None
                and not np.isnan(r[name])]
        if vals:
            print(f"  {label:<10} {unit:<10} {min(vals):12.4f} {max(vals):12.4f} {np.mean(vals):12.4f}")
    print(f"  {'-'*55}")

    return results, total_elapsed

# ============================================================================
# 合并结果
# ============================================================================

def merge_all_predictions():
    """合并所有预测结果"""
    global OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = {}

    # JARVIS 模型
    for name, local_dir, unit, label in JARVIS_MODELS:
        fpath = OUTPUT_DIR / f"jarvis_alignn_{name}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for cid, val in data.get("predictions", {}).items():
                if cid not in all_data:
                    all_data[cid] = {"config_id": cid}
                all_data[cid][name] = val

    # MP 模型
    for name, local_dir, unit, label in MP_MODELS:
        fpath = OUTPUT_DIR / f"mp_alignn_{name}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for cid, val in data.get("predictions", {}).items():
                if cid not in all_data:
                    all_data[cid] = {"config_id": cid}
                all_data[cid][name] = val

    if not all_data:
        print("没有找到预测结果文件")
        return

    results = [all_data[cid] for cid in sorted(all_data.keys())]

    combined_json = OUTPUT_DIR / "combined_predictions.json"
    combined_csv = OUTPUT_DIR / "combined_predictions.csv"

    props_order = ["jv_total_energy", "jv_formation_energy", "jv_ehull", "jv_bandgap",
                   "jv_bulk_modulus", "jv_magmom", "mp_formation_energy", "mp_bandgap"]
    props_labels = ["Etot", "ΔEf", "Ehull", "Eg", "Kv", "μ", "ΔEf(MP)", "Eg(MP)"]

    with open(combined_json, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n_structures": len(results),
                "sources": ["jarvis_alignn", "mp_alignn"],
            },
            "predictions": results,
        }, f, indent=2, ensure_ascii=False)

    with open(combined_csv, "w") as f:
        f.write("config_id," + ",".join(props_labels) + "\n")
        for r in results:
            row = [r.get("config_id", "")]
            for p in props_order:
                v = r.get(p)
                if v is not None and not np.isnan(v):
                    row.append(f"{v:.6f}")
                else:
                    row.append("")
            f.write(",".join(row) + "\n")

    print(f"\n{'='*60}")
    print(f"  合并完成: {len(results)} 条记录")
    print(f"  JSON: {combined_json}")
    print(f"  CSV:  {combined_csv}")
    print(f"{'='*60}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFP ALIGNN GPU 批量预测")
    parser.add_argument("--models", default="all",
                        choices=["jarvis", "mp", "all"],
                        help="选择模型组: jarvis(6个), mp(2个), all(全部8个)")
    parser.add_argument("--poscar-dir", type=Path, default=None,
                        help="POSCAR 文件目录")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前N个文件（测试用）")
    parser.add_argument("--force-cpu", action="store_true",
                        help="强制使用 CPU（忽略 GPU）")

    args = parser.parse_args()

    # 设置路径
    poscar_dir = args.poscar_dir or POSCAR_DIR
    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 检测设备
    device = "cpu" if args.force_cpu else get_device()

    # 获取文件列表
    if not poscar_dir.exists():
        print(f"错误: POSCAR 目录不存在: {poscar_dir}")
        sys.exit(1)

    all_files = sorted([
        f for f in os.listdir(poscar_dir)
        if f.endswith(".poscar") and not f.startswith("._")
    ])

    if args.limit > 0:
        all_files = all_files[:args.limit]
        print(f"[测试模式] 只处理前 {len(all_files)} 个文件\n")

    print(f"找到 {len(all_files)} 个 POSCAR 文件")

    poscar_paths = [
        (poscar_dir / f, f.replace(".poscar", ""))
        for f in all_files
    ]

    # 时间估算
    avg_time_per_sample = 11.0 if device == "cpu" else 2.0  # 秒
    n_models = len(JARVIS_MODELS) if args.models in ("jarvis", "all") else 0
    n_models += len(MP_MODELS) if args.models in ("mp", "all") else 0
    estimated_time = len(all_files) * n_models * avg_time_per_sample / 3600

    print(f"预估时间: ~{estimated_time:.1f} 小时")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_start = time.time()

    # 运行预测
    if args.models in ("jarvis", "all"):
        print(f"\n{'#'*60}")
        print(f"#  JARVIS-DFT 模型 (6 个属性)")
        print(f"{'#'*60}")
        run_prediction(
            poscar_paths, JARVIS_MODELS, JARVIS_DIR, device,
            "JARVIS-DFT", "jarvis_alignn"
        )

    if args.models in ("mp", "all"):
        print(f"\n{'#'*60}")
        print(f"#  MP 模型 (2 个属性)")
        print(f"{'#'*60}")
        run_prediction(
            poscar_paths, MP_MODELS, MP_DIR, device,
            "Materials Project", "mp_alignn"
        )

    if args.models == "all":
        merge_all_predictions()

    # 总结
    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"#  全部完成!")
    print(f"#  总耗时: {total_elapsed/3600:.2f} 小时")
    print(f"#  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
