#!/usr/bin/env python
"""
Ti掺杂LFP浓度系列 ALIGNN 批量预测脚本
========================================
预测60个Ti掺杂构型的关键性质：
- 形成能 (JARVIS + MP)
- 凸包能
- 带隙
- 体模量
- 磁矩
"""

import os
import sys
import json
import time
import glob
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================
POSCAR_DIR  = "/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/ti_concentration_series/poscar_files"
OUTPUT_DIR  = "/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/ti_concentration_series"
OUT_CSV     = os.path.join(OUTPUT_DIR, "ti_concentration_predictions.csv")
OUT_JSON    = os.path.join(OUTPUT_DIR, "ti_concentration_predictions.json")

JARVIS_PROPS = [
    "jv_formation_energy_peratom_alignn",
    "jv_optb88vdw_total_energy_alignn",
    "jv_ehull_alignn",
    "jv_optb88vdw_bandgap_alignn",
    "jv_bulk_modulus_kv_alignn",
    "jv_magmom_oszicar_alignn",
]

MP_BANDGAP_MODEL = "mp_gappbe_alignnn"

# ============================================================================
# 延迟导入
# ============================================================================
os.chdir("/Users/mxye/Myprojects/alignn")
import torch
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.pretrained import device

torch.set_num_threads(8)

# ============================================================================
# 模型加载
# ============================================================================

def load_jarvis_model(model_name):
    """从本地 checkpoint 加载 JARVIS ALIGNN 模型"""
    base_dir = "/Users/mxye/Myprojects/alignn/alignn/models/ALIGNN models on JARVIS-DFT dataset"
    ckpt_dir = os.path.join(base_dir, model_name)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_300.pt")
    cfg_path  = os.path.join(ckpt_dir, "config.json")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model_cfg = ALIGNNConfig(**cfg["model"])
    model = ALIGNN(config=model_cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def load_mp_model():
    """从本地 checkpoint 加载 MP ALIGNN formation energy 模型"""
    base_dir = "/Users/mxye/Myprojects/alignn/alignn/models/ALIGNN models on MP dataset"
    ckpt_path = os.path.join(base_dir, "mp_e_form_alignnn/checkpoint_300.pt")
    cfg_path  = os.path.join(base_dir, "mp_e_form_alignnn/config.json")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model_cfg = ALIGNNConfig(**cfg["model"])
    model = ALIGNN(config=model_cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def load_mp_bandgap_model():
    """从本地 checkpoint 加载 MP ALIGNN bandgap 模型"""
    base_dir = "/Users/mxye/Myprojects/alignn/alignn/models/ALIGNN models on MP dataset"
    model_name = MP_BANDGAP_MODEL
    ckpt_path = os.path.join(base_dir, f"{model_name}/checkpoint_300.pt")
    cfg_path  = os.path.join(base_dir, f"{model_name}/config.json")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model_cfg = ALIGNNConfig(**cfg["model"])
    model = ALIGNN(config=model_cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def predict_single(atoms, model):
    """用已加载模型对单个结构做预测"""
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=8.0, max_neighbors=12)
    lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model([g.to(device), lg.to(device), lat])
    return out.item()


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("Ti掺杂LFP浓度系列 ALIGNN 批量预测")
    print("=" * 70)

    # 收集 POSCAR 文件
    poscar_files = sorted(glob.glob(os.path.join(POSCAR_DIR, "*.poscar")))
    print(f"\n找到 {len(poscar_files)} 个 POSCAR 文件")

    # 加载模型
    print("\n=== [1/3] 加载模型 ===")
    t0 = time.time()

    jarvis_models = {}
    for mname in JARVIS_PROPS:
        t1 = time.time()
        jarvis_models[mname] = load_jarvis_model(mname)
        print(f"  {mname}: OK ({time.time()-t1:.1f}s)")

    t2 = time.time()
    mp_model = load_mp_model()
    print(f"  mp_e_form_alignn: OK ({time.time()-t2:.1f}s)")

    t3 = time.time()
    mp_bandgap_model = load_mp_bandgap_model()
    print(f"  mp_gappbe_alignn: OK ({time.time()-t3:.1f}s)")

    print(f"  所有模型加载完成 ({time.time()-t0:.1f}s)")

    # 批量预测
    print(f"\n=== [2/3] 批量预测 {len(poscar_files)} 个构型 ===")

    results = []
    total_start = time.time()

    for i, poscar_path in enumerate(poscar_files):
        fname = os.path.basename(poscar_path)
        t_batch = time.time()

        # 解析文件名
        # 格式: LFP_Fe=Ti_conc01_n1_c1_4x2x2.poscar
        name_part = fname.replace(".poscar", "")
        parts = name_part.split("_")

        # 提取信息
        dopant_elem = "Ti"

        # 安全提取
        conc_str = None
        n_str = None
        c_str = None
        sc_str = None
        for p in parts:
            if p.startswith("conc"):
                conc_str = p
            elif p.startswith("n") and p[1:].isdigit():
                n_str = p
            elif p.startswith("c") and p[1:].isdigit():
                c_str = p
            elif "x" in p:
                sc_str = p

        if conc_str is None or n_str is None or c_str is None or sc_str is None:
            print(f"  [WARN] {fname}: 无法解析文件名")
            continue

        target_conc = int(conc_str.replace("conc", ""))
        n_dopant = int(n_str[1:])
        config_idx = int(c_str[1:])

        # 读取 POSCAR
        try:
            atoms = Atoms.from_poscar(poscar_path)
        except Exception as e:
            print(f"  [WARN] {fname}: 读取失败 {e}")
            continue

        row = {
            "filename": fname,
            "dopant_element": dopant_elem,
            "target_concentration_pct": target_conc,
            "n_dopant": n_dopant,
            "config_idx": config_idx,
            "supercell": sc_str,
            "natoms": atoms.num_atoms,
            "formula": atoms.composition.reduced_formula,
        }

        # JARVIS 属性预测
        for mname in JARVIS_PROPS:
            try:
                val = predict_single(atoms, jarvis_models[mname])
                row[mname] = round(val, 6)
            except Exception as e:
                row[mname] = None
                print(f"  [WARN] {fname} {mname}: {e}")

        # MP formation energy
        try:
            val_mp = predict_single(atoms, mp_model)
            row["mp_e_form_alignn"] = round(val_mp, 6)
        except Exception as e:
            row["mp_e_form_alignn"] = None
            print(f"  [WARN] {fname} mp_e_form: {e}")

        # MP bandgap
        try:
            val_mp_gap = predict_single(atoms, mp_bandgap_model)
            row["mp_gappbe_alignn"] = round(val_mp_gap, 6)
        except Exception as e:
            row["mp_gappbe_alignn"] = None
            print(f"  [WARN] {fname} mp_gappbe: {e}")

        row["total_time_s"] = round(time.time() - t_batch, 2)
        results.append(row)

        # 进度打印
        if (i + 1) % 10 == 0 or i == 0 or i == len(poscar_files) - 1:
            elapsed = time.time() - total_start
            avg = elapsed / (i + 1)
            eta = avg * (len(poscar_files) - i - 1)
            print(f"  [{i+1:2d}/{len(poscar_files)}] {fname} "
                  f"| gap={row.get('jv_optb88vdw_bandgap_alignn','N/A')} "
                  f"| E_hull={row.get('jv_ehull_alignn','N/A')} "
                  f"| ETA={eta/60:.1f}min")

    # 保存结果
    print(f"\n=== [3/3] 保存结果 ===")

    all_keys = list(results[0].keys()) if results else []

    # CSV
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)

    # JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"  CSV: {OUT_CSV}")
    print(f"  JSON: {OUT_JSON}")

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)
    print(f"  总计: {len(results)}/{len(poscar_files)} 构型")
    print(f"  总耗时: {total_elapsed/60:.1f} min")
    print(f"  平均每构型: {total_elapsed/len(poscar_files):.1f}s")


if __name__ == "__main__":
    main()
