#!/usr/bin/env python
"""
纯LiFePO4 ALIGNN 预测脚本
==========================
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
OUT_CSV     = os.path.join(OUTPUT_DIR, "pure_lfp_predictions.csv")
OUT_JSON    = os.path.join(OUTPUT_DIR, "pure_lfp_predictions.json")

JARVIS_PROPS = [
    "jv_formation_energy_peratom_alignn",
    "jv_optb88vdw_total_energy_alignn",
    "jv_ehull_alignn",
    "jv_optb88vdw_bandgap_alignn",
    "jv_bulk_modulus_kv_alignn",
    "jv_magmom_oszicar_alignn",
]

MP_FORM_MODEL = "mp_e_form_alignnn"
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


def load_mp_model(model_subdir):
    base_dir = "/Users/mxye/Myprojects/alignn/alignn/models/ALIGNN models on MP dataset"
    ckpt_path = os.path.join(base_dir, f"{model_subdir}/checkpoint_300.pt")
    cfg_path  = os.path.join(base_dir, f"{model_subdir}/config.json")

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
    print("纯LiFePO4 ALIGNN 预测")
    print("=" * 70)

    # 只预测纯LFP文件
    poscar_files = sorted(glob.glob(os.path.join(POSCAR_DIR, "LFP_pure_*.poscar")))
    print(f"\n找到 {len(poscar_files)} 个纯LFP POSCAR 文件")

    # 加载模型
    print("\n=== [1/3] 加载模型 ===")
    t0 = time.time()

    jarvis_models = {}
    for mname in JARVIS_PROPS:
        t1 = time.time()
        jarvis_models[mname] = load_jarvis_model(mname)
        print(f"  {mname}: OK ({time.time()-t1:.1f}s)")

    t2 = time.time()
    mp_form_model = load_mp_model(MP_FORM_MODEL)
    print(f"  mp_e_form_alignn: OK ({time.time()-t2:.1f}s)")

    t3 = time.time()
    mp_gap_model = load_mp_model(MP_BANDGAP_MODEL)
    print(f"  mp_gappbe_alignn: OK ({time.time()-t3:.1f}s)")

    print(f"  所有模型加载完成 ({time.time()-t0:.1f}s)")

    # 批量预测
    print(f"\n=== [2/3] 批量预测 {len(poscar_files)} 个纯LFP构型 ===")

    results = []
    total_start = time.time()

    for i, poscar_path in enumerate(poscar_files):
        fname = os.path.basename(poscar_path)
        t_batch = time.time()

        # 读取 POSCAR
        try:
            atoms = Atoms.from_poscar(poscar_path)
        except Exception as e:
            print(f"  [WARN] {fname}: 读取失败 {e}")
            continue

        row = {
            "filename": fname,
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
            val_mp = predict_single(atoms, mp_form_model)
            row["mp_e_form_alignn"] = round(val_mp, 6)
        except Exception as e:
            row["mp_e_form_alignn"] = None
            print(f"  [WARN] {fname} mp_e_form: {e}")

        # MP bandgap
        try:
            val_mp_gap = predict_single(atoms, mp_gap_model)
            row["mp_gappbe_alignn"] = round(val_mp_gap, 6)
        except Exception as e:
            row["mp_gappbe_alignn"] = None
            print(f"  [WARN] {fname} mp_gappbe: {e}")

        row["total_time_s"] = round(time.time() - t_batch, 2)
        results.append(row)

        print(f"  [{i+1}/{len(poscar_files)}] {fname} | "
              f"MP_gap={row.get('mp_gappbe_alignn','N/A'):.3f} | "
              f"JV_gap={row.get('jv_optb88vdw_bandgap_alignn','N/A'):.3f} | "
              f"E_hull={row.get('jv_ehull_alignn','N/A'):.4f}")

    # 保存结果
    print(f"\n=== [3/3] 保存结果 ===")

    import csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"  CSV: {OUT_CSV}")
    print(f"  JSON: {OUT_JSON}")

    print("\n" + "=" * 70)
    print("纯LiFePO4 基线验证结果")
    print("=" * 70)
    
    # 文献参考值
    print("\n文献参考值:")
    print("  MP/PBE 带隙: 3.694 eV")
    print("  JV/optB88vdW 带隙: 3.708 eV")
    print("  JV 凸包能: 2.6381 eV/atom")
    print("  JV 体模量: 92.31 GPa")
    
    print("\nALIGNN预测结果:")
    print(f"{'构型':>15} | {'原子数':>6} | {'MP带隙':>10} | {'JV带隙':>10} | {'E_hull':>10} | {'体模量':>10}")
    print("-" * 75)
    for r in results:
        sc_name = r['filename'].replace('LFP_pure_', '').replace('.poscar', '')
        print(f"{sc_name:>15} | {r['natoms']:>6} | {r['mp_gappbe_alignn']:>10.4f} | {r['jv_optb88vdw_bandgap_alignn']:>10.4f} | {r['jv_ehull_alignn']:>10.4f} | {r['jv_bulk_modulus_kv_alignn']:>10.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
