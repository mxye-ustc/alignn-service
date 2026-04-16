#!/usr/bin/env python
"""
批量预测脚本：600个LFP掺杂构型
同时使用两套本地模型：
  1. JARVIS-DFT 预训练模型（6个属性，本地 checkpoint）
  2. MP 预训练模型（Formation energy，本地 checkpoint）
"""
import os, sys, json, time, glob
import numpy as np

POSCAR_DIR  = "/root/autodl-tmp/lfp_dopant_configs_v4/poscar_files"
JARVIS_BASE = "/root/autodl-tmp/alignn/models/ALIGNN models on JARVIS-DFT dataset"
MP_BASE     = "/root/autodl-tmp/alignn/models/ALIGNN models on MP dataset"
OUT_DIR     = "/root/autodl-tmp/prediction_results"
OUT_CSV     = os.path.join(OUT_DIR, "all_predictions.csv")
OUT_JSON    = os.path.join(OUT_DIR, "all_predictions.json")

JARVIS_PROPS = [
    "jv_formation_energy_peratom_alignn",
    "jv_optb88vdw_total_energy_alignn",
    "jv_ehull_alignn",
    "jv_optb88vdw_bandgap_alignn",
    "jv_bulk_modulus_kv_alignn",
    "jv_magmom_oszicar_alignn",
]

# ──────────────────────────────────────────────
# 延迟导入，避免卡住
# ──────────────────────────────────────────────
os.chdir("/root/autodl-tmp")
import torch
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.pretrained import device

torch.set_num_threads(8)
os.makedirs(OUT_DIR, exist_ok=True)

def load_jarvis_local(model_name):
    """从本地 checkpoint 加载 JARVIS ALIGNN 模型"""
    ckpt_dir = os.path.join(JARVIS_BASE, model_name)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_300.pt")
    cfg_path  = os.path.join(ckpt_dir, "config.json")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model_cfg = ALIGNNConfig(**cfg["model"])
    model = ALIGNN(config=model_cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def load_mp_local():
    """从本地 checkpoint 加载 MP ALIGNN formation energy 模型"""
    ckpt_path = os.path.join(MP_BASE, "mp_e_form_alignnn/checkpoint_300.pt")
    cfg_path  = os.path.join(MP_BASE, "mp_e_form_alignnn/config.json")

    with open(cfg_path) as f:
        cfg = json.load(f)

    model_cfg = ALIGNNConfig(**cfg["model"])
    model = ALIGNN(config=model_cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def predict_single(atoms, model):
    """用已加载模型对单个结构做预测，返回标量"""
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=8.0,
        max_neighbors=12,
    )
    lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model([g.to(device), lg.to(device), lat])
    return out.item()


# ──────────────────────────────────────────────
# 收集所有 POSCAR 文件
# ──────────────────────────────────────────────
poscar_files = sorted(glob.glob(os.path.join(POSCAR_DIR, "*.poscar")))
print(f"Found {len(poscar_files)} POSCAR files")
print(f"Output: {OUT_CSV}")

# ──────────────────────────────────────────────
# 预加载所有模型
# ──────────────────────────────────────────────
print("\n=== [1/3] 加载模型 ===")
t0 = time.time()

jarvis_models = {}
for mname in JARVIS_PROPS:
    t1 = time.time()
    jarvis_models[mname] = load_jarvis_local(mname)
    print(f"  {mname}: OK ({time.time()-t1:.1f}s)")

t2 = time.time()
mp_model = load_mp_local()
print(f"  mp_e_form_alignn: OK ({time.time()-t2:.1f}s)")

print(f"  所有模型加载完成 ({time.time()-t0:.1f}s)")

# ──────────────────────────────────────────────
# 批量预测
# ──────────────────────────────────────────────
print(f"\n=== [2/3] 批量预测 {len(poscar_files)} 个构型 ===")

results = []
total_start = time.time()

for i, poscar_path in enumerate(poscar_files):
    fname = os.path.basename(poscar_path)
    t_batch = time.time()

    # 解析文件名提取元信息
    # 格式: LFP_Fe=Co_x01736_n1_c1.poscar 或 LFP_P=S_x01736_n1_c1.poscar 等
    name_part = fname.replace(".poscar", "")
    # 提取掺杂位点和元素
    if "Fe=" in name_part:
        dopant_type = "Fe-site"
        dopant_elem = name_part.split("Fe=")[1].split("_")[0]
    elif "Li=" in name_part:
        dopant_type = "Li-site"
        dopant_elem = name_part.split("Li=")[1].split("_")[0]
    elif "P=" in name_part:
        dopant_type = "P-site"
        dopant_elem = name_part.split("P=")[1].split("_")[0]
    else:
        dopant_type = "unknown"
        dopant_elem = "unknown"

    parts = name_part.split("_")
    x_hex   = [p for p in parts if p.startswith("x")][0]
    n_val   = [p for p in parts if p.startswith("n")][0]
    c_val   = [p for p in parts if p.startswith("c")][0]
    n_atoms = int(n_val[1:])
    x_frac  = n_atoms / 576.0

    # 读取 POSCAR
    try:
        atoms = Atoms.from_poscar(poscar_path)
    except Exception as e:
        print(f"  [WARN] {fname}: 读取失败 {e}")
        continue

    row = {
        "filename":        fname,
        "dopant_type":     dopant_type,
        "dopant_element":  dopant_elem,
        "n_dopant":        n_atoms,
        "x_fraction":      round(x_frac, 5),
        "config":          c_val,
        "natoms":          atoms.num_atoms,
        "formula":         atoms.composition.reduced_formula,
    }

    # JARVIS 属性
    for mname in JARVIS_PROPS:
        t_pred = time.time()
        try:
            val = predict_single(atoms, jarvis_models[mname])
            row[mname] = round(val, 6)
        except Exception as e:
            row[mname] = None
            print(f"  [WARN] {fname} {mname}: {e}")
        row[f"{mname}_time_s"] = round(time.time() - t_pred, 2)

    # MP formation energy
    t_pred = time.time()
    try:
        val_mp = predict_single(atoms, mp_model)
        row["mp_e_form_alignn"] = round(val_mp, 6)
    except Exception as e:
        row["mp_e_form_alignn"] = None
        print(f"  [WARN] {fname} mp_e_form: {e}")
    row["mp_e_form_alignn_time_s"] = round(time.time() - t_pred, 2)

    row["total_time_s"] = round(time.time() - t_batch, 2)
    results.append(row)

    # 进度打印
    elapsed = time.time() - total_start
    avg = elapsed / (i + 1)
    eta = avg * (len(poscar_files) - i - 1)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1}/{len(poscar_files)}] {fname} "
              f"| E_form={row.get('jv_formation_energy_peratom_alignn','N/A')} "
              f"| time={row['total_time_s']}s "
              f"| ETA={eta/60:.1f}min")

# ──────────────────────────────────────────────
# 保存结果
# ──────────────────────────────────────────────
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
print(f"\n=== 完成 ===")
print(f"  总计: {len(results)}/{len(poscar_files)} 构型")
print(f"  总耗时: {total_elapsed/60:.1f} min")
print(f"  平均每构型: {total_elapsed/len(poscar_files):.1f}s")
