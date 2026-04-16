#!/usr/bin/env python
"""
Ti掺杂LFP浓度系列构型生成器
===========================
生成1%-20%浓度范围，每1%取3个随机构型，共60个

超胞设计策略：
- 使用2×2×1超胞（16个Fe位点）
- 覆盖范围：n=1(6.25%) 到 n=5(31.25%)
- 浓度计算：n/16 × 100%

对于1%这样的低浓度，使用更大的超胞（如3×3×2=72个Fe位点）

生成60个构型：20个浓度点 × 3个随机构型/浓度
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================================
# 配置参数
# ============================================================================
SUPERCELL_2X2X1 = (2, 2, 1)  # 16个Fe位点, 112原子
SUPERCELL_3X3X2 = (3, 3, 2)  # 72个Fe位点, 504原子 (用于低浓度)
SUPERCELL_4X2X2 = (4, 2, 2)  # 64个Fe位点, 448原子 (用于低浓度)

CONCENTRATIONS = list(range(1, 21))  # 1% 到 20%
CONFIGS_PER_CONC = 3  # 每个浓度3个构型

OUTPUT_DIR = Path("/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/ti_concentration_series")
POSCAR_DIR = OUTPUT_DIR / "poscar_files"
os.makedirs(POSCAR_DIR, exist_ok=True)

# ============================================================================
# POSCAR 写入
# ============================================================================

def write_poscar(filepath, lattice, species_ordered, species_copy, coords, atom_counts):
    with open(filepath, "w") as f:
        f.write("Ti-doped LFP Concentration Series\n")
        f.write("  1.0\n")
        for row in lattice:
            f.write(f"  {row[0]:15.10f}  {row[1]:15.10f}  {row[2]:15.10f}\n")
        f.write("  " + "  ".join(species_ordered) + "\n")
        f.write("  " + "  ".join(str(c) for c in atom_counts) + "\n")
        f.write("Cartesian\n")
        # 按物种分组写坐标
        sp_to_coords = {sp: [] for sp in species_ordered}
        for i, sp_i in enumerate(species_copy):
            sp_to_coords[sp_i].append(coords[i])
        for sp in species_ordered:
            for c in sp_to_coords[sp]:
                f.write(f"  {c[0]:15.10f}  {c[1]:15.10f}  {c[2]:15.10f}\n")


def generate_config_id(site, dopant, concentration, config_idx, n_dopant, supercell_name):
    return f"LFP_{site}={dopant}_conc{concentration:02d}_n{n_dopant}_c{config_idx}_{supercell_name}"


# ============================================================================
# 主函数
# ============================================================================

def main():
    t_start = time.time()
    random.seed(42)
    
    print("=" * 70)
    print("Ti掺杂LFP浓度系列构型生成器")
    print("浓度范围: 1% - 20%, 分辨率 1%, 每浓度3个随机构型")
    print("=" * 70)
    
    # 加载原始结构
    print("\n[1/4] 加载原始 LiFePO4 结构...")
    from pymatgen.core import Structure
    
    cif_path = "/Users/mxye/Myprojects/alignn/alignn/models/LiFePO4.cif"
    structure = Structure.from_file(cif_path)
    print(f"    原始: {structure.composition.reduced_formula}, {len(structure)} 原子")
    
    # Fe位点数量计算
    fe_count_per_primitive = 4  # 原胞有4个Fe
    
    # 计算每个超胞的Fe位点数
    fe_2x2x1 = fe_count_per_primitive * 2 * 2 * 1  # = 16
    fe_3x3x2 = fe_count_per_primitive * 3 * 3 * 2  # = 72
    fe_4x2x2 = fe_count_per_primitive * 4 * 2 * 2  # = 64
    
    print(f"\n超胞Fe位点配置:")
    print(f"  2×2×1: {fe_2x2x1} 个Fe位点, {fe_2x2x1 * 7} 原子 (浓度范围: 6.25% - 31.25%)")
    print(f"  3×3×2: {fe_3x3x2} 个Fe位点, {fe_3x3x2 * 7} 原子 (浓度范围: 1.39% - 20.83%)")
    print(f"  4×2×2: {fe_4x2x2} 个Fe位点, {fe_4x2x2 * 7} 原子 (浓度范围: 1.56% - 23.44%)")
    
    # 浓度到n的映射（选择最接近的超胞）
    def get_n_for_conc(conc, fe_count):
        """给定浓度和Fe位点数，计算需要替换的Fe数量"""
        n = round(conc / 100 * fe_count)
        return max(1, min(n, fe_count - 1))  # 至少1个，最多fe_count-1个
    
    def choose_supercell(conc):
        """为给定浓度选择最佳超胞"""
        # 1-5%: 用4×2×2 (64 Fe位点)
        if conc <= 5:
            return (4, 2, 2), fe_4x2x2
        # 6-20%: 用2×2×1 (16 Fe位点)
        elif conc <= 20:
            return (2, 2, 1), fe_2x2x1
        else:
            return (2, 2, 1), fe_2x2x1
    
    # 生成所有构型任务
    print("\n[2/4] 生成构型任务...")
    all_tasks = []
    
    for conc in CONCENTRATIONS:
        for config_idx in range(1, CONFIGS_PER_CONC + 1):
            task_counter = len(all_tasks) + 1
            rng = random.Random(42 + task_counter * 1000)
            
            all_tasks.append({
                "concentration": conc,
                "config_idx": config_idx,
                "seed": 42 + task_counter * 1000,
            })
    
    print(f"    总任务数: {len(all_tasks)}")
    
    # 存储所有元数据
    all_metadata = []
    
    print("\n[3/4] 生成构型...")
    
    for i, task in enumerate(all_tasks):
        conc = task["concentration"]
        config_idx = task["config_idx"]
        rng = random.Random(task["seed"])
        
        # 选择超胞
        sc_dims, fe_count = choose_supercell(conc)
        n_dopant = get_n_for_conc(conc, fe_count)
        
        # 创建超胞
        supercell = structure.copy()
        supercell.make_supercell(list(sc_dims))
        
        # 提取数据
        lattice = supercell.lattice.matrix.copy()
        coords = supercell.cart_coords.copy()
        species_arr = np.array([
            str(s).replace("+", "").replace("-", "").replace("2", "").replace("3", "").replace("5", "")
            for s in supercell.species
        ])
        
        # 获取Fe位点索引
        fe_indices = np.where(species_arr == "Fe")[0].tolist()
        
        # 随机选择要替换的Fe位点
        replace_indices = rng.sample(fe_indices, n_dopant)
        
        # 创建掺杂后的物种数组
        species_copy = species_arr.copy()
        for idx in replace_indices:
            species_copy[idx] = "Ti"
        
        # 计算实际浓度
        actual_conc = n_dopant / fe_count * 100
        
        # 生成config_id
        supercell_name = f"{sc_dims[0]}x{sc_dims[1]}x{sc_dims[2]}"
        config_id = generate_config_id("Fe", "Ti", conc, config_idx, n_dopant, supercell_name)
        
        # 原子计数
        species_count = Counter(species_copy)
        species_ordered = sorted(species_count.keys())
        atom_counts = [species_count[s] for s in species_ordered]
        
        # 写入POSCAR
        poscar_path = POSCAR_DIR / f"{config_id}.poscar"
        write_poscar(poscar_path, lattice, species_ordered, species_copy, coords, atom_counts)
        
        # 保存元数据
        metadata = {
            "config_id": config_id,
            "doping_site": "Fe",
            "dopant": "Ti",
            "target_concentration_pct": conc,
            "actual_concentration_pct": round(actual_conc, 2),
            "n_dopant_atoms": n_dopant,
            "fe_sites_total": fe_count,
            "supercell": list(sc_dims),
            "supercell_name": supercell_name,
            "config_idx": config_idx,
            "n_atoms": len(species_copy),
            "formula": "".join(f"{s}{c}" for s, c in zip(species_ordered, atom_counts)),
            "poscar_path": f"poscar_files/{config_id}.poscar",
        }
        all_metadata.append(metadata)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:2d}/{len(all_tasks)}] {config_id} "
                  f"(目标{conc}%, 实际{actual_conc:.1f}%, {len(species_copy)}原子)")
    
    # 保存元数据
    print("\n[4/4] 保存元数据...")
    
    # CSV格式
    import csv
    csv_path = OUTPUT_DIR / "ti_concentration_configs.csv"
    with open(csv_path, "w", newline="") as f:
        if all_metadata:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)
    
    # JSON格式
    json_path = OUTPUT_DIR / "ti_concentration_configs.json"
    with open(json_path, "w") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    # 统计
    print("\n" + "=" * 70)
    print("生成统计")
    print("=" * 70)
    
    by_conc = {}
    for m in all_metadata:
        conc = m["actual_concentration_pct"]
        by_conc.setdefault(conc, []).append(m["config_id"])
    
    print(f"\n按浓度分布 ({len(by_conc)} 个浓度点):")
    for conc in sorted(by_conc.keys()):
        configs = by_conc[conc]
        print(f"  {conc:5.2f}%: {len(configs)} 个构型")
    
    # 超胞使用统计
    sc_usage = Counter(m["supercell_name"] for m in all_metadata)
    print(f"\n超胞使用统计:")
    for sc, cnt in sc_usage.items():
        print(f"  {sc}: {cnt} 个构型")
    
    print(f"\n总构型数: {len(all_metadata)}")
    print(f"总原子数范围: {min(m['n_atoms'] for m in all_metadata)} - {max(m['n_atoms'] for m in all_metadata)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n耗时: {time.time() - t_start:.1f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
