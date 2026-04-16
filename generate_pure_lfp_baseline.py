#!/usr/bin/env python
"""
纯LiFePO4基线验证脚本
======================
生成不同超胞大小的纯LFP，验证ALIGNN预测是否与文献值一致
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/ti_concentration_series")

def write_poscar(filepath, lattice, species, coords):
    species_count = {}
    for s in species:
        species_count[s] = species_count.get(s, 0) + 1
    species_ordered = sorted(species_count.keys())
    atom_counts = [species_count[s] for s in species_ordered]
    
    with open(filepath, "w") as f:
        f.write("Pure LiFePO4 (Baseline)\n")
        f.write("  1.0\n")
        for row in lattice:
            f.write(f"  {row[0]:15.10f}  {row[1]:15.10f}  {row[2]:15.10f}\n")
        f.write("  " + "  ".join(species_ordered) + "\n")
        f.write("  " + "  ".join(str(c) for c in atom_counts) + "\n")
        f.write("Cartesian\n")
        sp_to_coords = {sp: [] for sp in species_ordered}
        for i, sp_i in enumerate(species):
            sp_to_coords[sp_i].append(coords[i])
        for sp in species_ordered:
            for c in sp_to_coords[sp]:
                f.write(f"  {c[0]:15.10f}  {c[1]:15.10f}  {c[2]:15.10f}\n")


def main():
    print("=" * 70)
    print("纯LiFePO4基线验证")
    print("=" * 70)
    
    # 加载原始结构
    from pymatgen.core import Structure
    cif_path = "/Users/mxye/Myprojects/alignn/alignn/models/LiFePO4.cif"
    structure = Structure.from_file(cif_path)
    
    print(f"\n原始原胞: {structure.composition.reduced_formula}, {len(structure)} 原子")
    
    # 超胞尺寸列表（与Ti掺杂系列一致）
    supercells = [
        ((1, 1, 1), "1x1x1"),   # 原胞
        ((2, 2, 1), "2x2x1"),   # 112原子
        ((3, 3, 2), "3x3x2"),   # 504原子
        ((4, 2, 2), "4x2x2"),   # 448原子
        ((8, 6, 3), "8x6x3"),   # 大超胞
    ]
    
    poscar_dir = OUTPUT_DIR / "poscar_files"
    os.makedirs(poscar_dir, exist_ok=True)
    
    all_configs = []
    
    print("\n生成不同超胞的纯LFP:")
    for sc_dims, sc_name in supercells:
        print(f"  {sc_name}: ", end="")
        
        supercell = structure.copy()
        supercell.make_supercell(list(sc_dims))
        
        lattice = supercell.lattice.matrix.copy()
        coords = supercell.cart_coords.copy()
        species = np.array([
            str(s).replace("+", "").replace("-", "").replace("2", "").replace("3", "").replace("5", "")
            for s in supercell.species
        ])
        
        config_id = f"LFP_pure_{sc_name}"
        poscar_path = poscar_dir / f"{config_id}.poscar"
        
        write_poscar(poscar_path, lattice, species, coords)
        
        n_atoms = len(species)
        fe_count = sum(1 for s in species if s == "Fe")
        li_count = sum(1 for s in species if s == "Li")
        o_count = sum(1 for s in species if s == "O")
        p_count = sum(1 for s in species if s == "P")
        
        print(f"{n_atoms} atoms (Fe:{fe_count}, Li:{li_count}, O:{o_count}, P:{p_count})")
        
        all_configs.append({
            "config_id": config_id,
            "supercell": sc_name,
            "n_atoms": n_atoms,
            "formula": f"Li{li_count}Fe{fe_count}P{p_count}O{o_count}",
            "poscar_path": str(poscar_path),
        })
    
    # 保存配置列表
    with open(OUTPUT_DIR / "pure_lfp_configs.json", "w") as f:
        json.dump(all_configs, f, indent=2)
    
    print(f"\n已生成 {len(all_configs)} 个纯LFP构型")
    print("=" * 70)


if __name__ == "__main__":
    main()
