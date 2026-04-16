#!/usr/bin/env python3
"""一次性修复 /root/autodl-tmp/scripts/predict_lfp_gpu.py 的所有问题"""

SCRIPT = "/root/autodl-tmp/scripts/predict_lfp_gpu.py"

with open(SCRIPT) as f:
    c = f.read()

# ======================
# 1. JARVIS_MODELS: key 改成完整目录名（和 all_results 存的 key 一致）
# ======================
old_jarvis = '''JARVIS_MODELS = [
    ("jv_total_energy",      "jv_optb88vdw_total_energy_alignn",    "eV/atom",   "Etot"),
    ("jv_formation_energy",  "jv_formation_energy_peratom_alignn",  "eV/atom",   "ΔEf"),
    ("jv_ehull",             "jv_ehull_alignn",                     "meV/atom",  "Ehull"),
    ("jv_bandgap",           "jv_optb88vdw_bandgap_alignn",        "eV",        "Eg"),
    ("jv_bulk_modulus",      "jv_bulk_modulus_kv_alignn",           "GPa",       "Kv"),
    ("jv_magmom",            "jv_magmom_oszicar_alignn",            "μB",        "μ"),
]'''

new_jarvis = '''JARVIS_MODELS = [
    ("jv_optb88vdw_total_energy_alignn",    "jv_optb88vdw_total_energy_alignn",    "eV/atom",   "Etot"),
    ("jv_formation_energy_peratom_alignn",  "jv_formation_energy_peratom_alignn",  "eV/atom",   "ΔEf"),
    ("jv_ehull_alignn",                     "jv_ehull_alignn",                     "meV/atom",  "Ehull"),
    ("jv_optb88vdw_bandgap_alignn",         "jv_optb88vdw_bandgap_alignn",         "eV",        "Eg"),
    ("jv_bulk_modulus_kv_alignn",          "jv_bulk_modulus_kv_alignn",           "GPa",       "Kv"),
    ("jv_magmom_oszicar_alignn",           "jv_magmom_oszicar_alignn",            "μB",        "μ"),
]'''

if old_jarvis in c:
    c = c.replace(old_jarvis, new_jarvis)
    print("OK: JARVIS_MODELS 已修复（key = 完整目录名）")
elif '("jv_total_energy"' in c:
    print("WARN: 旧 JARVIS_MODELS 格式，用逐行替换...")
    c = c.replace('("jv_total_energy",      "jv_optb88vdw_total_energy_alignn"', '("jv_optb88vdw_total_energy_alignn",    "jv_optb88vdw_total_energy_alignn"')
    c = c.replace('("jv_formation_energy",  "jv_formation_energy_peratom_alignn"', '("jv_formation_energy_peratom_alignn",  "jv_formation_energy_peratom_alignn"')
    c = c.replace('("jv_ehull",             "jv_ehull_alignn"', '("jv_ehull_alignn",                     "jv_ehull_alignn"')
    c = c.replace('("jv_bandgap",           "jv_optb88vdw_bandgap_alignn"', '("jv_optb88vdw_bandgap_alignn",         "jv_optb88vdw_bandgap_alignn"')
    c = c.replace('("jv_bulk_modulus",      "jv_bulk_modulus_kv_alignn"', '("jv_bulk_modulus_kv_alignn",          "jv_bulk_modulus_kv_alignn"')
    c = c.replace('("jv_magmom",            "jv_magmom_oszicar_alignn"', '("jv_magmom_oszicar_alignn",           "jv_magmom_oszicar_alignn"')
    print("OK: JARVIS_MODELS 已修复（逐行替换）")
else:
    print("INFO: JARVIS_MODELS 已是正确格式或无法识别")

# ======================
# 2. MP_MODELS: 恢复正确的 local_dir（不要去掉 mp_ 前缀）
# ======================
c = c.replace('("mp_formation_energy", "e_form_alignnn"', '("mp_e_form_alignnn", "mp_e_form_alignnn"')
c = c.replace('("mp_formation_energy",  "e_form_alignnn"', '("mp_e_form_alignnn", "mp_e_form_alignnn"')
c = c.replace('("mp_bandgap",          "gappbe_alignnn"', '("mp_gappbe_alignnn", "mp_gappbe_alignnn"')
c = c.replace('("mp_bandgap",           "gappbe_alignnn"', '("mp_gappbe_alignnn", "mp_gappbe_alignnn"')
print("OK: MP_MODELS local_dir 已恢复")

# ======================
# 3. 修复 merge props_order
# ======================
c = c.replace(
    'props_order = ["jv_total_energy", "jv_formation_energy", "jv_ehull", "jv_bandgap",',
    'props_order = ["jv_optb88vdw_total_energy_alignn", "jv_formation_energy_peratom_alignn", "jv_ehull_alignn", "jv_optb88vdw_bandgap_alignn",'
)
c = c.replace(
    '"jv_bulk_modulus", "jv_magmom", "mp_formation_energy", "mp_bandgap"]',
    '"jv_bulk_modulus_kv_alignn", "jv_magmom_oszicar_alignn", "mp_e_form_alignnn", "mp_gappbe_alignnn"]'
)
# 也处理旧格式
c = c.replace(
    '"bulk_modulus", "magmom", "mp_formation_energy", "mp_bandgap"]',
    '"jv_bulk_modulus_kv_alignn", "jv_magmom_oszicar_alignn", "mp_e_form_alignnn", "mp_gappbe_alignnn"]'
)
print("OK: props_order 已修复")

# ======================
# 4. 修复 merge_all_predictions JARVIS 文件查找
# ======================
old_jarvis_merge = '''    for name, local_dir, unit, label in JARVIS_MODELS:
        for fname in [f"jarvis_alignn_{name}.json", f"jarvis_alignn_jarvis_alignn_{name}.json"]:
            fpath = OUTPUT_DIR / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                for cid, val in data.get("predictions", {}).items():
                    if cid not in all_data:
                        all_data[cid] = {"config_id": cid}
                    all_data[cid][name] = val
                break'''

new_jarvis_merge = '''    for name, local_dir, unit, label in JARVIS_MODELS:
        fpath = OUTPUT_DIR / f"{output_prefix}_{name}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for cid, val in data.get("predictions", {}).items():
                if cid not in all_data:
                    all_data[cid] = {"config_id": cid}
                all_data[cid][name] = val'''

if old_jarvis_merge in c:
    c = c.replace(old_jarvis_merge, new_jarvis_merge)
    print("OK: merge JARVIS 文件查找已修复")
else:
    # 也尝试其他变体
    c = c.replace(
        'for fname in [f"jarvis_alignn_{name}.json", f"jarvis_alignn_jarvis_alignn_{name}.json"]:',
        ''
    )
    c = c.replace(
        '            fpath = OUTPUT_DIR / fname\n            if fpath.exists():',
        '            fpath = OUTPUT_DIR / f"{output_prefix}_{name}.json"\n            if fpath.exists():'
    )
    if 'break' in c and 'for fname in' not in c:
        # 去掉 break
        import re
        c = re.sub(r'\n\s*break\s*(?=\n\s*(?:    #|#|\n))', '\n', c)
    print("OK: merge JARVIS 文件查找已修复（变体）")

# ======================
# 5. 统计打印加 np.isnan
# ======================
c = c.replace(
    'if r.get(name) is not None\n                and not np.isnan(r[name])',
    'if r.get(name) is not None and not np.isnan(r[name])'
)
print("OK: 统计打印已修复（单行）")

# ======================
# 6. merge CSV NaN 检查
# ======================
c = c.replace(
    'row.append(f"{v:.6f}" if v is not None else "")',
    'if v is not None and not np.isnan(v):\n                    row.append(f"{v:.6f}")\n                else:\n                    row.append("")'
)
print("OK: merge CSV NaN 检查已修复")

# 写回文件
with open(SCRIPT, "w") as f:
    f.write(c)

print("\n文件已写回:", SCRIPT)

# ======================
# 验证
# ======================
import re
print("\n===== 验证 =====")
m = re.search(r'JARVIS_MODELS = \[(.*?)\]', c, re.DOTALL)
if m:
    print("JARVIS_MODELS:")
    for line in m.group(1).strip().split('\n'):
        if '("' in line:
            print(' ', line.strip())

m = re.search(r'MP_MODELS = \[(.*?)\]', c, re.DOTALL)
if m:
    print("MP_MODELS:")
    for line in m.group(1).strip().split('\n'):
        if '("' in line:
            print(' ', line.strip())

m = re.search(r'props_order = \[(.*?)\]', c, re.DOTALL)
if m:
    print("props_order:")
    print(' ', m.group(1).replace('\n',' ').strip())

ok = 'jarvis_alignn_jarvis_alignn' not in c
print(f"双重路径清除: {'OK' if ok else 'FAIL'}")
