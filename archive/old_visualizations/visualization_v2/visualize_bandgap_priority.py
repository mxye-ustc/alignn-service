#!/usr/bin/env python3
"""
ALIGNN 预测结果可视化分析脚本 (带隙降幅优先标准)
基于机器学习加速筛选磷酸铁锂掺杂改性方案

⚠️ 重要说明：MP vs JARVIS 预测可靠性
----------------------------------------------------------------------
MP (Materials Project) 预测：主要参考，可靠
  - 带隙：MP=3.694 eV，与实验值~3.7 eV完全一致
  - 形成能：MP=-2.532 eV/atom，与JARVIS高度相关（r=0.94）

JARVIS 预测：部分参考，可信度因属性而异
  - 带隙：JARVIS=0.058 eV，严重低估，不可信
  - 形成能：JARVIS=-2.070 eV/atom，与MP高度相关，可信
  - E_hull：JARVIS DFT计算，可参考

关键发现：
  - MP与JARVIS带隙预测：几乎不相关（r=0.074）
  - MP与JARVIS形成能预测：高度相关（r=0.943）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from math import pi

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

# 路径配置
DATA_DIR = Path('/Users/mxye/Myprojects/alignn/alignn/models')
OUTPUT_DIR = Path('/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/visualization_v2')
OUTPUT_DIR.mkdir(exist_ok=True)

# 读取数据
df = pd.read_csv(DATA_DIR / 'all_predictions.csv')

# 简化列名
df = df.rename(columns={
    'jv_formation_energy_peratom_alignn': 'E_form_JV',
    'jv_ehull_alignn': 'E_hull_JV',
    'jv_optb88vdw_bandgap_alignn': 'Bandgap_JV',  # ⚠️ JARVIS带隙，绝对值不可信
    'jv_bulk_modulus_kv_alignn': 'BulkMod_JV',
    'jv_magmom_oszicar_alignn': 'MagMom_JV',
    'mp_e_form_alignn': 'E_form_MP',
    'mp_gappbe_alignn': 'Bandgap_MP'  # ★ MP带隙，主要参考指标
})

# 原始LFP参考值
REF = {
    'E_form_JV': -2.06981,
    'E_hull_JV': 2.6381,
    'Bandgap_JV': 0.05774,  # ⚠️ JARVIS带隙（严重低估）
    'BulkMod_JV': 92.31,
    'MagMom_JV': 8.9124,
    'E_form_MP': -2.53219,
    'Bandgap_MP': 3.694  # ★ MP带隙是主要关注指标（与实验值一致）
}

# 排除原始LFP，计算带隙降幅
df_doped = df[df['dopant_element'] != 'none'].copy()
df_doped['BG_reduction_MP'] = (REF['Bandgap_MP'] - df_doped['Bandgap_MP']) / REF['Bandgap_MP'] * 100
df_doped['BG_reduction_JV'] = (REF['Bandgap_JV'] - df_doped['Bandgap_JV']) / REF['Bandgap_JV'] * 100

# 计算综合得分（带隙降幅优先）
# 权重：带隙降幅50% + 热力学稳定性30% + 体模量20%
df_doped['E_hull_stability'] = 1 - (df_doped['E_hull_JV'] - REF['E_hull_JV']) / 0.1  # 归一化
df_doped['BG_score'] = df_doped['BG_reduction_MP'] / 40  # 归一化到0-1
df_doped['BulkMod_score'] = df_doped['BulkMod_JV'] / REF['BulkMod_JV']
df_doped['Comprehensive_Score'] = (
    0.50 * df_doped['BG_score'].clip(0, 1) +
    0.30 * df_doped['E_hull_stability'].clip(0, 1.2) +
    0.20 * df_doped['BulkMod_score'].clip(0.8, 1.2)
)

# 颜色配置
SITE_COLORS = {'Fe-site': '#E74C3C', 'Li-site': '#3498DB', 'P-site': '#2ECC71'}

print("=" * 70)
print("ALIGNN 预测结果可视化分析 (带隙降幅优先标准)")
print("=" * 70)
print(f"总构型数: {len(df_doped)}")
print(f"原始LFP MP带隙: {REF['Bandgap_MP']} eV")

# ============================================================
# 图1: 带隙降幅分布直方图（核心图表）
# ============================================================
def plot_bandgap_reduction_distribution():
    """带隙降幅分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MP带隙降幅
    ax = axes[0]
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        ax.hist(subset['BG_reduction_MP'], bins=30, alpha=0.6, label=site, color=SITE_COLORS[site])

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_xlabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('MP带隙降幅分布（按掺杂位点）', fontsize=14, fontweight='bold')
    ax.legend()

    # JV带隙降幅
    ax = axes[1]
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        ax.hist(subset['BG_reduction_JV'], bins=30, alpha=0.6, label=site, color=SITE_COLORS[site])

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_xlabel('JARVIS带隙降幅 (%)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('JARVIS带隙降幅分布（按掺杂位点）', fontsize=14, fontweight='bold')
    ax.legend()

    plt.suptitle('带隙调控效果分布（负值表示带隙降低）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_bandgap_reduction_distribution.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 01_bandgap_reduction_distribution.png")

# ============================================================
# 图2: 浓度-带隙降幅趋势图（关键发现）
# ============================================================
def plot_concentration_bandgap_trend():
    """各元素带隙降幅随浓度变化趋势"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 代表性元素（按带隙降幅排序选择）
    key_elements = ['Ti', 'Nb', 'Mn', 'Cu', 'Co', 'Ni', 'Cr', 'Si', 'Mg']

    # MP带隙
    ax = axes[0]
    for elem in key_elements:
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['BG_reduction_MP'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, markersize=6, linewidth=2)

    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('MP带隙降幅随浓度变化趋势', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # JV带隙
    ax = axes[1]
    for elem in key_elements:
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['BG_reduction_JV'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, markersize=6, linewidth=2)

    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('JARVIS带隙降幅 (%)', fontsize=12)
    ax.set_title('JARVIS带隙降幅随浓度变化趋势', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('各元素带隙调控的浓度依赖性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_concentration_bandgap_trend.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 02_concentration_bandgap_trend.png")

# ============================================================
# 图3: 综合得分Top 10方案
# ============================================================
def plot_top10_comprehensive():
    """综合得分Top 10最优方案"""
    # 按元素和浓度计算平均得分
    grouped = df_doped.groupby(['dopant_element', 'n_dopant']).agg({
        'Comprehensive_Score': 'mean',
        'BG_reduction_MP': 'mean',
        'E_hull_JV': 'mean',
        'Bandgap_MP': 'mean'
    }).reset_index()

    top10 = grouped.nlargest(10, 'Comprehensive_Score')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 3a: 综合得分柱状图
    ax = axes[0]
    labels = [f"{row['dopant_element']} (n={int(row['n_dopant'])})" for _, row in top10.iterrows()]
    colors = ['#E74C3C' if row['n_dopant'] >= 8 else '#F39C12' if row['n_dopant'] >= 5 else '#3498DB'
             for _, row in top10.iterrows()]
    bars = ax.barh(labels[::-1], top10['Comprehensive_Score'].values[::-1], color=colors[::-1])

    ax.set_xlabel('综合得分 (带隙50% + 热力学30% + 体模量20%)', fontsize=11)
    ax.set_title('Top 10 最优掺杂方案', fontsize=14, fontweight='bold')

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{bar.get_width():.3f}', va='center', fontsize=10)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', label='高浓度(n≥8)'),
                      Patch(facecolor='#F39C12', label='中浓度(5≤n<8)'),
                      Patch(facecolor='#3498DB', label='低浓度(n<5)')]
    ax.legend(handles=legend_elements, loc='lower right')

    # 3b: 带隙降幅 vs 热力学稳定性
    ax = axes[1]
    for idx, (_, row) in enumerate(top10.iterrows()):
        ax.scatter(row['E_hull_JV'] - REF['E_hull_JV'], row['BG_reduction_MP'],
                  s=200, c=colors[idx], edgecolors='black', linewidth=1)
        ax.annotate(f"{row['dopant_element']}", 
                   (row['E_hull_JV'] - REF['E_hull_JV'], row['BG_reduction_MP']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('E_hull变化 (eV)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('带隙降幅 vs 热力学稳定性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注理想区域
    ax.annotate('理想区域\n(高带隙降幅 + 低E_hull)', xy=(0.02, 30), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_top10_comprehensive.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 03_top10_comprehensive.png")

# ============================================================
# 图4: Fe位元素带隙降幅对比
# ============================================================
def plot_fe_site_bandgap_comparison():
    """Fe位各掺杂元素带隙降幅详细对比"""
    fe_elements = ['Ti', 'Nb', 'Mn', 'Cu', 'Co', 'Ni', 'Cr', 'Zn', 'V', 'Mo', 'Er', 'Y', 'Nd']
    df_fe = df_doped[df_doped['dopant_type'] == 'Fe-site']

    # 按元素分组
    elem_stats = df_fe.groupby('dopant_element').agg({
        'BG_reduction_MP': 'mean',
        'BG_reduction_JV': 'mean',
        'E_hull_JV': 'mean',
        'Bandgap_MP': 'mean'
    }).reindex(fe_elements)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a: MP带隙降幅
    ax = axes[0, 0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(fe_elements)))
    bars = ax.bar(fe_elements, elem_stats['BG_reduction_MP'], color=colors)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('Fe位元素 MP带隙降幅对比', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='gray', linestyle='--')
    for bar, val in zip(bars, elem_stats['BG_reduction_MP']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 4b: JV带隙降幅
    ax = axes[0, 1]
    bars = ax.bar(fe_elements, elem_stats['BG_reduction_JV'], color=colors)
    ax.set_ylabel('JARVIS带隙降幅 (%)', fontsize=12)
    ax.set_title('Fe位元素 JARVIS带隙降幅对比', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='gray', linestyle='--')

    # 4c: MP带隙绝对值
    ax = axes[1, 0]
    bars = ax.bar(fe_elements, elem_stats['Bandgap_MP'], color=colors)
    ax.axhline(REF['Bandgap_MP'], color='red', linestyle='--', linewidth=2, label=f'原始LFP: {REF["Bandgap_MP"]} eV')
    ax.set_ylabel('MP带隙 (eV)', fontsize=12)
    ax.set_title('Fe位元素 MP带隙绝对值', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    # 4d: 热力学稳定性
    ax = axes[1, 1]
    bars = ax.bar(fe_elements, elem_stats['E_hull_JV'] - REF['E_hull_JV'], color=colors)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_ylabel('E_hull变化 (eV)', fontsize=12)
    ax.set_title('Fe位元素 热力学稳定性变化', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    plt.suptitle('Fe位13种掺杂元素详细对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_fe_site_bandgap_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 04_fe_site_bandgap_comparison.png")

# ============================================================
# 图5: 热力学拐点分析
# ============================================================
def plot_thermodynamic_onset():
    """热力学稳定起始浓度分析"""
    key_elements = ['Ti', 'Nb', 'Mn', 'Cu', 'Cr', 'Si', 'Mg']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # E_hull随浓度变化
    ax = axes[0]
    for elem in key_elements:
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['E_hull_JV'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, markersize=6, linewidth=2)

    ax.axhline(REF['E_hull_JV'] + 0.05, color='red', linestyle='--', linewidth=2,
              label='稳定性阈值 (+0.05 eV)')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('E_hull (eV)', fontsize=12)
    ax.set_title('E_hull随浓度变化趋势', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 带隙降幅随浓度变化
    ax = axes[1]
    for elem in key_elements:
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['BG_reduction_MP'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, markersize=6, linewidth=2)

    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('MP带隙降幅随浓度变化趋势', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('热力学稳定起始浓度分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_thermodynamic_onset.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 05_thermodynamic_onset.png")

# ============================================================
# 图6: Ti/Nb/Mn 高浓度效应详图
# ============================================================
def plot_top3_detailed():
    """Top 3方案(Ti/Nb/Mn)详细分析"""
    top3_elements = ['Ti', 'Nb', 'Mn']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 6a: 带隙随浓度变化
    ax = axes[0, 0]
    for elem, color in zip(top3_elements, ['#E74C3C', '#3498DB', '#2ECC71']):
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant').agg({
            'Bandgap_MP': 'mean',
            'Bandgap_JV': 'mean'
        })
        ax.plot(subset.index, subset['Bandgap_MP'], 'o-', label=f'{elem} (MP)',
               color=color, markersize=8, linewidth=2)

    ax.axhline(REF['Bandgap_MP'], color='black', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('MP带隙 (eV)', fontsize=12)
    ax.set_title('Top 3方案 MP带隙 vs 浓度', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注关键浓度点
    for elem, n_val, bg_val in [('Ti', 10, 2.710), ('Nb', 10, 2.594), ('Mn', 10, 2.627)]:
        ax.annotate(f'{elem}: {bg_val} eV', xy=(n_val, bg_val), xytext=(n_val-2, bg_val+0.2),
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    # 6b: 带隙降幅百分比
    ax = axes[0, 1]
    for elem, color in zip(top3_elements, ['#E74C3C', '#3498DB', '#2ECC71']):
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['BG_reduction_MP'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, color=color, markersize=8, linewidth=2)

    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('带隙降幅 (%)', fontsize=12)
    ax.set_title('Top 3方案 带隙降幅百分比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(25, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('25%阈值', xy=(9, 25), fontsize=10, color='gray')

    # 6c: E_hull变化
    ax = axes[1, 0]
    for elem, color in zip(top3_elements, ['#E74C3C', '#3498DB', '#2ECC71']):
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['E_hull_JV'].mean()
        ax.plot(subset.index, subset.values - REF['E_hull_JV'], 'o-', label=elem, color=color, markersize=8, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('E_hull变化 (eV)', fontsize=12)
    ax.set_title('Top 3方案 热力学稳定性变化', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6d: 综合得分
    ax = axes[1, 1]
    for elem, color in zip(top3_elements, ['#E74C3C', '#3498DB', '#2ECC71']):
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['Comprehensive_Score'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, color=color, markersize=8, linewidth=2)

    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('综合得分', fontsize=12)
    ax.set_title('Top 3方案 综合得分变化', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Top 3 最优掺杂方案详细分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_top3_detailed.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 06_top3_detailed.png")

# ============================================================
# 图7: P位和Li位排除依据
# ============================================================
def plot_p_li_exclusion():
    """P位和Li位掺杂不推荐的依据"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P位分析
    ax = axes[0]
    p_elements = ['Si', 'S']
    for elem, color in zip(p_elements, ['#F39C12', '#E74C3C']):
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant').agg({
            'BG_reduction_MP': 'mean',
            'E_hull_JV': 'mean'
        })
        ax.plot(subset.index, subset['BG_reduction_MP'], 'o-', label=f'{elem} 带隙降幅', color=color, linewidth=2)
        ax.plot(subset.index, (subset['E_hull_JV'] - REF['E_hull_JV']) * 100, 's--', label=f'{elem} E_hull×100', color=color, alpha=0.5)

    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='带隙降幅阈值(1%)')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('带隙降幅 (%) / E_hull变化×100', fontsize=12)
    ax.set_title('P位掺杂：带隙调控效果极弱', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 添加结论框
    ax.annotate('结论：P位掺杂\n带隙降幅<1%\n已明确排除',
               xy=(0.95, 0.05), xycoords='axes fraction',
               fontsize=11, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='#FFCCCC', alpha=0.8))

    # Li位分析
    ax = axes[1]
    li_elements = ['Mg', 'Al', 'Na', 'Ti', 'W']
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(li_elements)))
    for elem, color in zip(li_elements, colors):
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['BG_reduction_MP'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, color=color, linewidth=2)

    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='带隙降幅阈值(1%)')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('Li位掺杂：带隙调控效果较弱', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 添加结论框
    ax.annotate('结论：Li位掺杂\n带隙降幅<3%\n不作为首选方案',
               xy=(0.95, 0.05), xycoords='axes fraction',
               fontsize=11, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='#FFEE99', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_p_li_exclusion.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 07_p_li_exclusion.png")

# ============================================================
# 图8: 热力学边界分析
# ============================================================
def plot_thermodynamic_boundary():
    """热力学边界构型识别"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 所有Fe位构型的散点图
    ax = axes[0]
    df_fe = df_doped[df_doped['dopant_type'] == 'Fe-site']
    scatter = ax.scatter(df_fe['n_dopant'], df_fe['E_hull_JV'] - REF['E_hull_JV'],
                       c=df_fe['BG_reduction_MP'], cmap='RdYlGn', s=30, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='MP带隙降幅 (%)')

    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='稳定性阈值')
    ax.axhline(0.10, color='orange', linestyle='--', linewidth=2, label='稳定性上限')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('E_hull变化 (eV)', fontsize=12)
    ax.set_title('Fe位掺杂：热力学边界分析', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注可行区域
    ax.fill_between([0, 10], 0, 0.05, alpha=0.1, color='green')
    ax.annotate('热力学稳定区\n(E_hull变化<0.05 eV)', xy=(2, 0.03), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 热力学可行构型统计
    ax = axes[1]
    df_fe['is_feasible'] = (df_fe['E_hull_JV'] - REF['E_hull_JV']) < 0.05

    feasible_counts = df_fe.groupby(['dopant_element', 'is_feasible']).size().unstack(fill_value=0)
    fe_elem_order = ['Ti', 'Nb', 'Mn', 'Cu', 'Co', 'Ni', 'Cr', 'Zn', 'V', 'Mo', 'Er', 'Y', 'Nd']
    feasible_counts = feasible_counts.reindex([e for e in fe_elem_order if e in feasible_counts.index])
    feasible_counts.plot(kind='bar', ax=ax, color=['#E74C3C', '#2ECC71'])
    ax.set_xlabel('掺杂元素', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('各元素热力学稳定构型数量', fontsize=14, fontweight='bold')
    ax.legend(['不稳定', '稳定'])
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_thermodynamic_boundary.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 08_thermodynamic_boundary.png")

# ============================================================
# 图9: 最终推荐方案汇总
# ============================================================
def plot_final_recommendation():
    """最终推荐方案汇总图"""
    fig = plt.figure(figsize=(16, 10))

    # 创建表格数据
    recommendations = [
        ['D-1', 'Ti', 'Fe位', '1.736%', '2.710', '26.6%', '2.668', '0.030', '★★★★★'],
        ['D-2', 'Nb', 'Fe位', '1.736%', '2.594', '29.8%', '2.708', '0.070', '★★★★☆'],
        ['D-3', 'Mn', 'Fe位', '1.736%', '2.627', '28.9%', '2.696', '0.058', '★★★★☆'],
        ['备选-1', 'Cu', 'Fe位', '1.736%', '2.447', '33.8%', '2.682', '0.044', '★★★★☆'],
        ['备选-2', 'Co', 'Fe位', '1.736%', '2.500', '32.3%', '2.690', '0.052', '★★★☆☆'],
    ]

    ax = fig.add_subplot(111)
    ax.axis('off')

    # 创建表格
    table = ax.table(
        cellText=recommendations,
        colLabels=['方案', '元素', '位点', '浓度', 'MP带隙', '带隙降幅', 'E_hull', 'E_hull变化', '推荐度'],
        loc='center',
        cellLoc='center',
        colWidths=[0.09, 0.08, 0.08, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(len(recommendations[0])):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 设置行颜色
    for i in range(1, len(recommendations) + 1):
        for j in range(len(recommendations[0])):
            if i == 1:  # D-1
                table[(i, j)].set_facecolor('#D5F5E3')
            elif i <= 3:  # D-2, D-3
                table[(i, j)].set_facecolor('#FCF3CF')
            else:  # 备选
                table[(i, j)].set_facecolor('#FADBD8')

    plt.suptitle('最终推荐掺杂方案汇总', fontsize=16, fontweight='bold', y=0.95)

    # 添加说明文本
    note_text = """
    筛选标准：带隙降幅优先（权重50%）+ 热力学稳定性（权重30%）+ 体模量保持（权重20%）
    原始LFP基准：MP带隙=3.694 eV, E_hull=2.638 eV
    D-1方案：综合最优，兼顾带隙调控和热力学稳定性
    D-2/D-3方案：带隙降幅更显著，热力学稳定性略差
    """
    plt.figtext(0.5, 0.05, note_text, ha='center', fontsize=10, style='italic')

    plt.savefig(OUTPUT_DIR / '09_final_recommendation.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 已保存: 09_final_recommendation.png")

# ============================================================
# 图10: 属性相关性分析
# ============================================================
def plot_correlation_analysis():
    """各属性间相关性分析"""
    corr_cols = ['n_dopant', 'E_form_JV', 'E_hull_JV', 'Bandgap_JV', 'Bandgap_MP',
                'BulkMod_JV', 'BG_reduction_MP', 'Comprehensive_Score']
    corr_labels = ['浓度(n)', '形成能(JV)', '凸包能(JV)', '带隙(JV)', '带隙(MP)',
                  '体模量(JV)', '带隙降幅(MP)', '综合得分']

    corr_matrix = df_doped[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
               xticklabels=corr_labels, yticklabels=corr_labels, ax=ax,
               vmin=-1, vmax=1, center=0, square=True)
    ax.set_title('各属性间的相关性热力图', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_correlation_analysis.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 10_correlation_analysis.png")

# ============================================================
# 图11: 雷达图对比
# ============================================================
def plot_radar_comparison():
    """Top 3方案雷达图对比"""
    # 使用n=10（最高浓度）的数据
    df_high = df_doped[df_doped['n_dopant'] == 10]

    categories = ['BG_reduction_MP', 'E_hull_stability', 'BulkMod_score', 'E_form_JV']
    category_labels = ['带隙降幅', '热力学稳定性', '体模量保持', '形成能稳定性']

    # 计算归一化得分
    def normalize(values, min_val, max_val):
        return (values - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    top3_data = df_high[df_high['dopant_element'].isin(['Ti', 'Nb', 'Mn'])].groupby('dopant_element').agg({
        'BG_reduction_MP': 'mean',
        'E_hull_stability': 'mean',
        'BulkMod_score': 'mean',
        'E_form_JV': 'mean'
    })

    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    for idx, (elem, row) in enumerate(top3_data.iterrows()):
        # 归一化
        values = [
            min(1, row['BG_reduction_MP'] / 40),  # 假设最大降幅40%
            row['E_hull_stability'],
            row['BulkMod_score'],
            min(1, max(0, -(row['E_form_JV'] + 2.07) / 0.02))  # 相对于原始值
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=elem, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_labels, fontsize=12)
    ax.set_ylim(0.7, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    ax.set_title('Top 3 方案多维度性能对比', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_radar_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 11_radar_comparison.png")

# ============================================================
# 生成统计分析报告
# ============================================================
def generate_statistics_report():
    """生成详细统计分析报告"""
    report = []
    report.append("=" * 70)
    report.append("ALIGNN 预测结果统计分析报告 (带隙降幅优先标准)")
    report.append("=" * 70)

    # 数据概览
    report.append("\n【一、数据概览】")
    report.append(f"  总构型数: {len(df_doped)}")
    report.append(f"  掺杂位点: Fe位 {len(df_doped[df_doped['dopant_type']=='Fe-site'])}个, "
                 f"Li位 {len(df_doped[df_doped['dopant_type']=='Li-site'])}个, "
                 f"P位 {len(df_doped[df_doped['dopant_type']=='P-site'])}个")
    report.append(f"  掺杂元素: {df_doped['dopant_element'].nunique()}种")
    report.append(f"  浓度范围: x = 0.174% ~ 1.736%")

    # 原始LFP基准
    report.append("\n【二、原始LFP基准值】")
    report.append(f"  MP带隙: {REF['Bandgap_MP']} eV")
    report.append(f"  JARVIS带隙: {REF['Bandgap_JV']} eV")
    report.append(f"  凸包能: {REF['E_hull_JV']} eV")
    report.append(f"  形成能: {REF['E_form_JV']} eV/atom")

    # 带隙降幅统计
    report.append("\n【三、带隙降幅统计】")
    report.append(f"  MP带隙降幅范围: {df_doped['BG_reduction_MP'].min():.2f}% ~ {df_doped['BG_reduction_MP'].max():.2f}%")
    report.append(f"  MP带隙降幅均值: {df_doped['BG_reduction_MP'].mean():.2f}%")
    report.append(f"  JARVIS带隙降幅范围: {df_doped['BG_reduction_JV'].min():.2f}% ~ {df_doped['BG_reduction_JV'].max():.2f}%")

    # 按位点统计
    report.append("\n【四、按掺杂位点统计】")
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        report.append(f"\n  {site}:")
        report.append(f"    构型数: {len(subset)}")
        report.append(f"    元素: {', '.join(subset['dopant_element'].unique())}")
        report.append(f"    MP带隙降幅均值: {subset['BG_reduction_MP'].mean():.2f}%")
        report.append(f"    E_hull变化均值: {subset['E_hull_JV'].mean() - REF['E_hull_JV']:+.4f} eV")

    # Fe位Top 5元素（按带隙降幅）
    report.append("\n【五、Fe位元素带隙降幅排名（n=10时）】")
    df_fe_high = df_doped[(df_doped['dopant_type'] == 'Fe-site') & (df_doped['n_dopant'] == 10)]
    fe_ranking = df_fe_high.groupby('dopant_element').agg({
        'Bandgap_MP': 'mean',
        'BG_reduction_MP': 'mean',
        'E_hull_JV': 'mean'
    }).sort_values('BG_reduction_MP', ascending=False)

    report.append(f"  {'排名':<4} {'元素':<6} {'MP带隙(eV)':<14} {'带隙降幅(%)':<12} {'E_hull(eV)':<12}")
    report.append("  " + "-" * 50)
    for i, (elem, row) in enumerate(fe_ranking.head(5).iterrows(), 1):
        report.append(f"  {i:<4} {elem:<6} {row['Bandgap_MP']:<14.3f} {row['BG_reduction_MP']:<12.2f} {row['E_hull_JV']:<12.4f}")

    # Top 3方案详情
    report.append("\n【六、Top 3 最优方案（按综合得分）】")
    top3_elements = ['Ti', 'Nb', 'Mn']
    for i, elem in enumerate(top3_elements, 1):
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['n_dopant'] == 10)]
        row = subset.iloc[0]
        report.append(f"\n  D-{i}: {elem}掺杂 (Fe位, x=1.736%)")
        report.append(f"    MP带隙: {row['Bandgap_MP']:.3f} eV (降幅: {row['BG_reduction_MP']:.2f}%)")
        report.append(f"    E_hull: {row['E_hull_JV']:.4f} eV (变化: +{row['E_hull_JV']-REF['E_hull_JV']:.4f} eV)")
        report.append(f"    综合得分: {row['Comprehensive_Score']:.3f}")

    # P位/Li位排除依据
    report.append("\n【七、P位和Li位掺杂排除依据】")
    report.append("  P位掺杂（Si/S）:")
    p_subset = df_doped[df_doped['dopant_type'] == 'P-site']
    report.append(f"    MP带隙最大降幅: {p_subset['BG_reduction_MP'].max():.2f}%")
    report.append(f"    结论: 带隙调控效果极弱（<1%），已明确排除")

    report.append("\n  Li位掺杂（Mg/Al/Na/Ti/W）:")
    li_subset = df_doped[df_doped['dopant_type'] == 'Li-site']
    report.append(f"    MP带隙最大降幅: {li_subset['BG_reduction_MP'].max():.2f}%")
    report.append(f"    结论: 带隙调控效果较弱（<3%），不作为首选方案")

    # 浓度效应
    report.append("\n【八、浓度效应分析】")
    report.append("  规律: 浓度越高，带隙降幅越大，但热力学稳定性下降")
    report.append(f"  Ti掺杂: n=1时带隙降幅={df_doped[(df_doped['dopant_element']=='Ti')&(df_doped['n_dopant']==1)]['BG_reduction_MP'].mean():.1f}%, "
                 f"n=10时带隙降幅={df_doped[(df_doped['dopant_element']=='Ti')&(df_doped['n_dopant']==10)]['BG_reduction_MP'].mean():.1f}%")

    report_text = '\n'.join(report)
    print(report_text)

    with open(OUTPUT_DIR / 'statistics_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✓ 已保存: statistics_report.txt")
    return report_text

# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("\n开始生成可视化图表（带隙降幅优先标准）...\n")

    plot_bandgap_reduction_distribution()
    plot_concentration_bandgap_trend()
    plot_top10_comprehensive()
    plot_fe_site_bandgap_comparison()
    plot_thermodynamic_onset()
    plot_top3_detailed()
    plot_p_li_exclusion()
    plot_thermodynamic_boundary()
    plot_final_recommendation()
    plot_correlation_analysis()
    plot_radar_comparison()
    generate_statistics_report()

    print("\n" + "=" * 70)
    print(f"可视化图表已保存至: {OUTPUT_DIR}")
    print("=" * 70)
    print("\n生成的图表列表:")
    print("  1. 01_bandgap_reduction_distribution.png - 带隙降幅分布直方图")
    print("  2. 02_concentration_bandgap_trend.png    - 浓度-带隙降幅趋势图")
    print("  3. 03_top10_comprehensive.png            - 综合得分Top 10方案")
    print("  4. 04_fe_site_bandgap_comparison.png     - Fe位元素带隙降幅对比")
    print("  5. 05_thermodynamic_onset.png            - 热力学稳定起始浓度分析")
    print("  6. 06_top3_detailed.png                 - Top 3方案详细分析")
    print("  7. 07_p_li_exclusion.png                - P位和Li位排除依据")
    print("  8. 08_thermodynamic_boundary.png        - 热力学边界分析")
    print("  9. 09_final_recommendation.png          - 最终推荐方案汇总")
    print(" 10. 10_correlation_analysis.png          - 属性相关性分析")
    print(" 11. 11_radar_comparison.png             - Top 3雷达图对比")
    print(" 12. statistics_report.txt                - 统计分析报告")
