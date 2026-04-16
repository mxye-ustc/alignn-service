#!/usr/bin/env python3
"""
ALIGNN 预测结果可视化分析脚本（优化版）
基于MP数据库为主要参考的设计

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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 路径配置
DATA_DIR = Path('/Users/mxye/Myprojects/alignn/alignn/models')
OUTPUT_DIR = Path('/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/visualization_v3')
OUTPUT_DIR.mkdir(exist_ok=True)

# 读取数据
df = pd.read_csv(DATA_DIR / 'all_predictions.csv')

# 简化列名
df = df.rename(columns={
    'jv_formation_energy_peratom_alignn': 'E_form_JV',
    'jv_ehull_alignn': 'E_hull_JV',
    'jv_optb88vdw_bandgap_alignn': 'Bandgap_JV',
    'jv_bulk_modulus_kv_alignn': 'BulkMod_JV',
    'mp_e_form_alignn': 'E_form_MP',
    'mp_gappbe_alignn': 'Bandgap_MP'
})

# 原始LFP参考值
REF = {
    'E_form_JV': -2.06981,
    'E_form_MP': -2.53219,
    'E_hull_JV': 2.6381,
    'Bandgap_JV': 0.05774,
    'Bandgap_MP': 3.694,
    'BulkMod_JV': 92.31
}

# 排除原始LFP，计算各项指标
df_doped = df[df['dopant_element'] != 'none'].copy()
df_doped['BG_reduction_MP'] = (REF['Bandgap_MP'] - df_doped['Bandgap_MP']) / REF['Bandgap_MP'] * 100
df_doped['E_hull_change'] = df_doped['E_hull_JV'] - REF['E_hull_JV']
df_doped['E_form_change_MP'] = df_doped['E_form_MP'] - REF['E_form_MP']

# 计算综合得分（带隙50% + 热力学30% + 体模量20%）
df_doped['BG_score'] = df_doped['BG_reduction_MP'] / 40
df_doped['E_hull_stability'] = 1 - df_doped['E_hull_change'] / 0.1
df_doped['BulkMod_score'] = df_doped['BulkMod_JV'] / REF['BulkMod_JV']
df_doped['Comprehensive_Score'] = (
    0.50 * df_doped['BG_score'].clip(0, 1) +
    0.30 * df_doped['E_hull_stability'].clip(0, 1.2) +
    0.20 * df_doped['BulkMod_score'].clip(0.8, 1.2)
)

# 按元素+浓度分组（用于图表）
grouped = df_doped.groupby(['dopant_element', 'n_dopant']).agg({
    'Comprehensive_Score': 'mean',
    'BG_reduction_MP': 'mean',
    'E_hull_change': 'mean',
    'E_form_change_MP': 'mean',
    'Bandgap_MP': 'mean'
}).reset_index()

# 颜色配置
SITE_COLORS = {'Fe-site': '#E74C3C', 'Li-site': '#3498DB', 'P-site': '#2ECC71'}
ELEM_COLORS = {
    'Al': '#E74C3C', 'Co': '#3498DB', 'Cr': '#2ECC71', 'Cu': '#9B59B6',
    'Er': '#F39C12', 'Mg': '#1ABC9C', 'Mn': '#E91E63', 'Mo': '#00BCD4',
    'Na': '#FF5722', 'Nb': '#795548', 'Nd': '#607D8B', 'Ni': '#8BC34A',
    'S': '#FF9800', 'Si': '#9C27B0', 'Ti': '#03A9F4', 'V': '#CDDC39',
    'W': '#009688', 'Y': '#3F51B5', 'Zn': '#F44336'
}

print("=" * 70)
print("生成优化版可视化图表（MP为主要参考）")
print("=" * 70)

# ============================================================
# 图1: 带隙调控效果（MP为主，JARVIS为辅）
# ============================================================
def plot_bandgap_effect():
    """带隙调控效果对比：MP主图 + JARVIS辅助"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：MP带隙降幅分布（主图）
    ax = axes[0]
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        ax.hist(subset['BG_reduction_MP'], bins=25, alpha=0.6, label=site, color=SITE_COLORS[site])

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_xlabel('MP带隙降幅 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('MP带隙降幅分布（★ 主要参考）', fontsize=14, fontweight='bold')
    ax.legend()

    # 添加说明文字
    ax.text(0.02, 0.98, 'MP带隙=3.694 eV\n与实验值一致', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 右图：JARVIS带隙分布（辅助）
    ax = axes[1]
    df_doped['BG_reduction_JV'] = (REF['Bandgap_JV'] - df_doped['Bandgap_JV']) / REF['Bandgap_JV'] * 100
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        ax.hist(subset['BG_reduction_JV'], bins=25, alpha=0.6, label=site, color=SITE_COLORS[site])

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_xlabel('JARVIS带隙降幅 (%)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('JARVIS带隙降幅分布（⚠️ 参考）', fontsize=14, fontweight='bold')
    ax.legend()

    # 添加警告说明
    ax.text(0.02, 0.98, 'JARVIS带隙=0.058 eV\n严重低估，仅作参考', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('带隙调控效果分布（负值表示带隙降低）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_bandgap_effect_mp_jv.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 01_bandgap_effect_mp_jv.png")

# ============================================================
# 图2: Top 10综合得分（MP为主要依据）
# ============================================================
def plot_top10_scores():
    """Top 10最优方案"""
    top10 = grouped.nlargest(10, 'Comprehensive_Score').reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：综合得分柱状图（排名从上到下）
    ax = axes[0]
    # 从上到下排列：第1名在最上面，第10名在最下面
    y_positions = range(len(top10) - 1, -1, -1)
    labels = [f"{row['dopant_element']}(n={int(row['n_dopant'])})" for _, row in top10.iterrows()]

    # 颜色：按元素着色
    colors = [ELEM_COLORS.get(row['dopant_element'], '#95A5A6') for _, row in top10.iterrows()]
    bars = ax.barh(list(y_positions), top10['Comprehensive_Score'].values, color=colors)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels)  # 正序标签
    ax.set_xlabel('综合得分', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 最优掺杂方案\n(带隙50% + 热力学30% + 体模量20%)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)

    # 添加数值标签
    for bar, score in zip(bars, top10['Comprehensive_Score'].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{score:.3f}', va='center', fontsize=10)

    # 添加图例
    legend_elements = [Patch(facecolor=ELEM_COLORS[e], label=e) for e in ['Cu', 'Ti', 'Mn']]
    ax.legend(handles=legend_elements, loc='lower right')

    # 右图：带隙降幅 vs 热力学稳定性
    ax = axes[1]

    # 图例元素：全部19种掺杂元素 + 原始LFP（用Line2D支持marker）
    all_elems = sorted(grouped['dopant_element'].unique())
    legend_elements = [Patch(facecolor=ELEM_COLORS.get(e, '#95A5A6'), label=e) for e in all_elems]
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                   markersize=12, label='原始LFP ★'))

    # 按元素分组绘制（不同颜色）
    for elem in all_elems:
        elem_data = grouped[grouped['dopant_element'] == elem]
        is_top10 = elem_data['Comprehensive_Score'].isin(top10['Comprehensive_Score'])
        color = ELEM_COLORS.get(elem, '#95A5A6')
        # 非Top10用小点，Top10用大点
        ax.scatter(elem_data[~is_top10]['E_hull_change'], elem_data[~is_top10]['BG_reduction_MP'],
                  s=25, c=color, alpha=0.6)
        ax.scatter(elem_data[is_top10]['E_hull_change'], elem_data[is_top10]['BG_reduction_MP'],
                  s=150, c=color, edgecolors='black', linewidth=1.5, zorder=5)

    # 原始LiFePO4参考点（星星）
    ax.scatter(0, 0, marker='*', s=400, c='red', edgecolors='black', linewidth=1.5, zorder=10)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('E_hull变化 (eV)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'带隙降幅 vs 热力学稳定性\n(全部{len(grouped)}个方案，Top 10高亮★)', fontsize=14, fontweight='bold')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_top10_scores.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 02_top10_scores.png")

# ============================================================
# 图3: 浓度效应曲线（MP为主）
# ============================================================
def plot_concentration_effect():
    """各元素带隙降幅随浓度变化（MP为主）"""
    key_elements = ['Cu', 'Ti', 'Mn', 'Nb', 'Co', 'Ni', 'Cr']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：MP带隙降幅
    ax = axes[0]
    for elem in key_elements:
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['BG_reduction_MP'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, markersize=6, linewidth=2,
               color=ELEM_COLORS.get(elem, None))

    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12, fontweight='bold')
    ax.set_title('MP带隙降幅随浓度变化（★ 主要参考）', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(25, color='red', linestyle=':', alpha=0.5, label='25%阈值')

    # 右图：热力学稳定性
    ax = axes[1]
    for elem in key_elements:
        subset = df_doped[df_doped['dopant_element'] == elem].groupby('n_dopant')['E_hull_change'].mean()
        ax.plot(subset.index, subset.values, 'o-', label=elem, markersize=6, linewidth=2,
               color=ELEM_COLORS.get(elem, None))

    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('E_hull变化 (eV)', fontsize=12)
    ax.set_title('热力学稳定性随浓度变化', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.05, color='red', linestyle=':', alpha=0.5, label='稳定性阈值')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('浓度依赖性效应分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_concentration_effect.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 03_concentration_effect.png")

# ============================================================
# 图4: 综合分析雷达图
# ============================================================
def plot_radar_analysis():
    """Top 4方案多维度雷达图"""
    from math import pi

    top4 = ['Cu', 'Ti', 'Mn', 'Nb']
    n_dopant = 10

    # 获取数据
    data = {}
    for elem in top4:
        subset = grouped[(grouped['dopant_element'] == elem) & (grouped['n_dopant'] == n_dopant)]
        if len(subset) > 0:
            row = subset.iloc[0]
            data[elem] = {
                '带隙降幅': row['BG_reduction_MP'] / 40,  # 归一化
                '热力学稳定性': 1 - row['E_hull_change'] / 0.1,
                '形成能稳定性': 1 - row['E_form_change_MP'] / 0.01,
                '综合得分': row['Comprehensive_Score']
            }

    # 雷达图
    categories = ['带隙降幅', '热力学稳定性', '形成能稳定性', '综合得分']
    N = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    for elem in top4:
        if elem in data:
            values = [data[elem][cat] for cat in categories]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=elem, color=ELEM_COLORS[elem])
            ax.fill(angles, values, alpha=0.15, color=ELEM_COLORS[elem])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Top 4 方案多维度性能对比\n★ 基于MP数据库', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_radar_analysis.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 04_radar_analysis.png")

# ============================================================
# 图5: 稳定性与形成能分析
# ============================================================
def plot_stability_analysis():
    """热力学稳定性与形成能分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：E_hull变化
    ax = axes[0]
    fe_elements = ['Ti', 'Nb', 'Mn', 'Cu', 'Co', 'Ni', 'Cr', 'V', 'Mo', 'Zn']
    ehull_data = []

    for elem in fe_elements:
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['n_dopant'] == 10)]
        if len(subset) > 0:
            ehull_data.append({
                'element': elem,
                'mean': subset['E_hull_change'].mean(),
                'std': subset['E_hull_change'].std()
            })

    ehull_df = pd.DataFrame(ehull_data)
    colors = [ELEM_COLORS.get(e, '#95A5A6') for e in ehull_df['element']]
    bars = ax.bar(ehull_df['element'], ehull_df['mean'], color=colors, alpha=0.7)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='稳定性阈值(0.05 eV)')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('掺杂元素', fontsize=12)
    ax.set_ylabel('E_hull变化 (eV)', fontsize=12)
    ax.set_title('Fe位元素 E_hull变化（n=10）', fontsize=14, fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, val in zip(bars, ehull_df['mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 右图：形成能变化（MP）
    ax = axes[1]
    eform_data = []
    for elem in fe_elements:
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['n_dopant'] == 10)]
        if len(subset) > 0:
            eform_data.append({
                'element': elem,
                'mean': subset['E_form_change_MP'].mean()
            })

    eform_df = pd.DataFrame(eform_data)
    colors = [ELEM_COLORS.get(e, '#95A5A6') for e in eform_df['element']]
    bars = ax.bar(eform_df['element'], eform_df['mean'], color=colors, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('掺杂元素', fontsize=12)
    ax.set_ylabel('MP形成能变化 (eV/atom)', fontsize=12, fontweight='bold')
    ax.set_title('MP形成能变化（n=10，★ 主要参考）', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, val in zip(bars, eform_df['mean']):
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.0005 if val >= 0 else -0.0005),
               f'{val:.4f}', ha='center', va=va, fontsize=9)

    plt.suptitle('热力学稳定性与形成能分析（★ 基于MP数据库）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_stability_analysis.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 05_stability_analysis.png")

# ============================================================
# 图6: 最终推荐方案对比
# ============================================================
def plot_final_comparison():
    """Top 4方案最终对比"""
    top4_data = [
        {'方案': 'D-1', '元素': 'Cu', '带隙降幅': 33.76, 'E_hull变化': 0.044, '形成能变化': 0.0069, '综合得分': 0.790},
        {'方案': 'D-2', '元素': 'Ti', '带隙降幅': 26.64, 'E_hull变化': 0.029, '形成能变化': -0.0026, '综合得分': 0.745},
        {'方案': 'D-3', '元素': 'Mn', '带隙降幅': 28.90, 'E_hull变化': 0.058, '形成能变化': -0.0001, '综合得分': 0.687},
        {'方案': 'D-4', '元素': 'Nb', '带隙降幅': 29.77, 'E_hull变化': 0.070, '形成能变化': -0.0012, '综合得分': 0.662},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 指标1：带隙降幅
    ax = axes[0, 0]
    elems = [d['元素'] for d in top4_data]
    values = [d['带隙降幅'] for d in top4_data]
    colors = [ELEM_COLORS[e] for e in elems]
    bars = ax.bar(elems, values, color=colors, alpha=0.8)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12, fontweight='bold')
    ax.set_title('① 带隙降幅（★ MP主要参考）', fontsize=12)
    ax.set_ylim(0, 40)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # 指标2：热力学稳定性
    ax = axes[0, 1]
    values = [d['E_hull变化'] for d in top4_data]
    bars = ax.bar(elems, values, color=colors, alpha=0.8)
    ax.axhline(0.05, color='red', linestyle='--', label='阈值')
    ax.set_ylabel('E_hull变化 (eV)', fontsize=12)
    ax.set_title('② 热力学稳定性', fontsize=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{val:.3f}', ha='center', fontsize=11)

    # 指标3：形成能
    ax = axes[1, 0]
    values = [d['形成能变化'] for d in top4_data]
    bars = ax.bar(elems, values, color=colors, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='-')
    ax.set_ylabel('MP形成能变化 (eV/atom)', fontsize=12, fontweight='bold')
    ax.set_title('③ 形成能变化（★ MP主要参考）', fontsize=12)
    for bar, val in zip(bars, values):
        va = 'bottom' if val >= 0 else 'top'
        offset = 0.0005 if val >= 0 else -0.0005
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
               f'{val:.4f}', ha='center', va=va, fontsize=10)

    # 指标4：综合得分
    ax = axes[1, 1]
    values = [d['综合得分'] for d in top4_data]
    bars = ax.bar(elems, values, color=colors, alpha=0.8)
    ax.set_ylabel('综合得分', fontsize=12)
    ax.set_title('④ 综合得分', fontsize=12)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('Top 4 最优方案多维度对比（★ 基于MP数据库）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_final_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 06_final_comparison.png")

# ============================================================
# 执行所有绘图函数
# ============================================================
if __name__ == '__main__':
    plot_bandgap_effect()
    plot_top10_scores()
    plot_concentration_effect()
    plot_radar_analysis()
    plot_stability_analysis()
    plot_final_comparison()

    print("\n" + "=" * 70)
    print("所有图表生成完成！")
    print("=" * 70)
