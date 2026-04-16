#!/usr/bin/env python3
"""
ALIGNN 预测结果可视化分析脚本
生成各类图表展示掺杂预测数据的规律
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

# 读取数据
DATA_DIR = Path('/Users/mxye/Myprojects/alignn/alignn/models')
OUTPUT_DIR = Path('/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/visualization')
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / 'all_predictions.csv')

# 简化列名
df = df.rename(columns={
    'jv_formation_energy_peratom_alignn': 'E_form_JV',
    'jv_ehull_alignn': 'E_hull_JV',
    'jv_optb88vdw_bandgap_alignn': 'Bandgap_JV',
    'jv_bulk_modulus_kv_alignn': 'BulkMod_JV',
    'jv_magmom_oszicar_alignn': 'MagMom_JV',
    'mp_e_form_alignn': 'E_form_MP',
    'mp_gappbe_alignn': 'Bandgap_MP'
})

# 原始LFP参考值
REF = {
    'E_form_JV': -2.06981,
    'E_hull_JV': 2.6381,
    'Bandgap_JV': 0.05774,
    'BulkMod_JV': 92.31,
    'MagMom_JV': 8.9124,
    'E_form_MP': -2.53219,
    'Bandgap_MP': 3.694
}

# 颜色配置
SITE_COLORS = {'Fe-site': '#E74C3C', 'Li-site': '#3498DB', 'P-site': '#2ECC71'}
ELEMENT_COLORS = {
    'Co': '#1abc9c', 'Cr': '#2ecc71', 'Cu': '#3498db', 'Er': '#9b59b6',
    'Mn': '#e67e22', 'Mo': '#f1c40f', 'Nb': '#e74c3c', 'Nd': '#34495e',
    'Ni': '#95a5a6', 'Ti': '#d35400', 'V': '#27ae60', 'Y': '#8e44ad',
    'Zn': '#2c3e50', 'Al': '#16a085', 'Mg': '#2980b9', 'Na': '#c0392b',
    'W': '#7f8c8d', 'Si': '#f39c12', 'S': '#1abc9c'
}

print("=" * 60)
print("ALIGNN 预测结果可视化分析")
print("=" * 60)
print(f"总构型数: {len(df)}")
print(f"掺杂位点: {df['dopant_type'].unique()}")
print(f"掺杂元素: {df['dopant_element'].unique()}")

# ============================================================
# 图1: 各属性分布直方图
# ============================================================
def plot_distribution_overview():
    """所有预测属性的分布直方图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    properties = ['E_form_JV', 'E_hull_JV', 'Bandgap_JV', 'BulkMod_JV', 'E_form_MP', 'Bandgap_MP']
    titles = ['形成能 (JARVIS)', '凸包能 (JARVIS)', '带隙 (JARVIS, eV)', '体模量 (JARVIS, GPa)', '形成能 (MP)', '带隙 (MP, eV)']

    for i, (prop, title) in enumerate(zip(properties, titles)):
        ax = axes[i]
        for site in ['Fe-site', 'Li-site', 'P-site']:
            subset = df[df['dopant_type'] == site][prop]
            ax.hist(subset, bins=30, alpha=0.6, label=site, color=SITE_COLORS[site])

        # 添加原始值参考线
        ax.axvline(REF[prop], color='black', linestyle='--', linewidth=2, label='原始LFP')

        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('构型数量', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)

    plt.suptitle('ALIGNN 预测属性分布概览', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_distribution_overview.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 01_distribution_overview.png")

# ============================================================
# 图2: 按掺杂位点分组的箱线图
# ============================================================
def plot_boxplot_by_site():
    """各属性按掺杂位点的箱线图对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    properties = ['E_form_JV', 'E_hull_JV', 'Bandgap_JV', 'BulkMod_JV', 'E_form_MP', 'Bandgap_MP']
    titles = ['形成能 (JV)', '凸包能 (JV)', '带隙 (JV)', '体模量 (JV)', '形成能 (MP)', '带隙 (MP)']

    for i, (prop, title) in enumerate(zip(properties, titles)):
        ax = axes[i]
        df.boxplot(column=prop, by='dopant_type', ax=ax, patch_artist=True)

        # 添加原始值参考线
        ax.axhline(REF[prop], color='red', linestyle='--', linewidth=2, label='原始LFP')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('掺杂位点', fontsize=10)
        ax.set_ylabel(prop, fontsize=10)

    plt.suptitle('各属性按掺杂位点分布对比（箱线图）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_boxplot_by_site.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 02_boxplot_by_site.png")

# ============================================================
# 图3: 各元素各属性的热力图
# ============================================================
def plot_heatmap_by_element():
    """按掺杂元素分组的平均属性热力图"""
    # 计算各元素各属性的平均值
    grouped = df.groupby(['dopant_type', 'dopant_element']).agg({
        'E_form_JV': 'mean',
        'E_hull_JV': 'mean',
        'Bandgap_JV': 'mean',
        'BulkMod_JV': 'mean'
    }).reset_index()

    # 凸包能热力图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, prop in enumerate(['E_form_JV', 'E_hull_JV', 'Bandgap_JV', 'BulkMod_JV']):
        ax = axes[idx // 2, idx % 2]
        pivot = grouped.pivot(index='dopant_type', columns='dopant_element', values=prop)

        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                   center=REF.get(prop, pivot.mean().mean()))
        ax.set_title(f'{prop} 按掺杂位点和元素', fontsize=12, fontweight='bold')

    plt.suptitle('各属性热力图（按位点和元素分组）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_heatmap_by_element.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 03_heatmap_by_element.png")

# ============================================================
# 图4: 浓度依赖性分析
# ============================================================
def plot_concentration_trend():
    """各属性随掺杂浓度的变化趋势"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    properties = ['E_form_JV', 'E_hull_JV', 'Bandgap_JV', 'BulkMod_JV', 'E_form_MP', 'Bandgap_MP']
    titles = ['形成能 (JV)', '凸包能 (JV)', '带隙 (JV)', '体模量 (JV)', '形成能 (MP)', '带隙 (MP)']

    # 选取代表性元素
    elements = ['Co', 'Mn', 'Mg', 'Si']

    for i, (prop, title) in enumerate(zip(properties, titles)):
        ax = axes[i]

        for elem in elements:
            subset = df[df['dopant_element'] == elem].groupby('n_dopant')[prop].agg(['mean', 'std'])
            ax.plot(subset.index, subset['mean'], 'o-', label=elem, markersize=6)
            ax.fill_between(subset.index, subset['mean'] - subset['std'],
                          subset['mean'] + subset['std'], alpha=0.2)

        ax.axhline(REF[prop], color='black', linestyle='--', linewidth=2, label='原始LFP')
        ax.set_xlabel('掺杂原子数 (n)', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('各属性随掺杂浓度的变化趋势（代表性元素）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_concentration_trend.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 04_concentration_trend.png")

# ============================================================
# 图5: E_form vs E_hull 散点图（Pareto分析）
# ============================================================
def plot_scatter_formation_ehull():
    """形成能 vs 凸包能的散点图"""
    fig, ax = plt.subplots(figsize=(12, 8))

    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df[df['dopant_type'] == site]
        ax.scatter(subset['E_form_JV'], subset['E_hull_JV'],
                  c=SITE_COLORS[site], alpha=0.5, s=30, label=site)

    # 原始LFP参考点
    ax.scatter(REF['E_form_JV'], REF['E_hull_JV'], c='black', s=200,
              marker='*', label='原始LFP', zorder=5)

    ax.set_xlabel('形成能 (eV/atom)', fontsize=12)
    ax.set_ylabel('凸包能 (eV)', fontsize=12)
    ax.set_title('形成能 vs 凸包能（热力学稳定性分析）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加注释
    ax.annotate('越靠近左下角\n热力学越稳定', xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_scatter_formation_ehull.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 05_scatter_formation_ehull.png")

# ============================================================
# 图6: 带隙工程分析
# ============================================================
def plot_bandgap_analysis():
    """带隙调控效果分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 6a: JARVIS带隙分布
    ax = axes[0]
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df[df['dopant_type'] == site]
        ax.scatter(subset['n_dopant'], subset['Bandgap_JV'],
                  c=SITE_COLORS[site], alpha=0.4, s=20, label=site)

    ax.axhline(REF['Bandgap_JV'], color='black', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('JARVIS带隙 (eV)', fontsize=12)
    ax.set_title('JARVIS带隙随浓度变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 6b: MP带隙分布
    ax = axes[1]
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df[df['dopant_type'] == site]
        ax.scatter(subset['n_dopant'], subset['Bandgap_MP'],
                  c=SITE_COLORS[site], alpha=0.4, s=20, label=site)

    ax.axhline(REF['Bandgap_MP'], color='black', linestyle='--', linewidth=2, label='原始LFP')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('MP带隙 (eV)', fontsize=12)
    ax.set_title('Materials Project带隙随浓度变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('带隙工程分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_bandgap_analysis.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 06_bandgap_analysis.png")

# ============================================================
# 图7: 各元素综合对比雷达图
# ============================================================
def plot_radar_comparison():
    """各掺杂元素的综合性能雷达图"""
    from math import pi

    # 计算各元素各属性的归一化得分
    elements_of_interest = ['Si', 'Mg', 'Al', 'Cr', 'Mn', 'Co', 'Nb', 'Zn']

    # 使用n=1（低浓度）数据
    df_low = df[df['n_dopant'] == 1].groupby('dopant_element').agg({
        'E_form_JV': 'mean',
        'E_hull_JV': 'mean',
        'Bandgap_JV': 'mean',
        'BulkMod_JV': 'mean'
    }).reset_index()

    # 计算相对于原始值的归一化得分（0-1，越接近1越好）
    scores = {}
    for _, row in df_low.iterrows():
        elem = row['dopant_element']
        if elem in elements_of_interest:
            scores[elem] = {
                'E_form': 1 - (row['E_form_JV'] - REF['E_form_JV']) / 0.02,  # 假设±0.02范围
                'E_hull': 1 - (row['E_hull_JV'] - REF['E_hull_JV']) / 0.05,
                'Bandgap': row['Bandgap_JV'] / REF['Bandgap_JV'],
                'BulkMod': row['BulkMod_JV'] / REF['BulkMod_JV']
            }

    # 创建雷达图
    categories = ['E_form', 'E_hull', 'Bandgap', 'BulkMod']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(scores)))

    for idx, (elem, vals) in enumerate(scores.items()):
        values = [max(0, min(1, vals[cat])) for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=elem, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['形成能', '凸包能', '带隙', '体模量'], fontsize=12)
    ax.set_ylim(0.8, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.set_title('各元素综合性能对比（雷达图）', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_radar_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 07_radar_comparison.png")

# ============================================================
# 图8: Fe位各元素详细对比
# ============================================================
def plot_fe_site_detailed():
    """Fe位各掺杂元素的详细对比"""
    fe_elements = ['Co', 'Cr', 'Cu', 'Er', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'Ti', 'V', 'Y', 'Zn']
    df_fe = df[df['dopant_type'] == 'Fe-site']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 8a: 形成能
    ax = axes[0, 0]
    means = df_fe.groupby('dopant_element')['E_form_JV'].mean().reindex(fe_elements)
    stds = df_fe.groupby('dopant_element')['E_form_JV'].std().reindex(fe_elements)
    bars = ax.bar(fe_elements, means, yerr=stds, capsize=3, color='#E74C3C', alpha=0.7)
    ax.axhline(REF['E_form_JV'], color='black', linestyle='--', linewidth=2)
    ax.set_title('形成能 (Fe位)', fontsize=12, fontweight='bold')
    ax.set_ylabel('E_form (eV/atom)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    # 8b: 凸包能
    ax = axes[0, 1]
    means = df_fe.groupby('dopant_element')['E_hull_JV'].mean().reindex(fe_elements)
    stds = df_fe.groupby('dopant_element')['E_hull_JV'].std().reindex(fe_elements)
    ax.bar(fe_elements, means, yerr=stds, capsize=3, color='#E74C3C', alpha=0.7)
    ax.axhline(REF['E_hull_JV'], color='black', linestyle='--', linewidth=2)
    ax.set_title('凸包能 (Fe位)', fontsize=12, fontweight='bold')
    ax.set_ylabel('E_hull (eV)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    # 8c: 带隙
    ax = axes[1, 0]
    means = df_fe.groupby('dopant_element')['Bandgap_JV'].mean().reindex(fe_elements)
    stds = df_fe.groupby('dopant_element')['Bandgap_JV'].std().reindex(fe_elements)
    ax.bar(fe_elements, means, yerr=stds, capsize=3, color='#E74C3C', alpha=0.7)
    ax.axhline(REF['Bandgap_JV'], color='black', linestyle='--', linewidth=2)
    ax.set_title('JARVIS带隙 (Fe位)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bandgap (eV)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    # 8d: 体模量
    ax = axes[1, 1]
    means = df_fe.groupby('dopant_element')['BulkMod_JV'].mean().reindex(fe_elements)
    stds = df_fe.groupby('dopant_element')['BulkMod_JV'].std().reindex(fe_elements)
    ax.bar(fe_elements, means, yerr=stds, capsize=3, color='#E74C3C', alpha=0.7)
    ax.axhline(REF['BulkMod_JV'], color='black', linestyle='--', linewidth=2)
    ax.set_title('体模量 (Fe位)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BulkMod (GPa)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Fe位13种掺杂元素详细对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_fe_site_detailed.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 08_fe_site_detailed.png")

# ============================================================
# 图9: Top 10 候选方案可视化
# ============================================================
def plot_top10_candidates():
    """Top 10最优候选方案可视化"""
    # 计算综合得分
    df['Score'] = (
        0.30 * (1 - (df['E_hull_JV'] - REF['E_hull_JV']) / 0.1) +
        0.25 * (1 - abs(df['E_form_JV'] - REF['E_form_JV']) / 0.02) +
        0.20 * (df['Bandgap_JV'] / REF['Bandgap_JV']) +
        0.15 * (df['BulkMod_JV'] / REF['BulkMod_JV']) +
        0.10 * (1 - abs(df['Bandgap_MP'] - REF['Bandgap_MP']) / 0.2)
    )

    # 获取Top 10
    df['n_dopant'] = df['n_dopant'].astype(float)
    top10 = df[df['n_dopant'] == 1].nlargest(10, 'Score')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 9a: 综合得分柱状图
    ax = axes[0]
    labels = [f"{row['dopant_element']} ({row['dopant_type'].replace('-site','')})"
             for _, row in top10.iterrows()]
    colors = [SITE_COLORS[t] for t in top10['dopant_type']]
    bars = ax.barh(labels[::-1], top10['Score'].values[::-1], color=colors[::-1])
    ax.set_xlabel('综合得分', fontsize=12)
    ax.set_title('Top 10 最优掺杂方案', fontsize=12, fontweight='bold')
    ax.set_xlim(0.7, 1.0)

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{bar.get_width():.3f}', va='center', fontsize=10)

    # 9b: 雷达图
    ax = axes[1]
    from math import pi
    categories = ['E_hull', 'E_form', 'Bandgap', 'BulkMod']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    for idx, (_, row) in enumerate(top10.head(5).iterrows()):
        elem = row['dopant_element']
        site = row['dopant_type'].replace('-site', '')
        label = f"{elem}@{site}"

        values = [
            1 - (row['E_hull_JV'] - REF['E_hull_JV']) / 0.1,
            1 - abs(row['E_form_JV'] - REF['E_form_JV']) / 0.02,
            row['Bandgap_JV'] / REF['Bandgap_JV'],
            row['BulkMod_JV'] / REF['BulkMod_JV']
        ]
        values = [max(0, min(1.5, v)) for v in values]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['凸包能', '形成能', '带隙', '体模量'], fontsize=10)
    ax.set_ylim(0.8, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Top 5 方案雷达图', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_top10_candidates.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 09_top10_candidates.png")

# ============================================================
# 图10: 相关性热力图
# ============================================================
def plot_correlation_heatmap():
    """各属性间的相关性热力图"""
    corr_cols = ['E_form_JV', 'E_hull_JV', 'Bandgap_JV', 'BulkMod_JV', 'MagMom_JV', 'E_form_MP', 'Bandgap_MP']
    corr_labels = ['形成能(JV)', '凸包能(JV)', '带隙(JV)', '体模量(JV)', '磁矩(JV)', '形成能(MP)', '带隙(MP)']

    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
               xticklabels=corr_labels, yticklabels=corr_labels, ax=ax,
               vmin=-1, vmax=1, center=0)
    ax.set_title('各预测属性间的相关性热力图', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 10_correlation_heatmap.png")

# ============================================================
# 生成统计摘要
# ============================================================
def generate_summary():
    """生成统计分析摘要"""
    summary = []

    summary.append("=" * 60)
    summary.append("ALIGNN 预测结果统计分析摘要")
    summary.append("=" * 60)

    # 按位点统计
    summary.append("\n【按掺杂位点统计】")
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df[df['dopant_type'] == site]
        summary.append(f"\n{site}:")
        summary.append(f"  构型数: {len(subset)}")
        summary.append(f"  元素: {', '.join(subset['dopant_element'].unique())}")
        summary.append(f"  形成能范围: {subset['E_form_JV'].min():.4f} ~ {subset['E_form_JV'].max():.4f}")
        summary.append(f"  凸包能范围: {subset['E_hull_JV'].min():.4f} ~ {subset['E_hull_JV'].max():.4f}")

    # 按元素统计Top 5
    summary.append("\n\n【E_hull最低的5种元素（热力学最稳定）】")
    elem_stats = df.groupby('dopant_element').agg({
        'E_hull_JV': 'mean',
        'E_form_JV': 'mean',
        'Bandgap_JV': 'mean'
    }).sort_values('E_hull_JV')

    for i, (elem, row) in enumerate(elem_stats.head(5).iterrows()):
        summary.append(f"  {i+1}. {elem}: E_hull={row['E_hull_JV']:.4f}, E_form={row['E_form_JV']:.4f}")

    # 浓度效应统计
    summary.append("\n\n【浓度效应（n=1 vs n=10对比）】")
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df[df['dopant_type'] == site]
        n1 = subset[subset['n_dopant'] == 1]['E_hull_JV'].mean()
        n10 = subset[subset['n_dopant'] == 10]['E_hull_JV'].mean()
        summary.append(f"  {site}: E_hull变化 = {n1:.4f} → {n10:.4f} (Δ={n10-n1:+.4f})")

    summary_text = '\n'.join(summary)
    print(summary_text)

    with open(OUTPUT_DIR / 'summary_statistics.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\n✓ 已保存: summary_statistics.txt")
    return summary_text

# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("\n开始生成可视化图表...\n")

    plot_distribution_overview()
    plot_boxplot_by_site()
    plot_heatmap_by_element()
    plot_concentration_trend()
    plot_scatter_formation_ehull()
    plot_bandgap_analysis()
    plot_radar_comparison()
    plot_fe_site_detailed()
    plot_top10_candidates()
    plot_correlation_heatmap()
    generate_summary()

    print("\n" + "=" * 60)
    print(f"可视化图表已保存至: {OUTPUT_DIR}")
    print("=" * 60)
    print("\n生成的图表列表:")
    print("  1. 01_distribution_overview.png    - 各属性分布直方图")
    print("  2. 02_boxplot_by_site.png         - 按位点分组的箱线图")
    print("  3. 03_heatmap_by_element.png      - 元素-位点热力图")
    print("  4. 04_concentration_trend.png     - 浓度依赖性趋势")
    print("  5. 05_scatter_formation_ehull.png - 形成能vs凸包能散点图")
    print("  6. 06_bandgap_analysis.png        - 带隙工程分析")
    print("  7. 07_radar_comparison.png        - 综合性能雷达图")
    print("  8. 08_fe_site_detailed.png        - Fe位元素详细对比")
    print("  9. 09_top10_candidates.png        - Top10最优方案")
    print(" 10. 10_correlation_heatmap.png     - 相关性热力图")
    print(" 11. summary_statistics.txt        - 统计分析摘要")
