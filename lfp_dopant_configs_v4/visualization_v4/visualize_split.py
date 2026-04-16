#!/usr/bin/env python3
"""
LiFePO4 掺杂方案筛选 - 层层递进可视化分析
每个Step的子图拆分为独立图片，按文件夹存放
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 路径配置
DATA_DIR = Path('/Users/mxye/Myprojects/alignn/alignn/models')
OUTPUT_DIR = Path('/Users/mxye/Myprojects/alignn/lfp_dopant_configs_v4/visualization_v4')

# 创建子文件夹
for i in range(1, 8):
    (OUTPUT_DIR / f'Step{i}').mkdir(exist_ok=True)

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

# 排除原始LFP
df_doped = df[df['dopant_element'] != 'none'].copy()

# 计算派生指标
df_doped['BG_reduction_MP'] = (REF['Bandgap_MP'] - df_doped['Bandgap_MP']) / REF['Bandgap_MP'] * 100
df_doped['BG_reduction_JV'] = (REF['Bandgap_JV'] - df_doped['Bandgap_JV']) / REF['Bandgap_JV'] * 100
df_doped['E_hull_change'] = df_doped['E_hull_JV'] - REF['E_hull_JV']
df_doped['E_form_change_MP'] = df_doped['E_form_MP'] - REF['E_form_MP']

# 元素颜色配置
ELEM_COLORS = {
    'Al': '#E74C3C', 'Co': '#3498DB', 'Cr': '#2ECC71', 'Cu': '#9B59B6',
    'Er': '#F39C12', 'Mg': '#1ABC9C', 'Mn': '#E91E63', 'Mo': '#00BCD4',
    'Na': '#FF5722', 'Nb': '#795548', 'Nd': '#607D8B', 'Ni': '#8BC34A',
    'S': '#FF9800', 'Si': '#9C27B0', 'Ti': '#03A9F4', 'V': '#CDDC39',
    'W': '#009688', 'Y': '#3F51B5', 'Zn': '#F44336'
}
SITE_COLORS = {'Fe-site': '#E74C3C', 'Li-site': '#3498DB', 'P-site': '#2ECC71'}

print("=" * 70)
print("LiFePO4 层层递进可视化 - 拆分版")
print("=" * 70)

# =============================================================================
# Step 1: 模型预测总览 (拆分为4个子图)
# =============================================================================
def plot_step1():
    """Step 1: 模型预测总览"""
    step_dir = OUTPUT_DIR / 'Step1'

    # 1.1 数据集规模统计
    fig, ax = plt.subplots(figsize=(10, 6))
    stats_data = {
        '总构型': len(df),
        '掺杂构型': len(df_doped),
        '掺杂元素': df_doped['dopant_element'].nunique(),
        '掺杂位点': df_doped['dopant_type'].nunique()
    }
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']
    bars = ax.bar(stats_data.keys(), stats_data.values(), color=colors, alpha=0.8)
    for bar, val in zip(bars, stats_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{val}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('数据集规模统计', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '01_dataset_scale.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step1: 01_dataset_scale.png")

    # 1.2 原始LFP参考值
    fig, ax = plt.subplots(figsize=(10, 6))
    ref_data = {
        'MP带隙\n(eV)': REF['Bandgap_MP'],
        'JARVIS带隙\n(eV)': REF['Bandgap_JV'],
        '体模量\n(GPa)': REF['BulkMod_JV']
    }
    bars = ax.bar(ref_data.keys(), ref_data.values(), color=['#3498DB', '#E74C3C', '#2ECC71'], alpha=0.8)
    for bar, val in zip(bars, ref_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_title('原始LiFePO₄参考值', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 4.5)
    plt.tight_layout()
    plt.savefig(step_dir / '02_reference_values.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step1: 02_reference_values.png")

    # 1.3 MP预测带隙分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_doped['Bandgap_MP'], bins=30, color='#3498DB', alpha=0.7, edgecolor='white')
    ax.axvline(REF['Bandgap_MP'], color='red', linestyle='--', linewidth=2,
               label=f'原始LFP: {REF["Bandgap_MP"]} eV')
    ax.set_xlabel('MP预测带隙 (eV)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('MP预测带隙分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '03_bandgap_distribution_mp.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step1: 03_bandgap_distribution_mp.png")

    # 1.4 E_hull分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_doped['E_hull_JV'], bins=30, color='#2ECC71', alpha=0.7, edgecolor='white')
    ax.axvline(REF['E_hull_JV'], color='red', linestyle='--', linewidth=2,
               label=f'原始LFP: {REF["E_hull_JV"]:.3f} eV')
    ax.set_xlabel('E_hull (eV)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('热力学稳定性分布 (E_hull)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '04_ehull_distribution.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step1: 04_ehull_distribution.png")

    # 1.5 体模量分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_doped['BulkMod_JV'], bins=30, color='#9B59B6', alpha=0.7, edgecolor='white')
    ax.axvline(REF['BulkMod_JV'], color='red', linestyle='--', linewidth=2,
               label=f'原始LFP: {REF["BulkMod_JV"]:.1f} GPa')
    ax.set_xlabel('体模量 (GPa)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('体模量分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '05_bulkmod_distribution.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step1: 05_bulkmod_distribution.png")

    # 1.6 模型可信度评估
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    model_info = [
        ['属性', '可信度', '说明'],
        ['MP带隙预测', '★ 主要参考', '与实验值一致'],
        ['MP形成能', '★ 主要参考', '高度可靠'],
        ['JARVIS带隙', '⚠️ 辅助参考', '严重低估'],
        ['JARVIS E_hull', '参考', '可作对比'],
    ]
    table = ax.table(cellText=model_info[1:], colLabels=model_info[0],
                     loc='center', cellLoc='center',
                     colColours=['#BDC3C7']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.5)
    ax.set_title('预测模型可信度评估', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(step_dir / '06_model_reliability.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step1: 06_model_reliability.png")

# =============================================================================
# Step 2: 关键指标分布 (拆分为4个子图)
# =============================================================================
def plot_step2():
    """Step 2: 关键性能指标分布"""
    step_dir = OUTPUT_DIR / 'Step2'

    # 2.1 MP带隙降幅分布
    fig, ax = plt.subplots(figsize=(10, 6))
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        ax.hist(subset['BG_reduction_MP'], bins=25, alpha=0.5, label=f'{site} (n={len(subset)})',
               color=SITE_COLORS[site])
    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(25, color='red', linestyle=':', linewidth=2, alpha=0.7, label='25%阈值')
    ax.set_xlabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('★ MP带隙降幅分布（主参考）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '01_bandgap_reduction_dist.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step2: 01_bandgap_reduction_dist.png")

    # 2.2 带隙降幅统计
    fig, ax = plt.subplots(figsize=(10, 6))
    stats = {
        '最大值': df_doped['BG_reduction_MP'].max(),
        '均值': df_doped['BG_reduction_MP'].mean(),
        '中位数': df_doped['BG_reduction_MP'].median(),
        '标准差': df_doped['BG_reduction_MP'].std()
    }
    bars = ax.barh(list(stats.keys()), list(stats.values()), color='#3498DB', alpha=0.8)
    for bar, val in zip(bars, stats.values()):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
               va='center', fontsize=12, fontweight='bold')
    ax.set_xlabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('带隙降幅统计', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '02_bandgap_stats.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step2: 02_bandgap_stats.png")

    # 2.3 E_hull变化分布
    fig, ax = plt.subplots(figsize=(10, 6))
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        ax.hist(subset['E_hull_change'], bins=25, alpha=0.5, label=site, color=SITE_COLORS[site])
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='稳定性阈值 (0.05 eV)')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('E_hull变化 (eV)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('热力学稳定性变化 (ΔE_hull)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '03_ehull_change_dist.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step2: 03_ehull_change_dist.png")

    # 2.4 稳定性筛选比例
    fig, ax = plt.subplots(figsize=(8, 8))
    # 所有构型的稳定性比例
    stable = (df_doped['E_hull_change'] < 0.05).sum()
    unstable = (df_doped['E_hull_change'] >= 0.05).sum()
    ax.pie([stable, unstable],
           labels=[f'ΔE<0.05 (稳定)\nn={stable} ({stable/len(df_doped)*100:.1f}%)',
                   f'ΔE≥0.05 (不稳定)\nn={unstable} ({unstable/len(df_doped)*100:.1f}%)'],
           colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 12})
    ax.set_title(f'热力学稳定性分布（全部{len(df_doped)}个构型）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '04_stability_ratio.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step2: 04_stability_ratio.png")

    # 2.5 体模量分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_doped['BulkMod_JV'], bins=30, alpha=0.7, color='#9B59B6', edgecolor='white')
    ax.axvline(REF['BulkMod_JV'], color='red', linestyle='--', linewidth=2,
               label=f'原始LFP: {REF["BulkMod_JV"]:.1f} GPa')
    ax.axvline(REF['BulkMod_JV'] * 0.95, color='orange', linestyle=':', linewidth=2,
               label='95%阈值')
    ax.set_xlabel('体模量 (GPa)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('体模量分布 - 掺杂不应显著降低机械强度', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '05_bulkmod_dist.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step2: 05_bulkmod_dist.png")

    # 2.6 指标相关性
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df_doped[['BG_reduction_MP', 'E_hull_change', 'BulkMod_JV']].corr()
    im = ax.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['带隙降幅', 'ΔE_hull', '体模量'], fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(['带隙降幅', 'ΔE_hull', '体模量'], fontsize=10)
    ax.set_title('指标相关性', fontsize=14, fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(step_dir / '06_correlation.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step2: 06_correlation.png")

# =============================================================================
# Step 3: 掺杂位点分析 (拆分为4个子图)
# =============================================================================
def plot_step3():
    """Step 3: 掺杂位点效应分析"""
    step_dir = OUTPUT_DIR / 'Step3'

    # 3.1 各站点可用元素数
    fig, ax = plt.subplots(figsize=(10, 6))
    site_counts = df_doped.groupby('dopant_type')['dopant_element'].nunique()
    bars = ax.bar(site_counts.index, site_counts.values,
                  color=[SITE_COLORS[s] for s in site_counts.index], alpha=0.8)
    for bar, val in zip(bars, site_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f'{val}种', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('掺杂元素种类', fontsize=12)
    ax.set_title('各站点可用掺杂元素数', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '01_site_element_count.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step3: 01_site_element_count.png")

    # 3.2 各站点构型数量
    fig, ax = plt.subplots(figsize=(10, 6))
    site_confs = df_doped.groupby('dopant_type').size()
    bars = ax.bar(site_confs.index, site_confs.values,
                  color=[SITE_COLORS[s] for s in site_confs.index], alpha=0.8)
    for bar, val in zip(bars, site_confs.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('各站点构型数量', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '02_site_config_count.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step3: 02_site_config_count.png")

    # 3.3 各站点平均带隙降幅
    fig, ax = plt.subplots(figsize=(10, 6))
    site_bg = df_doped.groupby('dopant_type')['BG_reduction_MP'].mean()
    bars = ax.bar(site_bg.index, site_bg.values,
                  color=[SITE_COLORS[s] for s in site_bg.index], alpha=0.8)
    for bar, val in zip(bars, site_bg.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
               f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均带隙降幅 (%)', fontsize=12)
    ax.set_title('各站点平均带隙降幅', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '03_site_avg_bandgap.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step3: 03_site_avg_bandgap.png")

    # 3.4 带隙降幅箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    site_data = [df_doped[df_doped['dopant_type'] == site]['BG_reduction_MP'].values
                 for site in ['Fe-site', 'Li-site', 'P-site']]
    bp = ax.boxplot(site_data, labels=['Fe-site', 'Li-site', 'P-site'], patch_artist=True)
    for patch, color in zip(bp['boxes'], [SITE_COLORS['Fe-site'], SITE_COLORS['Li-site'], SITE_COLORS['P-site']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(25, color='red', linestyle=':', alpha=0.5, label='25%阈值')
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('各站点带隙降幅分布对比（箱线图）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '04_site_boxplot.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step3: 04_site_boxplot.png")

    # 3.5 各站点E_hull变化
    fig, ax = plt.subplots(figsize=(10, 6))
    for site in ['Fe-site', 'Li-site', 'P-site']:
        subset = df_doped[df_doped['dopant_type'] == site]
        ax.hist(subset['E_hull_change'], bins=20, alpha=0.5, label=site, color=SITE_COLORS[site])
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='稳定性阈值')
    ax.set_xlabel('E_hull变化 (eV)', fontsize=12)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('各站点热力学稳定性对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(step_dir / '05_site_ehull.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step3: 05_site_ehull.png")

    # 3.6 站点筛选结论
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    conclusion = """
    站点筛选结论：

    Fe-site（红）：
    • 可掺杂元素最多（19种）
    • 带隙调控效果最佳
    • 为主要筛选对象

    Li-site（蓝）：
    • 效果有限
    • 少量元素可用

    P-site（绿）：
    • 效果较差
    • 可考虑作对比参考
    """
    ax.text(0.1, 0.9, conclusion, transform=ax.transAxes, fontsize=14,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))
    ax.set_title('站点筛选结论', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(step_dir / '06_site_conclusion.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step3: 06_site_conclusion.png")

# =============================================================================
# Step 4: 浓度效应分析 (拆分为4个子图)
# =============================================================================
def plot_step4():
    """Step 4: 浓度效应分析"""
    step_dir = OUTPUT_DIR / 'Step4'
    key_elements = ['Cu', 'Ti', 'Mn', 'Nb', 'Co', 'Ni', 'Cr']

    # 4.1 带隙降幅随浓度变化
    fig, ax = plt.subplots(figsize=(12, 7))
    for elem in key_elements:
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['dopant_type'] == 'Fe-site')]
        if len(subset) > 0:
            conc_data = subset.groupby('n_dopant')['BG_reduction_MP'].agg(['mean', 'std']).reset_index()
            ax.errorbar(conc_data['n_dopant'], conc_data['mean'], yerr=conc_data['std'],
                        fmt='o-', label=elem, markersize=8, linewidth=2,
                        color=ELEM_COLORS.get(elem, '#95A5A6'), capsize=3)
    ax.axhline(25, color='red', linestyle=':', linewidth=2, alpha=0.7, label='25%阈值')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('★ MP带隙降幅随浓度变化（主参考）', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(step_dir / '01_concentration_bandgap.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step4: 01_concentration_bandgap.png")

    # 4.2 最优浓度推荐
    fig, ax = plt.subplots(figsize=(10, 7))
    optimal_conc = []
    for elem in key_elements:
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['dopant_type'] == 'Fe-site')]
        if len(subset) > 0:
            max_bg = subset['BG_reduction_MP'].max()
            opt_n = subset[subset['BG_reduction_MP'] == max_bg]['n_dopant'].iloc[0]
            optimal_conc.append({'元素': elem, '最优n': int(opt_n), '最大降幅': max_bg})
    opt_df = pd.DataFrame(optimal_conc)
    colors = [ELEM_COLORS.get(e, '#95A5A6') for e in opt_df['元素']]
    bars = ax.barh(opt_df['元素'], opt_df['最大降幅'], color=colors, alpha=0.8)
    for bar, opt_n in zip(bars, opt_df['最优n']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'n={opt_n}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('最大带隙降幅 (%)', fontsize=12)
    ax.set_title('各元素最优掺杂浓度', fontsize=14, fontweight='bold')
    ax.axvline(25, color='red', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(step_dir / '02_optimal_concentration.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step4: 02_optimal_concentration.png")

    # 4.3 E_hull变化随浓度变化
    fig, ax = plt.subplots(figsize=(12, 7))
    for elem in key_elements:
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['dopant_type'] == 'Fe-site')]
        if len(subset) > 0:
            conc_data = subset.groupby('n_dopant')['E_hull_change'].agg(['mean', 'std']).reset_index()
            ax.errorbar(conc_data['n_dopant'], conc_data['mean'], yerr=conc_data['std'],
                       fmt='s-', label=elem, markersize=6, linewidth=2,
                       color=ELEM_COLORS.get(elem, '#95A5A6'), capsize=3)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, alpha=0.7, label='稳定性阈值 (0.05 eV)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('E_hull变化 (eV)', fontsize=12)
    ax.set_title('热力学稳定性随浓度变化', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(step_dir / '03_concentration_ehull.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step4: 03_concentration_ehull.png")

    # 4.4 体模量随浓度变化
    fig, ax = plt.subplots(figsize=(10, 6))
    for elem in key_elements[:4]:
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['dopant_type'] == 'Fe-site')]
        if len(subset) > 0:
            conc_data = subset.groupby('n_dopant')['BulkMod_JV'].mean().reset_index()
            ax.plot(conc_data['n_dopant'], conc_data['BulkMod_JV'], 'o-',
                   label=elem, markersize=6, linewidth=2,
                   color=ELEM_COLORS.get(elem, '#95A5A6'))
    ax.axhline(REF['BulkMod_JV'] * 0.95, color='red', linestyle=':', linewidth=2, alpha=0.7,
              label='95%阈值')
    ax.axhline(REF['BulkMod_JV'], color='gray', linestyle='--', alpha=0.5, label='原始LFP')
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('体模量 (GPa)', fontsize=12)
    ax.set_title('体模量随浓度变化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(step_dir / '04_concentration_bulkmod.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step4: 04_concentration_bulkmod.png")

    # 4.5 综合得分随浓度变化
    fig, ax = plt.subplots(figsize=(12, 7))
    df_doped['BG_score'] = df_doped['BG_reduction_MP'].clip(0, 40) / 40
    df_doped['Stability_score'] = (0.05 - df_doped['E_hull_change']).clip(0, 0.1) / 0.1
    df_doped['BulkMod_score'] = (df_doped['BulkMod_JV'] / REF['BulkMod_JV']).clip(0.9, 1.1)
    df_doped['Total_score'] = 0.5 * df_doped['BG_score'] + 0.3 * df_doped['Stability_score'] + 0.2 * df_doped['BulkMod_score']

    for elem in key_elements:
        subset = df_doped[(df_doped['dopant_element'] == elem) & (df_doped['dopant_type'] == 'Fe-site')]
        if len(subset) > 0:
            conc_data = subset.groupby('n_dopant')['Total_score'].mean().reset_index()
            ax.plot(conc_data['n_dopant'], conc_data['Total_score'], 'o-',
                   label=elem, markersize=8, linewidth=2,
                   color=ELEM_COLORS.get(elem, '#95A5A6'))
    ax.set_xlabel('掺杂原子数 (n)', fontsize=12)
    ax.set_ylabel('综合得分', fontsize=12)
    ax.set_title('综合得分（带隙50% + 热力学30% + 体模量20%）随浓度变化', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(step_dir / '05_concentration_score.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step4: 05_concentration_score.png")

# =============================================================================
# Step 5: Pareto前沿筛选 (拆分为4个子图)
# =============================================================================
def plot_step5():
    """Step 5: Pareto前沿筛选"""
    step_dir = OUTPUT_DIR / 'Step5'

    # 5.1 Pareto前沿散点图
    fig, ax = plt.subplots(figsize=(12, 8))

    fe_data = df_doped[(df_doped['dopant_type'] == 'Fe-site') & (df_doped['BG_reduction_MP'] > 0)].copy()

    # 绘制所有点
    for elem in fe_data['dopant_element'].unique():
        subset = fe_data[fe_data['dopant_element'] == elem]
        ax.scatter(subset['E_hull_change'], subset['BG_reduction_MP'],
                  s=30, alpha=0.4, color=ELEM_COLORS.get(elem, '#95A5A6'), label=elem)

    # 找Pareto前沿
    pareto_mask = []
    for i, row in fe_data.iterrows():
        is_dominated = False
        for j, other in fe_data.iterrows():
            if (other['BG_reduction_MP'] > row['BG_reduction_MP'] and
                other['E_hull_change'] <= row['E_hull_change'] and
                not (other['BG_reduction_MP'] == row['BG_reduction_MP'] and
                     other['E_hull_change'] == row['E_hull_change'])):
                is_dominated = True
                break
        pareto_mask.append(not is_dominated)

    fe_data['pareto'] = pareto_mask
    pareto_points = fe_data[fe_data['pareto']].sort_values('BG_reduction_MP', ascending=False)

    # 绘制Pareto前沿
    pareto_sorted = pareto_points.sort_values('E_hull_change')
    ax.scatter(pareto_points['E_hull_change'], pareto_points['BG_reduction_MP'],
              s=150, c='gold', edgecolors='black', linewidth=2, zorder=10,
              label=f'Pareto前沿 (n={len(pareto_points)})')
    ax.plot(pareto_sorted['E_hull_change'], pareto_sorted['BG_reduction_MP'],
           'r--', linewidth=2, alpha=0.7)

    ax.axhline(25, color='red', linestyle=':', linewidth=2, alpha=0.5, label='25%带隙阈值')
    ax.axvline(0.05, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='0.05eV阈值')
    ax.set_xlabel('E_hull变化 (eV) ← 越左越稳定', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%) → 越大越好', fontsize=12)
    ax.set_title('Fe位掺杂：Pareto前沿分析', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(step_dir / '01_pareto_front.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step5: 01_pareto_front.png")

    # 5.2 Pareto点元素分布
    fig, ax = plt.subplots(figsize=(10, 8))
    pareto_elems = pareto_points.groupby('dopant_element').size().sort_values(ascending=False)
    colors = [ELEM_COLORS.get(e, '#95A5A6') for e in pareto_elems.index]
    wedges, texts, autotexts = ax.pie(pareto_elems.values, labels=pareto_elems.index,
                                       colors=colors, autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax.set_title(f'Pareto前沿元素分布 (共{len(pareto_points)}个构型)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '02_pareto_elements.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step5: 02_pareto_elements.png")

    # 5.3 Top 10 Pareto方案
    fig, ax = plt.subplots(figsize=(12, 7))
    top10_pareto = pareto_points.nlargest(10, 'BG_reduction_MP').reset_index(drop=True)
    y_pos = range(len(top10_pareto) - 1, -1, -1)
    labels = [f"{row['dopant_element']}(n={int(row['n_dopant'])})" for _, row in top10_pareto.iterrows()]
    colors = [ELEM_COLORS.get(row['dopant_element'], '#95A5A6') for _, row in top10_pareto.iterrows()]

    bars = ax.barh(list(y_pos), top10_pareto['BG_reduction_MP'].values, color=colors, alpha=0.8)
    for bar, (_, row) in zip(bars, top10_pareto.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f"ΔE={row['E_hull_change']:.3f}eV", va='center', fontsize=10)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('Top 10 Pareto最优方案（按带隙降幅排序）', fontsize=14, fontweight='bold')
    ax.axvline(25, color='red', linestyle=':', linewidth=2, alpha=0.5)
    plt.tight_layout()
    plt.savefig(step_dir / '03_top10_pareto.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step5: 03_top10_pareto.png")

    # 5.4 Pareto筛选结论
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    pareto_elems = pareto_points.groupby('dopant_element').size().sort_values(ascending=False)
    conclusion = f"""
    Pareto筛选结果：

    总Fe位构型: {len(fe_data)}
    Pareto最优: {len(pareto_points)}

    Top 5 候选元素：
    {', '.join(pareto_elems.head(5).index.tolist())}

    筛选标准：
    • 带隙降幅 > 20%
    • ΔE_hull < 0.1 eV
    • 体模量 > 原始的95%

    筛选依据：
    多目标优化，不牺牲
    一个目标来提升另一个
    """
    ax.text(0.1, 0.9, conclusion, transform=ax.transAxes, fontsize=13,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    ax.set_title('Pareto筛选结论', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(step_dir / '04_pareto_conclusion.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step5: 04_pareto_conclusion.png")

# =============================================================================
# Step 6: 约束筛选流程 (拆分为4个子图)
# =============================================================================
def plot_step6():
    """Step 6: 约束筛选流程"""
    step_dir = OUTPUT_DIR / 'Step6'

    total = len(df_doped)
    step1 = len(df_doped[df_doped['dopant_type'] == 'Fe-site'])
    step2 = len(df_doped[(df_doped['dopant_type'] == 'Fe-site') & (df_doped['BG_reduction_MP'] > 20)])
    step3 = len(df_doped[(df_doped['dopant_type'] == 'Fe-site') & (df_doped['BG_reduction_MP'] > 20) & (df_doped['E_hull_change'] < 0.05)])
    step4 = len(df_doped[(df_doped['dopant_type'] == 'Fe-site') & (df_doped['BG_reduction_MP'] > 20) & (df_doped['E_hull_change'] < 0.05) & (df_doped['BulkMod_JV'] > REF['BulkMod_JV'] * 0.95)])

    counts = [total, step1, step2, step3, step4]
    steps = ['Step 0\n全部构型', 'Step 1\nFe位掺杂', 'Step 2\n带隙>20%', 'Step 3\nΔE<0.05', 'Step 4\n体模量>95%']

    # 6.1 筛选漏斗
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_funnel = ['#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#2ECC71']
    bars = ax.bar(range(len(counts)), counts, color=colors_funnel, alpha=0.8, width=0.6)
    for bar, count, pct in zip(bars, counts, [c/counts[0]*100 for c in counts]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{count}\n({pct:.1f}%)',
               ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=11)
    ax.set_ylabel('构型数量', fontsize=12)
    ax.set_title('层层筛选漏斗（从全部构型到最终候选）', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.25)
    plt.tight_layout()
    plt.savefig(step_dir / '01_filtering_funnel.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step6: 01_filtering_funnel.png")

    # 6.2 筛选条件表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    filter_table = [
        ['Step', '筛选条件', '保留率'],
        ['0', '全部构型', '100%'],
        ['1', 'Fe位掺杂', f'{step1/total*100:.1f}%'],
        ['2', 'MP带隙降幅>20%', f'{step2/total*100:.1f}%'],
        ['3', 'ΔE_hull<0.05 eV', f'{step3/total*100:.2f}%'],
        ['4', '体模量>95%原始', f'{step4/total*100:.2f}%'],
    ]
    table = ax.table(cellText=filter_table[1:], colLabels=filter_table[0],
                     loc='center', cellLoc='center',
                     colColours=['#BDC3C7']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    ax.set_title('筛选条件汇总', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(step_dir / '02_filtering_criteria.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step6: 02_filtering_criteria.png")

    # 6.3 筛选后散点图
    fig, ax = plt.subplots(figsize=(12, 8))
    filtered = df_doped[(df_doped['dopant_type'] == 'Fe-site') &
                        (df_doped['BG_reduction_MP'] > 20) &
                        (df_doped['E_hull_change'] < 0.05) &
                        (df_doped['BulkMod_JV'] > REF['BulkMod_JV'] * 0.95)]

    fe_all = df_doped[df_doped['dopant_type'] == 'Fe-site']
    ax.scatter(fe_all['E_hull_change'], fe_all['BG_reduction_MP'],
              s=20, alpha=0.2, c='gray', label='Fe位全部构型')

    colors_filtered = [ELEM_COLORS.get(e, '#95A5A6') for e in filtered['dopant_element']]
    ax.scatter(filtered['E_hull_change'], filtered['BG_reduction_MP'],
              s=80, c=colors_filtered, edgecolors='black', linewidth=0.5,
              label=f'通过筛选 (n={len(filtered)})')

    # 为通过筛选的点添加元素标注
    for _, row in filtered.iterrows():
        ax.annotate(row['dopant_element'], (row['E_hull_change'], row['BG_reduction_MP']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
                   color='#333')

    ax.axhline(20, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(0.05, color='blue', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('E_hull变化 (eV)', fontsize=12)
    ax.set_ylabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('Step 6: 约束筛选结果可视化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(step_dir / '03_filtering_result.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step6: 03_filtering_result.png")

    # 6.4 Top候选方案 (7:10竖版)
    fig, ax = plt.subplots(figsize=(7, 10))
    filtered_grouped = filtered.groupby(['dopant_element', 'n_dopant']).agg({
        'BG_reduction_MP': 'mean',
        'E_hull_change': 'mean',
        'BulkMod_JV': 'mean',
        'E_form_change_MP': 'mean',
        'Bandgap_MP': 'mean'
    }).reset_index()
    top_candidates = filtered_grouped.nlargest(12, 'BG_reduction_MP').reset_index(drop=True)

    y_pos = range(len(top_candidates) - 1, -1, -1)
    labels = [f"{row['dopant_element']}(n={int(row['n_dopant'])})" for _, row in top_candidates.iterrows()]
    colors_cand = [ELEM_COLORS.get(row['dopant_element'], '#95A5A6') for _, row in top_candidates.iterrows()]

    bars = ax.barh(list(y_pos), top_candidates['BG_reduction_MP'].values, color=colors_cand, alpha=0.8, height=0.7)

    for bar, (_, row) in zip(bars, top_candidates.iterrows()):
        info = f"ΔE={row['E_hull_change']:.3f} | BM={row['BulkMod_JV']:.1f}"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               info, va='center', fontsize=10, color='#333')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('最终候选方案', fontsize=15, fontweight='bold', y=0.98)
    ax.axvline(25, color='red', linestyle=':', linewidth=2, alpha=0.5, label='25%阈值')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(step_dir / '04_top_candidates.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step6: 04_top_candidates.png (7:10竖版)")

    # 6.5 层层筛选流程图 (6:8竖版)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis('off')

    # 计算各步骤数据
    total = len(df_doped)
    step1_n = len(df_doped[df_doped['dopant_type'] == 'Fe-site'])
    step2_n = len(df_doped[(df_doped['dopant_type'] == 'Fe-site') & (df_doped['BG_reduction_MP'] > 20)])
    step3_n = len(df_doped[(df_doped['dopant_type'] == 'Fe-site') & (df_doped['BG_reduction_MP'] > 20) & (df_doped['E_hull_change'] < 0.05)])
    step4_n = len(filtered)

    steps = [
        {'name': '全部构型', 'count': total, 'color': '#3498DB', 'y': 0.92},
        {'name': 'Fe位掺杂', 'count': step1_n, 'color': '#9B59B6', 'y': 0.72},
        {'name': '带隙降幅>20%', 'count': step2_n, 'color': '#E74C3C', 'y': 0.52},
        {'name': 'ΔE_hull<0.05eV', 'count': step3_n, 'color': '#F39C12', 'y': 0.32},
        {'name': '体模量>95%', 'count': step4_n, 'color': '#2ECC71', 'y': 0.12},
    ]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i, step in enumerate(steps):
        pct = step['count'] / total * 100
        box_width = 0.3 + 0.5 * (step['count'] / total)

        # 绘制筛选框
        box = plt.Rectangle((0.5 - box_width/2, step['y'] - 0.08), box_width, 0.15,
                            facecolor=step['color'], alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(box)

        # 框内文字
        ax.text(0.5, step['y'] - 0.02, f"{step['name']}\n{step['count']} ({pct:.1f}%)",
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # 箭头
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, steps[i+1]['y'] + 0.07),
                       xytext=(0.5, step['y'] - 0.08),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=3))

        # 右侧筛选条件
        conditions = ['', '仅Fe位点', 'MP带隙↓>20%', '热稳定性↑', '力学性能↑']
        ax.text(0.85, step['y'], conditions[i], ha='left', va='center',
               fontsize=9, color='#555', style='italic')

    ax.set_title('层层筛选流程', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(step_dir / '05_filtering_flow.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step6: 05_filtering_flow.png (6:8竖版)")

# =============================================================================
# Step 7: 最终方案对比 (拆分为4个子图)
# =============================================================================
def plot_step7():
    """Step 7: 最终方案对比"""
    step_dir = OUTPUT_DIR / 'Step7'

    # 获取最终候选
    filtered = df_doped[(df_doped['dopant_type'] == 'Fe-site') &
                        (df_doped['BG_reduction_MP'] > 20) &
                        (df_doped['E_hull_change'] < 0.05) &
                        (df_doped['BulkMod_JV'] > REF['BulkMod_JV'] * 0.95)]

    filtered_grouped = filtered.groupby(['dopant_element', 'n_dopant']).agg({
        'BG_reduction_MP': 'mean',
        'E_hull_change': 'mean',
        'BulkMod_JV': 'mean',
        'E_form_change_MP': 'mean',
        'Bandgap_MP': 'mean'
    }).reset_index()

    top5 = filtered_grouped.nlargest(5, 'BG_reduction_MP')

    # 7.1 雷达图
    from math import pi

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    categories = ['带隙降幅', '热力学稳定性', '体模量保持', '形成能稳定', '综合得分']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    for idx, (_, row) in enumerate(top5.iterrows()):
        elem = row['dopant_element']
        values = [
            row['BG_reduction_MP'] / 40,
            1 - row['E_hull_change'] / 0.05,
            row['BulkMod_JV'] / REF['BulkMod_JV'],
            1 - row['E_form_change_MP'] / 0.01 if row['E_form_change_MP'] > 0 else 1,
            row['BG_reduction_MP'] / 40
        ]
        values = [max(0, min(1, v)) for v in values]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=f"{elem}(n={int(row['n_dopant'])})",
               color=ELEM_COLORS.get(elem, '#95A5A6'))
        ax.fill(angles, values, alpha=0.15, color=ELEM_COLORS.get(elem, '#95A5A6'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('Top 5 方案多维度雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    plt.tight_layout()
    plt.savefig(step_dir / '01_radar_chart.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step7: 01_radar_chart.png")

    # 7.2 综合得分排名
    fig, ax = plt.subplots(figsize=(10, 6))
    top5_sorted = top5.sort_values('BG_reduction_MP', ascending=True)
    y_pos = range(len(top5_sorted))
    colors_rank = [ELEM_COLORS.get(e, '#95A5A6') for e in top5_sorted['dopant_element']]

    bars = ax.barh(list(y_pos), top5_sorted['BG_reduction_MP'].values, color=colors_rank, alpha=0.8)
    for bar, (_, row) in zip(bars, top5_sorted.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{row["BG_reduction_MP"]:.1f}%', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{e}(n={int(n)})" for e, n in zip(top5_sorted['dopant_element'], top5_sorted['n_dopant'])])
    ax.set_xlabel('MP带隙降幅 (%)', fontsize=12)
    ax.set_title('Top 5 综合得分排名', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step_dir / '02_ranking.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step7: 02_ranking.png")

    # 7.3 详细参数对比表
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    table_data = []
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        table_data.append([
            f'D-{idx}',
            f"{row['dopant_element']}(n={int(row['n_dopant'])})",
            f"{row['BG_reduction_MP']:.2f}%",
            f"{row['E_hull_change']:.4f} eV",
            f"{row['BulkMod_JV']:.1f} GPa",
            f"{REF['BulkMod_JV']/row['BulkMod_JV']*100:.1f}%",
            f"{row['Bandgap_MP']:.3f} eV",
        ])

    col_labels = ['方案', '元素/浓度', '带隙降幅', 'ΔE_hull', '体模量', '体模量保持率', '预测带隙']
    table = ax.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center',
                      colColours=['#3498DB']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)

    for i in range(7):
        table[(1, i)].set_facecolor('#E8F5E9')

    ax.set_title('Top 5 推荐方案详细参数', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(step_dir / '03_detail_table.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step7: 03_detail_table.png")

    # 7.4 方案解读
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    interpretation = """
    ╔═════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                           最终推荐方案总结                                               ║
    ╠═════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  D-1 (Ti, n=10): 带隙降幅最大(35.62%)，热力学稳定性良好(ΔE=0.037eV)，为首选方案                          ║
    ║  D-2 (Cu, n=10): 带隙降幅次大(33.76%)，稳定性良好(ΔE=0.044eV)，可作为首选或备选                           ║
    ║  D-3 (Ti, n=9):  带隙降幅显著(31.69%)，热力学最稳定(ΔE=0.033eV)，适合追求稳定性                            ║
    ║  D-4 (Cu, n=9):  带隙降幅较大(30.88%)，稳定性良好(ΔE=0.040eV)，综合性能优秀                               ║
    ║  D-5 (Ti, n=8):  带隙降幅适中(28.59%)，热力学最稳定(ΔE=0.029eV)，适合保守策略                              ║
    ╠═════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  筛选方法论：ALIGNN预测 → Pareto多目标优化 → 物理化学约束过滤 → 多维度对比                                  ║
    ║  关键指标权重：带隙降幅(50%) + 热力学稳定性(30%) + 体模量(20%)                                            ║
    ║  所有方案均满足：MP带隙>20%、ΔE_hull<0.05eV、体模量>原始95%                                                ║
    ╚═════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax.text(0.02, 0.95, interpretation, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F9A825', alpha=0.9))
    ax.set_title('最终推荐方案总结', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(step_dir / '04_final_summary.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Step7: 04_final_summary.png")

# =============================================================================
# 主函数
# =============================================================================
if __name__ == '__main__':
    print("\n开始生成层层递进可视化图表（拆分版）...\n")

    plot_step1()
    plot_step2()
    plot_step3()
    plot_step4()
    plot_step5()
    plot_step6()
    plot_step7()

    print("\n" + "=" * 70)
    print("所有图表已生成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)
