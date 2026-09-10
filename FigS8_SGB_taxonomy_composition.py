from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')
INPUT_FILE = 'sgb.info.comb.tsv'
OUTPUT_DIR = './sgb_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/source_data', exist_ok=True)
COLORS_GROUP = {'PCOS': '#E64B35', 'Healthy': '#4DBBD5'}
COLORS_BATCH = {'PRJNA530971': '#00A087', 'PRJNA549764': '#3C5488', 'PRJNA791492': '#F39B7F'}
PHYLUM_COLORS = {'p__Bacillota_A': '#E64B35', 'p__Bacteroidota': '#4DBBD5', 'p__Actinomycetota': '#00A087', 'p__Bacillota_I': '#3C5488', 'p__Bacillota_C': '#F39B7F', 'p__Pseudomonadota': '#8491B4', 'p__Bacillota': '#91D1C2', 'p__Desulfobacterota': '#DC0000', 'p__Verrucomicrobiota': '#7E6148', 'p__Cyanobacteriota': '#B09C85', 'p__Fusobacteriota': '#E18727', 'p__Patescibacteria': '#FFDC91', 'p__Methanobacteriota': '#6A6599', 'Other': '#CCCCCC'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 1.2, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.major.size': 5, 'ytick.major.size': 5})
print('=' * 60)
print('加载数据...')
df = pd.read_csv(INPUT_FILE, sep='\t')
print(f'  总MAG数: {len(df)}')
print(f'  列名: {list(df.columns)}')
df['Sample'] = df['ID'].str.replace('_bin\\.\\d+$', '', regex=True)
df['is_novel'] = (df['S'] == 's__') | df['S'].isna() | (df['S'] == '')
print(f"  已知物种MAG: {(~df['is_novel']).sum()}")
print(f"  新物种MAG: {df['is_novel'].sum()}")

def classify_quality(row):
    if row['completeness'] >= 90 and row['contamination'] < 5:
        return 'High'
    elif row['completeness'] >= 50 and row['contamination'] < 10:
        return 'Medium'
    else:
        return 'Low'
df['quality'] = df.apply(classify_quality, axis=1)
print(f"  High quality: {(df['quality'] == 'High').sum()}")
print(f"  Medium quality: {(df['quality'] == 'Medium').sum()}")
print(f"  Low quality: {(df['quality'] == 'Low').sum()}")
df['is_representative'] = df['centrality'] == 0.0
print(f"  SGB代表基因组: {df['is_representative'].sum()}")
print(f"  非代表基因组: {(~df['is_representative']).sum()}")
phylum_map = {'p__Bacillota_A': 'Bacillota_A', 'p__Bacteroidota': 'Bacteroidota', 'p__Actinomycetota': 'Actinomycetota', 'p__Bacillota_I': 'Bacillota_I', 'p__Bacillota_C': 'Bacillota_C', 'p__Pseudomonadota': 'Pseudomonadota', 'p__Bacillota': 'Bacillota', 'p__Desulfobacterota': 'Desulfobacterota', 'p__Verrucomicrobiota': 'Verrucomicrobiota', 'p__Cyanobacteriota': 'Cyanobacteriota', 'p__Fusobacteriota': 'Fusobacteriota', 'p__Patescibacteria': 'Patescibacteria', 'p__Methanobacteriota': 'Methanobacteriota'}
df['Phylum_short'] = df['P'].map(phylum_map).fillna('Other')

def save_fig(fig, name, tight=True):
    if tight:
        fig.savefig(f'{OUTPUT_DIR}/{name}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.15)
        fig.savefig(f'{OUTPUT_DIR}/{name}.jpg', format='jpg', bbox_inches='tight', pad_inches=0.15, dpi=300)
    else:
        fig.savefig(f'{OUTPUT_DIR}/{name}.pdf', format='pdf')
        fig.savefig(f'{OUTPUT_DIR}/{name}.jpg', format='jpg', dpi=300)
    plt.close(fig)
    print(f'  已保存: {name}.pdf / .jpg')

def save_source(data, name):
    data.to_csv(f'{OUTPUT_DIR}/source_data/{name}.tsv', sep='\t', index=True)
    print(f'  已保存源数据: source_data/{name}.tsv')
print('\n' + '=' * 60)
print('模块1: MAG质量评估全景图')
print('=' * 60)
print('\n--- Figure 1A: Completeness vs Contamination ---')
fig, ax = plt.subplots(figsize=(7, 6))
ax.axhspan(0, 5, xmin=0, xmax=1, alpha=0.03, color='green', zorder=0)
ax.axhspan(5, 10, xmin=0, xmax=1, alpha=0.03, color='orange', zorder=0)
phyla_order = df['P'].value_counts().index.tolist()
for phylum in phyla_order:
    mask = df['P'] == phylum
    color = PHYLUM_COLORS.get(phylum, '#CCCCCC')
    label = phylum_map.get(phylum, phylum.replace('p__', ''))
    ax.scatter(df.loc[mask, 'completeness'], df.loc[mask, 'contamination'], c=color, s=35, alpha=0.7, edgecolors='white', linewidths=0.3, label=f'{label} (n={mask.sum()})', zorder=2)
ax.axvline(x=90, color='#333333', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
ax.axvline(x=50, color='#333333', linestyle=':', linewidth=0.8, alpha=0.4, zorder=1)
ax.axhline(y=5, color='#333333', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
ax.axhline(y=10, color='#333333', linestyle=':', linewidth=0.8, alpha=0.4, zorder=1)
ax.text(95, 0.3, 'High\nQuality', fontsize=9, color='green', alpha=0.7, ha='center', fontweight='bold')
ax.text(70, 7, 'Medium Quality', fontsize=9, color='orange', alpha=0.7, ha='center', fontweight='bold')
ax.set_xlabel('Completeness (%)')
ax.set_ylabel('Contamination (%)')
ax.set_title('MAG Quality Assessment (MIMAG Standards)', fontweight='bold', pad=12)
ax.set_xlim(78, 101)
ax.set_ylim(-0.3, max(df['contamination']) * 1.1)
handles, labels = ax.get_legend_handles_labels()
keep = [i for i, l in enumerate(labels) if int(l.split('n=')[1].rstrip(')')) >= 3]
ax.legend([handles[i] for i in keep], [labels[i] for i in keep], loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5, ncol=2, columnspacing=0.8, handletextpad=0.3, markerscale=0.9)
sns.despine()
save_fig(fig, 'Fig1A_completeness_vs_contamination')
src_1a = df[['ID', 'P', 'Phylum_short', 'completeness', 'contamination', 'quality', 'BIOPROJECT', 'Group']].copy()
save_source(src_1a, 'Fig1A_completeness_vs_contamination')
print('\n--- Figure 1B: MAG quality by cohort ---')
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = [('completeness', 'Completeness (%)', (78, 101)), ('contamination', 'Contamination (%)', None), ('N50', 'N50 (bp)', None)]
for idx, (metric, ylabel, ylim) in enumerate(metrics):
    ax = axes[idx]
    plot_data = df[['BIOPROJECT', 'Group', metric]].copy()
    parts = ax.violinplot([df.loc[df['BIOPROJECT'] == bp, metric].values for bp in COLORS_BATCH.keys()], positions=range(len(COLORS_BATCH)), showmeans=False, showmedians=False, showextrema=False)
    for i, (bp, pc) in enumerate(zip(COLORS_BATCH.keys(), parts['bodies'])):
        pc.set_facecolor(COLORS_BATCH[bp])
        pc.set_alpha(0.35)
        pc.set_edgecolor(COLORS_BATCH[bp])
    bp_data = [df.loc[df['BIOPROJECT'] == bp, metric].values for bp in COLORS_BATCH.keys()]
    box = ax.boxplot(bp_data, positions=range(len(COLORS_BATCH)), widths=0.25, patch_artist=True, showfliers=False, zorder=3)
    for i, (patch, bp_name) in enumerate(zip(box['boxes'], COLORS_BATCH.keys())):
        patch.set_facecolor(COLORS_BATCH[bp_name])
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    for element in ['whiskers', 'caps']:
        for line in box[element]:
            line.set_color('black')
            line.set_linewidth(0.8)
    for line in box['medians']:
        line.set_color('white')
        line.set_linewidth(1.5)
    for i, bp_name in enumerate(COLORS_BATCH.keys()):
        vals = df.loc[df['BIOPROJECT'] == bp_name, metric].values
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, c=COLORS_BATCH[bp_name], s=12, alpha=0.4, edgecolors='none', zorder=2)
    ax.set_xticks(range(len(COLORS_BATCH)))
    batch_labels = [f"{bp}\n(n={len(df[df['BIOPROJECT'] == bp])})" for bp in COLORS_BATCH.keys()]
    ax.set_xticklabels(batch_labels, fontsize=9.5)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if metric == 'N50':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1000:.0f}k'))
    ax.set_title(ylabel.split(' (')[0], fontweight='bold')
    sns.despine(ax=ax)
plt.tight_layout(w_pad=3)
save_fig(fig, 'Fig1B_quality_by_cohort')
src_1b = df[['ID', 'BIOPROJECT', 'Group', 'completeness', 'contamination', 'N50']].copy()
save_source(src_1b, 'Fig1B_quality_by_cohort')
print('\n--- Figure 1C: Genome size vs N50 ---')
fig, ax = plt.subplots(figsize=(7, 6))
for phylum in phyla_order:
    mask = df['P'] == phylum
    if mask.sum() < 3:
        continue
    color = PHYLUM_COLORS.get(phylum, '#CCCCCC')
    label = phylum_map.get(phylum, phylum.replace('p__', ''))
    ax.scatter(df.loc[mask, 'length'] / 1000000.0, df.loc[mask, 'N50'] / 1000.0, c=color, s=35, alpha=0.7, edgecolors='white', linewidths=0.3, label=f'{label} (n={mask.sum()})', zorder=2)
ax.set_xlabel('Genome Size (Mb)')
ax.set_ylabel('N50 (kb)')
ax.set_title('Genome Size vs Assembly Contiguity', fontweight='bold', pad=12)
handles, labels = ax.get_legend_handles_labels()
keep = [i for i, l in enumerate(labels) if int(l.split('n=')[1].rstrip(')')) >= 3]
ax.legend([handles[i] for i in keep], [labels[i] for i in keep], loc='upper right', frameon=True, framealpha=0.9, fontsize=8.5, ncol=2, columnspacing=0.8, handletextpad=0.3, markerscale=0.9)
sns.despine()
save_fig(fig, 'Fig1C_genome_size_vs_N50')
src_1c = df[['ID', 'P', 'Phylum_short', 'length', 'N50', 'BIOPROJECT', 'Group']].copy()
save_source(src_1c, 'Fig1C_genome_size_vs_N50')
print('\n--- Table 1: MAG quality summary ---')
rows = []
for bp in ['PRJNA530971', 'PRJNA549764', 'PRJNA791492', 'All']:
    for grp in ['PCOS', 'Healthy', 'All']:
        if bp == 'All' and grp == 'All':
            sub = df
        elif bp == 'All':
            sub = df[df['Group'] == grp]
        elif grp == 'All':
            sub = df[df['BIOPROJECT'] == bp]
        else:
            sub = df[(df['BIOPROJECT'] == bp) & (df['Group'] == grp)]
        if len(sub) == 0:
            continue
        n_samples = sub['Sample'].nunique()
        rows.append({'Cohort': bp, 'Group': grp, 'N_MAGs': len(sub), 'N_Samples': n_samples, 'MAGs_per_sample': f'{len(sub) / n_samples:.1f}', 'High_quality_N': (sub['quality'] == 'High').sum(), 'High_quality_pct': f"{(sub['quality'] == 'High').mean() * 100:.1f}%", 'Median_completeness': f"{sub['completeness'].median():.1f}", 'Median_contamination': f"{sub['contamination'].median():.2f}", 'Median_N50': f"{sub['N50'].median():.0f}", 'Median_genome_size_Mb': f"{sub['length'].median() / 1000000.0:.2f}", 'N_novel_species': sub['is_novel'].sum()})
table1 = pd.DataFrame(rows)
table1.to_csv(f'{OUTPUT_DIR}/Table1_MAG_quality_summary.tsv', sep='\t', index=False)
print(f'  已保存: Table1_MAG_quality_summary.tsv')
print(table1.to_string(index=False))
print('\n--- Figure 1 Combined Panel ---')
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
ax_a = fig.add_subplot(gs[0, :2])
for phylum in phyla_order:
    mask = df['P'] == phylum
    if mask.sum() < 3:
        continue
    color = PHYLUM_COLORS.get(phylum, '#CCCCCC')
    label = phylum_map.get(phylum, phylum.replace('p__', ''))
    ax_a.scatter(df.loc[mask, 'completeness'], df.loc[mask, 'contamination'], c=color, s=28, alpha=0.7, edgecolors='white', linewidths=0.2, label=f'{label} (n={mask.sum()})', zorder=2)
ax_a.axvline(x=90, color='#333', linestyle='--', linewidth=0.8, alpha=0.5)
ax_a.axhline(y=5, color='#333', linestyle='--', linewidth=0.8, alpha=0.5)
ax_a.axhspan(0, 5, alpha=0.03, color='green')
ax_a.text(95, 0.3, 'High Quality', fontsize=8, color='green', alpha=0.7, ha='center', fontweight='bold')
ax_a.set_xlabel('Completeness (%)')
ax_a.set_ylabel('Contamination (%)')
ax_a.set_xlim(78, 101)
ax_a.set_ylim(-0.3, max(df['contamination']) * 1.1)
handles, labels_leg = ax_a.get_legend_handles_labels()
keep = [i for i, l in enumerate(labels_leg) if int(l.split('n=')[1].rstrip(')')) >= 3]
ax_a.legend([handles[i] for i in keep], [labels_leg[i] for i in keep], loc='upper left', frameon=True, framealpha=0.9, fontsize=7.5, ncol=2, columnspacing=0.6, handletextpad=0.2, markerscale=0.8)
ax_a.set_title('Completeness vs Contamination', fontweight='bold')
ax_a.text(-0.08, 1.05, 'A', transform=ax_a.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_a)
ax_b = fig.add_subplot(gs[0, 2])
q_counts = df['quality'].value_counts()
colors_q = {'High': '#00A087', 'Medium': '#F39B7F', 'Low': '#E64B35'}
wedges, texts, autotexts = ax_b.pie([q_counts.get('High', 0), q_counts.get('Medium', 0), q_counts.get('Low', 0)], labels=['High', 'Medium', 'Low'], colors=[colors_q['High'], colors_q['Medium'], colors_q['Low']], autopct='%1.1f%%', startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
for t in autotexts:
    t.set_fontsize(10)
    t.set_fontweight('bold')
for t in texts:
    t.set_fontsize(10)
ax_b.set_title(f'MAG Quality Distribution\n(n={len(df)})', fontweight='bold', fontsize=11)
ax_b.text(-0.08, 1.05, 'B', transform=ax_b.transAxes, fontsize=18, fontweight='bold', va='top')
for ci, (metric, ylabel, ylim_r) in enumerate([('completeness', 'Completeness (%)', (78, 101)), ('contamination', 'Contamination (%)', None), ('N50', 'N50 (bp)', None)]):
    ax_c = fig.add_subplot(gs[1, ci])
    bp_data = [df.loc[df['BIOPROJECT'] == bp, metric].values for bp in COLORS_BATCH.keys()]
    parts = ax_c.violinplot(bp_data, positions=range(3), showmeans=False, showmedians=False, showextrema=False)
    for i, (bp_name, pc) in enumerate(zip(COLORS_BATCH.keys(), parts['bodies'])):
        pc.set_facecolor(COLORS_BATCH[bp_name])
        pc.set_alpha(0.35)
        pc.set_edgecolor(COLORS_BATCH[bp_name])
    box = ax_c.boxplot(bp_data, positions=range(3), widths=0.22, patch_artist=True, showfliers=False, zorder=3)
    for i, (patch, bp_name) in enumerate(zip(box['boxes'], COLORS_BATCH.keys())):
        patch.set_facecolor(COLORS_BATCH[bp_name])
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    for el in ['whiskers', 'caps']:
        for line in box[el]:
            line.set_color('black')
            line.set_linewidth(0.8)
    for line in box['medians']:
        line.set_color('white')
        line.set_linewidth(1.5)
    ax_c.set_xticks(range(3))
    ax_c.set_xticklabels([f"{bp}\n(n={len(df[df['BIOPROJECT'] == bp])})" for bp in COLORS_BATCH], fontsize=8.5)
    ax_c.set_ylabel(ylabel, fontsize=11)
    if ylim_r:
        ax_c.set_ylim(ylim_r)
    if metric == 'N50':
        ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1000:.0f}k'))
    panel_label = chr(ord('C') + ci)
    ax_c.text(-0.12, 1.05, panel_label, transform=ax_c.transAxes, fontsize=18, fontweight='bold', va='top')
    ax_c.set_title(ylabel.split(' (')[0], fontweight='bold', fontsize=11)
    sns.despine(ax=ax_c)
save_fig(fig, 'Fig1_MAG_quality_combined')
print('\n' + '=' * 60)
print('模块2: SGB分类组成特征分析')
print('=' * 60)
print('\n--- Figure 2A: Phylum composition ---')
df['Cohort_Group'] = df['BIOPROJECT'] + '\n' + df['Group']
phyla_top = df['P'].value_counts().head(8).index.tolist()
comp_data = []
for cg in ['PRJNA530971\nPCOS', 'PRJNA530971\nHealthy', 'PRJNA549764\nPCOS', 'PRJNA549764\nHealthy', 'PRJNA791492\nPCOS', 'PRJNA791492\nHealthy']:
    sub = df[df['Cohort_Group'] == cg]
    total = len(sub)
    row = {'Cohort_Group': cg}
    for p in phyla_top:
        row[phylum_map.get(p, p)] = (sub['P'] == p).sum() / total * 100 if total > 0 else 0
    other_count = sum((sub['P'] == p for p in df['P'].unique() if p not in phyla_top))
    row['Other'] = (total - sum(((sub['P'] == p).sum() for p in phyla_top))) / total * 100 if total > 0 else 0
    row['n'] = total
    comp_data.append(row)
comp_df = pd.DataFrame(comp_data)
comp_df = comp_df.set_index('Cohort_Group')
n_vals = comp_df['n']
comp_df = comp_df.drop('n', axis=1)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comp_df))
bottom = np.zeros(len(comp_df))
bar_colors = []
for col in comp_df.columns:
    for k, v in phylum_map.items():
        if v == col:
            bar_colors.append(PHYLUM_COLORS[k])
            break
    else:
        bar_colors.append(PHYLUM_COLORS.get('Other', '#CCCCCC'))
for i, (col, color) in enumerate(zip(comp_df.columns, bar_colors)):
    vals = comp_df[col].values
    ax.bar(x, vals, bottom=bottom, label=col, color=color, edgecolor='white', linewidth=0.5, width=0.7)
    bottom += vals
ax.set_xticks(x)
xlabels = [f'{idx}\n(n={int(n)})' for idx, n in zip(comp_df.index, n_vals)]
ax.set_xticklabels(xlabels, fontsize=9, ha='center')
ax.set_ylabel('Proportion of MAGs (%)')
ax.set_title('Phylum-level Composition of MAGs across Cohorts', fontweight='bold', pad=12)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)
ax.set_ylim(0, 105)
sns.despine()
plt.tight_layout()
save_fig(fig, 'Fig2A_phylum_composition')
save_source(comp_df.reset_index(), 'Fig2A_phylum_composition')
print('\n--- Figure 2B: Top 15 Family heatmap ---')
family_counts = df['F'].value_counts()
top15_families = family_counts.head(15).index.tolist()
groups_6 = ['PRJNA530971\nPCOS', 'PRJNA530971\nHealthy', 'PRJNA549764\nPCOS', 'PRJNA549764\nHealthy', 'PRJNA791492\nPCOS', 'PRJNA791492\nHealthy']
fam_matrix = pd.DataFrame(index=[f.replace('f__', '') for f in top15_families], columns=groups_6)
for cg in groups_6:
    sub = df[df['Cohort_Group'] == cg]
    total = len(sub)
    for f in top15_families:
        cnt = (sub['F'] == f).sum()
        fam_matrix.loc[f.replace('f__', ''), cg] = cnt / total * 100 if total > 0 else 0
fam_matrix = fam_matrix.astype(float)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(fam_matrix, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.8, linecolor='white', ax=ax, cbar_kws={'label': 'Proportion (%)', 'shrink': 0.7}, annot_kws={'size': 9})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.set_title('Top 15 Families: MAG Proportion by Cohort × Group', fontweight='bold', pad=12)
plt.tight_layout()
save_fig(fig, 'Fig2B_family_heatmap')
save_source(fam_matrix, 'Fig2B_family_heatmap')
print('\n--- Figure 2C: Top 20 Genus bubble plot ---')
genus_counts = df['G'].value_counts()
top20_genera = genus_counts.head(20).index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
for gi, grp in enumerate(['PCOS', 'Healthy']):
    ax = axes[gi]
    sub = df[df['Group'] == grp]
    genera_data = []
    for g in top20_genera:
        g_sub = sub[sub['G'] == g]
        cnt = len(g_sub)
        if cnt == 0:
            continue
        phylum = g_sub['P'].mode().iloc[0] if len(g_sub) > 0 else 'Other'
        genera_data.append({'Genus': g.replace('g__', ''), 'Count': cnt, 'Phylum': phylum})
    gdf = pd.DataFrame(genera_data)
    if len(gdf) == 0:
        continue
    gdf = gdf.sort_values('Count', ascending=True)
    y_pos = np.arange(len(gdf))
    colors = [PHYLUM_COLORS.get(p, '#CCCCCC') for p in gdf['Phylum']]
    sizes = gdf['Count'].values * 12
    ax.scatter(gdf['Count'], y_pos, s=sizes, c=colors, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=3)
    for yi, (cnt, genus) in enumerate(zip(gdf['Count'], gdf['Genus'])):
        ax.text(cnt + 0.5, yi, str(cnt), va='center', fontsize=8.5, color='#333')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gdf['Genus'], fontsize=10)
    ax.set_xlabel('Number of MAGs')
    ax.set_title(f'{grp}', fontweight='bold', fontsize=14, color=COLORS_GROUP[grp])
    ax.set_xlim(0, gdf['Count'].max() * 1.3)
    sns.despine(ax=ax)
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PHYLUM_COLORS.get(p, '#CCCCCC'), markersize=10, label=phylum_map.get(p, p.replace('p__', ''))) for p in phyla_order[:8]]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Top 20 Genera: MAG Distribution', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, 'Fig2C_genus_bubble')
genus_src = []
for g in top20_genera:
    for grp in ['PCOS', 'Healthy']:
        sub = df[(df['G'] == g) & (df['Group'] == grp)]
        genus_src.append({'Genus': g, 'Group': grp, 'MAG_count': len(sub), 'Phylum': sub['P'].mode().iloc[0] if len(sub) > 0 else 'NA'})
save_source(pd.DataFrame(genus_src), 'Fig2C_genus_bubble')
print('\n--- Table 2: Taxonomy summary ---')
tax_rows = []
for level, col, prefix in [('Phylum', 'P', 'p__'), ('Family', 'F', 'f__'), ('Genus', 'G', 'g__')]:
    for taxon in df[col].value_counts().head(20).index:
        for grp in ['PCOS', 'Healthy', 'All']:
            sub_grp = df if grp == 'All' else df[df['Group'] == grp]
            cnt = (sub_grp[col] == taxon).sum()
            pct = cnt / len(sub_grp) * 100
            tax_rows.append({'Level': level, 'Taxon': taxon.replace(prefix, ''), 'Group': grp, 'MAG_count': cnt, 'Proportion_pct': round(pct, 2)})
table2 = pd.DataFrame(tax_rows)
table2.to_csv(f'{OUTPUT_DIR}/Table2_taxonomy_summary.tsv', sep='\t', index=False)
print(f'  已保存: Table2_taxonomy_summary.tsv')
print('\n--- Figure 2 Combined Panel ---')
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
ax_a = fig.add_subplot(gs[0, 0])
bottom = np.zeros(len(comp_df))
for i, (col, color) in enumerate(zip(comp_df.columns, bar_colors)):
    vals = comp_df[col].values
    ax_a.bar(np.arange(len(comp_df)), vals, bottom=bottom, label=col, color=color, edgecolor='white', linewidth=0.5, width=0.7)
    bottom += vals
ax_a.set_xticks(np.arange(len(comp_df)))
ax_a.set_xticklabels([f'{idx}\n(n={int(n)})' for idx, n in zip(comp_df.index, n_vals)], fontsize=7.5, ha='center')
ax_a.set_ylabel('Proportion (%)')
ax_a.set_ylim(0, 105)
ax_a.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7.5, frameon=True)
ax_a.set_title('Phylum-level Composition', fontweight='bold', fontsize=12)
ax_a.text(-0.08, 1.05, 'A', transform=ax_a.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_a)
ax_b = fig.add_subplot(gs[0, 1])
sns.heatmap(fam_matrix, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5, linecolor='white', ax=ax_b, cbar_kws={'label': '%', 'shrink': 0.7}, annot_kws={'size': 7})
ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=45, ha='right', fontsize=7)
ax_b.set_yticklabels(ax_b.get_yticklabels(), fontsize=8)
ax_b.set_title('Top 15 Families', fontweight='bold', fontsize=12)
ax_b.text(-0.08, 1.05, 'B', transform=ax_b.transAxes, fontsize=18, fontweight='bold', va='top')
ax_c = fig.add_subplot(gs[1, 0])
sub = df[df['Group'] == 'PCOS']
gdata = []
for g in top20_genera:
    g_sub = sub[sub['G'] == g]
    if len(g_sub) > 0:
        gdata.append({'Genus': g.replace('g__', ''), 'Count': len(g_sub), 'Phylum': g_sub['P'].mode().iloc[0]})
gdf_p = pd.DataFrame(gdata).sort_values('Count', ascending=True)
if len(gdf_p) > 0:
    y_pos = np.arange(len(gdf_p))
    colors_p = [PHYLUM_COLORS.get(p, '#CCCCCC') for p in gdf_p['Phylum']]
    ax_c.scatter(gdf_p['Count'], y_pos, s=gdf_p['Count'] * 10, c=colors_p, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=3)
    for yi, cnt in enumerate(gdf_p['Count']):
        ax_c.text(cnt + 0.3, yi, str(cnt), va='center', fontsize=7.5, color='#333')
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(gdf_p['Genus'], fontsize=8.5)
    ax_c.set_xlabel('Number of MAGs')
    ax_c.set_title('PCOS', fontweight='bold', color=COLORS_GROUP['PCOS'])
    ax_c.set_xlim(0, gdf_p['Count'].max() * 1.3)
ax_c.text(-0.12, 1.05, 'C', transform=ax_c.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_c)
ax_d = fig.add_subplot(gs[1, 1])
sub = df[df['Group'] == 'Healthy']
gdata = []
for g in top20_genera:
    g_sub = sub[sub['G'] == g]
    if len(g_sub) > 0:
        gdata.append({'Genus': g.replace('g__', ''), 'Count': len(g_sub), 'Phylum': g_sub['P'].mode().iloc[0]})
gdf_h = pd.DataFrame(gdata).sort_values('Count', ascending=True)
if len(gdf_h) > 0:
    y_pos = np.arange(len(gdf_h))
    colors_h = [PHYLUM_COLORS.get(p, '#CCCCCC') for p in gdf_h['Phylum']]
    ax_d.scatter(gdf_h['Count'], y_pos, s=gdf_h['Count'] * 10, c=colors_h, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=3)
    for yi, cnt in enumerate(gdf_h['Count']):
        ax_d.text(cnt + 0.3, yi, str(cnt), va='center', fontsize=7.5, color='#333')
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(gdf_h['Genus'], fontsize=8.5)
    ax_d.set_xlabel('Number of MAGs')
    ax_d.set_title('Healthy', fontweight='bold', color=COLORS_GROUP['Healthy'])
    ax_d.set_xlim(0, gdf_h['Count'].max() * 1.3)
ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_d)
save_fig(fig, 'Fig2_taxonomy_combined')
print('\n' + '=' * 60)
print('模块3: 新物种发现与特征分析')
print('=' * 60)
novel = df[df['is_novel']].copy()
known = df[~df['is_novel']].copy()
print(f'  新物种MAG: {len(novel)}')
print(f'  已知物种MAG: {len(known)}')
print('\n--- Figure 3A: Known vs Novel donut ---')
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
groups_for_pie = [('Overall', df), ('PRJNA530971', df[df['BIOPROJECT'] == 'PRJNA530971']), ('PRJNA549764', df[df['BIOPROJECT'] == 'PRJNA549764']), ('PRJNA791492', df[df['BIOPROJECT'] == 'PRJNA791492'])]
colors_kn = ['#4DBBD5', '#E64B35']
for idx, (title, sub_df) in enumerate(groups_for_pie):
    ax = axes[idx]
    n_known = (~sub_df['is_novel']).sum()
    n_novel = sub_df['is_novel'].sum()
    if n_novel == 0:
        wedges, texts, autotexts = ax.pie([n_known], labels=None, colors=[colors_kn[0]], autopct='%1.1f%%', startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    else:
        wedges, texts, autotexts = ax.pie([n_known, n_novel], labels=None, colors=colors_kn, autopct='%1.1f%%', startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax.text(0, 0, f'n={len(sub_df)}', ha='center', va='center', fontsize=13, fontweight='bold', color='#333')
    ax.set_title(f'{title}', fontweight='bold', fontsize=12, pad=8)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_kn[0], label=f'Known species (n={len(known)})'), Patch(facecolor=colors_kn[1], label=f'Novel species (n={len(novel)})')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.05), frameon=True)
fig.suptitle('Proportion of Known vs Novel Species MAGs', fontweight='bold', fontsize=14, y=1.03)
plt.tight_layout()
save_fig(fig, 'Fig3A_known_vs_novel_donut')
src_3a = pd.DataFrame([{'Category': title, 'Known': (~sub_df['is_novel']).sum(), 'Novel': sub_df['is_novel'].sum(), 'Total': len(sub_df), 'Novel_pct': sub_df['is_novel'].mean() * 100} for title, sub_df in groups_for_pie])
save_source(src_3a, 'Fig3A_known_vs_novel')
print('\n--- Figure 3B: Novel species taxonomy ---')
novel_tax = novel[['P', 'C', 'O', 'F', 'G']].copy()
for col in ['P', 'C', 'O', 'F', 'G']:
    novel_tax[col] = novel_tax[col].str.replace('^[a-z]__', '', regex=True)
genus_novel_counts = novel_tax['G'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
novel_genus_data = []
for g, cnt in genus_novel_counts.items():
    phylum = novel[novel['G'].str.contains(g)]['P'].mode().iloc[0]
    novel_genus_data.append({'Genus': g, 'Count': cnt, 'Phylum': phylum})
ngdf = pd.DataFrame(novel_genus_data).sort_values('Count', ascending=True)
y_pos = np.arange(len(ngdf))
colors_ng = [PHYLUM_COLORS.get(p, '#CCCCCC') for p in ngdf['Phylum']]
bars = ax.barh(y_pos, ngdf['Count'], color=colors_ng, edgecolor='white', height=0.7, alpha=0.9)
for yi, (cnt, genus) in enumerate(zip(ngdf['Count'], ngdf['Genus'])):
    ax.text(cnt + 0.2, yi, f'{cnt}', va='center', fontsize=11, fontweight='bold', color='#333')
ax.set_yticks(y_pos)
ax.set_yticklabels(ngdf['Genus'], fontsize=11)
ax.set_xlabel('Number of Novel Species MAGs', fontsize=12)
ax.set_title('Taxonomic Origin of Novel Species', fontweight='bold', pad=12)
ax.set_xlim(0, ngdf['Count'].max() * 1.25)
unique_phyla = ngdf['Phylum'].unique()
legend_el = [Patch(facecolor=PHYLUM_COLORS.get(p, '#CCCCCC'), label=phylum_map.get(p, p.replace('p__', ''))) for p in unique_phyla]
ax.legend(handles=legend_el, loc='lower right', fontsize=10, frameon=True, title='Phylum')
sns.despine()
save_fig(fig, 'Fig3B_novel_taxonomy_origin')
save_source(ngdf, 'Fig3B_novel_taxonomy_origin')
print('\n--- Figure 3C: Novel species PCOS vs Healthy ---')
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ax = axes[0]
novel_grp = novel['Group'].value_counts()
pcos_n = novel_grp.get('PCOS', 0)
healthy_n = novel_grp.get('Healthy', 0)
bars = ax.bar(['PCOS', 'Healthy'], [pcos_n, healthy_n], color=[COLORS_GROUP['PCOS'], COLORS_GROUP['Healthy']], edgecolor='white', width=0.5, alpha=0.9)
for bar, val in zip(bars, [pcos_n, healthy_n]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val), ha='center', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Novel Species MAGs')
ax.set_title('Novel Species by Group', fontweight='bold')
sns.despine(ax=ax)
ax2 = axes[1]
pcos_total = len(df[df['Group'] == 'PCOS'])
healthy_total = len(df[df['Group'] == 'Healthy'])
pcos_pct = pcos_n / pcos_total * 100
healthy_pct = healthy_n / healthy_total * 100
bars2 = ax2.bar(['PCOS', 'Healthy'], [pcos_pct, healthy_pct], color=[COLORS_GROUP['PCOS'], COLORS_GROUP['Healthy']], edgecolor='white', width=0.5, alpha=0.9)
for bar, val in zip(bars2, [pcos_pct, healthy_pct]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
from scipy.stats import fisher_exact
table_fisher = [[pcos_n, pcos_total - pcos_n], [healthy_n, healthy_total - healthy_n]]
odds_ratio, p_fisher = fisher_exact(table_fisher)
sig_text = f"Fisher's exact test\nOR = {odds_ratio:.2f}, P = {p_fisher:.3f}"
ax2.text(0.5, 0.85, sig_text, transform=ax2.transAxes, fontsize=10, ha='center', va='top', style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
ax2.set_ylabel('Novel Species Proportion (%)')
ax2.set_title('Novel Species Rate by Group', fontweight='bold')
sns.despine(ax=ax2)
plt.tight_layout(w_pad=3)
save_fig(fig, 'Fig3C_novel_PCOS_vs_Healthy')
src_3c = pd.DataFrame({'Group': ['PCOS', 'Healthy'], 'Novel_MAGs': [pcos_n, healthy_n], 'Total_MAGs': [pcos_total, healthy_total], 'Novel_pct': [pcos_pct, healthy_pct], 'Fisher_OR': [odds_ratio, odds_ratio], 'Fisher_P': [p_fisher, p_fisher]})
save_source(src_3c, 'Fig3C_novel_PCOS_vs_Healthy')
print('\n--- Figure 3D: Collinsella novel species focus ---')
collinsella = novel[novel['G'] == 'g__Collinsella'].copy()
print(f'  Collinsella新物种MAG数: {len(collinsella)}')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes[0, 0]
colors_col = [COLORS_GROUP[g] for g in collinsella['Group']]
markers_col = ['o' if bp == 'PRJNA530971' else 's' if bp == 'PRJNA791492' else '^' for bp in collinsella['BIOPROJECT']]
for i, (_, row) in enumerate(collinsella.iterrows()):
    m = 'o' if row['BIOPROJECT'] == 'PRJNA530971' else 's' if row['BIOPROJECT'] == 'PRJNA791492' else '^'
    ax.scatter(row['completeness'], row['contamination'], c=COLORS_GROUP[row['Group']], marker=m, s=80, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=3)
ax.axvline(x=90, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Completeness (%)')
ax.set_ylabel('Contamination (%)')
ax.set_title('D1: Quality Assessment', fontweight='bold')
grp_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS_GROUP['PCOS'], markersize=9, label='PCOS'), Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS_GROUP['Healthy'], markersize=9, label='Healthy')]
bp_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='PRJNA530971'), Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='PRJNA791492')]
ax.legend(handles=grp_handles + bp_handles, fontsize=8, loc='upper left', frameon=True)
sns.despine(ax=ax)
ax = axes[0, 1]
for grp in ['PCOS', 'Healthy']:
    sub_c = collinsella[collinsella['Group'] == grp]
    if len(sub_c) > 0:
        ax.hist(sub_c['length'] / 1000000.0, bins=8, alpha=0.6, color=COLORS_GROUP[grp], edgecolor='white', label=f'{grp} (n={len(sub_c)})')
ax.set_xlabel('Genome Size (Mb)')
ax.set_ylabel('Count')
ax.set_title('D2: Genome Size Distribution', fontweight='bold')
ax.legend(fontsize=10, frameon=True)
sns.despine(ax=ax)
ax = axes[1, 0]
col_dist = collinsella.groupby(['BIOPROJECT', 'Group']).size().unstack(fill_value=0)
for g in ['PCOS', 'Healthy']:
    if g not in col_dist.columns:
        col_dist[g] = 0
col_dist = col_dist[['PCOS', 'Healthy']]
x_cd = np.arange(len(col_dist))
width = 0.35
bars1 = ax.bar(x_cd - width / 2, col_dist['PCOS'], width, label='PCOS', color=COLORS_GROUP['PCOS'], edgecolor='white')
bars2 = ax.bar(x_cd + width / 2, col_dist['Healthy'], width, label='Healthy', color=COLORS_GROUP['Healthy'], edgecolor='white')
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15, str(int(h)), ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(x_cd)
ax.set_xticklabels(col_dist.index, fontsize=10)
ax.set_ylabel('Number of MAGs')
ax.set_title('D3: Distribution by Cohort × Group', fontweight='bold')
ax.legend(fontsize=10, frameon=True)
sns.despine(ax=ax)
ax = axes[1, 1]
for i, (_, row) in enumerate(collinsella.iterrows()):
    ax.scatter(row['N50'] / 1000.0, row['completeness'], c=COLORS_GROUP[row['Group']], s=row['length'] / 10000.0, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=3)
ax.set_xlabel('N50 (kb)')
ax.set_ylabel('Completeness (%)')
ax.set_title('D4: N50 vs Completeness\n(bubble size ∝ genome size)', fontweight='bold')
for sz, label in [(100, '1 Mb'), (200, '2 Mb'), (300, '3 Mb')]:
    ax.scatter([], [], s=sz, c='gray', alpha=0.5, edgecolors='black', linewidths=0.5, label=label)
ax.legend(fontsize=8.5, loc='lower right', frameon=True, title='Genome size')
sns.despine(ax=ax)
fig.suptitle('Collinsella Novel Species: Detailed Characterization (n=23)', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, 'Fig3D_Collinsella_focus')
src_3d = collinsella[['ID', 'G', 'BIOPROJECT', 'Group', 'completeness', 'contamination', 'length', 'N50', 'quality']].copy()
save_source(src_3d, 'Fig3D_Collinsella_focus')
print('\n--- Table 3: All novel species ---')
table3 = novel[['ID', 'K', 'P', 'C', 'O', 'F', 'G', 'S', 'BIOPROJECT', 'Group', 'completeness', 'contamination', 'strain_heterogeneity', 'length', 'N50', 'quality']].copy()
table3['Genome_size_Mb'] = (table3['length'] / 1000000.0).round(2)
table3 = table3.sort_values(['G', 'BIOPROJECT', 'Group'])
table3.to_csv(f'{OUTPUT_DIR}/Table3_novel_species_all.tsv', sep='\t', index=False)
print(f'  已保存: Table3_novel_species_all.tsv')
print('\n--- Figure 3 Combined Panel ---')
fig = plt.figure(figsize=(18, 16))
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4, height_ratios=[1, 1, 1.2])
for idx, (title, sub_df) in enumerate(groups_for_pie):
    ax = fig.add_subplot(gs[0, idx])
    n_kn = (~sub_df['is_novel']).sum()
    n_nv = sub_df['is_novel'].sum()
    if n_nv == 0:
        wedges, texts, autotexts = ax.pie([n_kn], labels=None, colors=[colors_kn[0]], autopct='%1.1f%%', startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    else:
        wedges, texts, autotexts = ax.pie([n_kn, n_nv], labels=None, colors=colors_kn, autopct='%1.1f%%', startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax.text(0, 0, f'n={len(sub_df)}', ha='center', va='center', fontsize=12, fontweight='bold', color='#333')
    ax.set_title(title, fontweight='bold', fontsize=11)
    if idx == 0:
        ax.text(-0.15, 1.1, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
ax_b = fig.add_subplot(gs[1, :2])
ngdf_sorted = ngdf.sort_values('Count', ascending=True)
y_pos_b = np.arange(len(ngdf_sorted))
colors_ng_b = [PHYLUM_COLORS.get(p, '#CCCCCC') for p in ngdf_sorted['Phylum']]
ax_b.barh(y_pos_b, ngdf_sorted['Count'], color=colors_ng_b, edgecolor='white', height=0.7, alpha=0.9)
for yi, cnt in enumerate(ngdf_sorted['Count']):
    ax_b.text(cnt + 0.15, yi, str(cnt), va='center', fontsize=10, fontweight='bold', color='#333')
ax_b.set_yticks(y_pos_b)
ax_b.set_yticklabels(ngdf_sorted['Genus'], fontsize=10)
ax_b.set_xlabel('Number of Novel Species MAGs')
ax_b.set_title('Taxonomic Origin of Novel Species', fontweight='bold')
ax_b.set_xlim(0, ngdf_sorted['Count'].max() * 1.25)
unique_p = ngdf_sorted['Phylum'].unique()
leg_el = [Patch(facecolor=PHYLUM_COLORS.get(p, '#CCC'), label=phylum_map.get(p, p.replace('p__', ''))) for p in unique_p]
ax_b.legend(handles=leg_el, loc='lower right', fontsize=8.5, frameon=True, title='Phylum', title_fontsize=9)
ax_b.text(-0.1, 1.05, 'B', transform=ax_b.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_b)
ax_c1 = fig.add_subplot(gs[1, 2])
bars_c = ax_c1.bar(['PCOS', 'Healthy'], [pcos_n, healthy_n], color=[COLORS_GROUP['PCOS'], COLORS_GROUP['Healthy']], edgecolor='white', width=0.5, alpha=0.9)
for bar, val in zip(bars_c, [pcos_n, healthy_n]):
    ax_c1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val), ha='center', fontsize=12, fontweight='bold')
ax_c1.set_ylabel('Count')
ax_c1.set_title('Novel by Group', fontweight='bold')
ax_c1.text(-0.15, 1.05, 'C', transform=ax_c1.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_c1)
ax_c2 = fig.add_subplot(gs[1, 3])
bars_c2 = ax_c2.bar(['PCOS', 'Healthy'], [pcos_pct, healthy_pct], color=[COLORS_GROUP['PCOS'], COLORS_GROUP['Healthy']], edgecolor='white', width=0.5, alpha=0.9)
for bar, val in zip(bars_c2, [pcos_pct, healthy_pct]):
    ax_c2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax_c2.text(0.5, 0.88, f'P = {p_fisher:.3f}', transform=ax_c2.transAxes, fontsize=9.5, ha='center', style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
ax_c2.set_ylabel('Proportion (%)')
ax_c2.set_title('Novel Rate by Group', fontweight='bold')
ax_c2.text(-0.15, 1.05, 'D', transform=ax_c2.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_c2)
gs_sub = gs[2, :].subgridspec(1, 4, wspace=0.4)
ax_e1 = fig.add_subplot(gs_sub[0])
for _, row in collinsella.iterrows():
    m = 'o' if row['BIOPROJECT'] == 'PRJNA530971' else 's'
    ax_e1.scatter(row['completeness'], row['contamination'], c=COLORS_GROUP[row['Group']], marker=m, s=60, alpha=0.8, edgecolors='black', linewidths=0.4, zorder=3)
ax_e1.axvline(x=90, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
ax_e1.axhline(y=5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
ax_e1.set_xlabel('Completeness (%)', fontsize=10)
ax_e1.set_ylabel('Contamination (%)', fontsize=10)
ax_e1.set_title('Collinsella Quality', fontweight='bold', fontsize=10)
ax_e1.text(-0.15, 1.08, 'E', transform=ax_e1.transAxes, fontsize=18, fontweight='bold', va='top')
sns.despine(ax=ax_e1)
ax_e2 = fig.add_subplot(gs_sub[1])
for grp in ['PCOS', 'Healthy']:
    sub_c = collinsella[collinsella['Group'] == grp]
    if len(sub_c) > 0:
        ax_e2.hist(sub_c['length'] / 1000000.0, bins=8, alpha=0.6, color=COLORS_GROUP[grp], edgecolor='white', label=f'{grp} (n={len(sub_c)})')
ax_e2.set_xlabel('Genome Size (Mb)', fontsize=10)
ax_e2.set_ylabel('Count', fontsize=10)
ax_e2.set_title('Genome Size', fontweight='bold', fontsize=10)
ax_e2.legend(fontsize=8, frameon=True)
sns.despine(ax=ax_e2)
ax_e3 = fig.add_subplot(gs_sub[2])
if len(col_dist) > 0:
    x_e3 = np.arange(len(col_dist))
    w = 0.35
    b1 = ax_e3.bar(x_e3 - w / 2, col_dist['PCOS'], w, label='PCOS', color=COLORS_GROUP['PCOS'], edgecolor='white')
    b2 = ax_e3.bar(x_e3 + w / 2, col_dist['Healthy'], w, label='Healthy', color=COLORS_GROUP['Healthy'], edgecolor='white')
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax_e3.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)), ha='center', fontsize=9.5, fontweight='bold')
    ax_e3.set_xticks(x_e3)
    ax_e3.set_xticklabels(col_dist.index, fontsize=8.5)
    ax_e3.set_ylabel('Count', fontsize=10)
    ax_e3.set_title('Cohort × Group', fontweight='bold', fontsize=10)
    ax_e3.legend(fontsize=8, frameon=True)
sns.despine(ax=ax_e3)
ax_e4 = fig.add_subplot(gs_sub[3])
for _, row in collinsella.iterrows():
    ax_e4.scatter(row['N50'] / 1000.0, row['completeness'], c=COLORS_GROUP[row['Group']], s=row['length'] / 15000.0, alpha=0.7, edgecolors='black', linewidths=0.4, zorder=3)
ax_e4.set_xlabel('N50 (kb)', fontsize=10)
ax_e4.set_ylabel('Completeness (%)', fontsize=10)
ax_e4.set_title('N50 vs Completeness', fontweight='bold', fontsize=10)
sns.despine(ax=ax_e4)
save_fig(fig, 'Fig3_novel_species_combined')
print('\n' + '=' * 60)
print('所有分析完成！')
print(f'输出目录: {OUTPUT_DIR}')
print('=' * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    if os.path.isfile(f'{OUTPUT_DIR}/{f}'):
        size = os.path.getsize(f'{OUTPUT_DIR}/{f}') / 1024
        print(f'  {f} ({size:.1f} KB)')
print(f'\n源数据:')
for f in sorted(os.listdir(f'{OUTPUT_DIR}/source_data')):
    size = os.path.getsize(f'{OUTPUT_DIR}/source_data/{f}') / 1024
    print(f'  source_data/{f} ({size:.1f} KB)')
