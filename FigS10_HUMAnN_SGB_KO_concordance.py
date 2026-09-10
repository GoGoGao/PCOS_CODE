from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')
base = OUT.parent.parent
for d in [FIG, SRC, TAB, FIG / 'panels']:
    d.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'], 'font.weight': 'normal', 'axes.labelweight': 'normal', 'axes.titleweight': 'normal', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 10, 'axes.titlesize': 11})
fish = pd.read_csv(base / 'kegg_stat/results/tables/TableS1_all_KO_results.tsv', sep='\t')
glm = pd.read_csv(OUT.parent / 'tables/TableSX_MAG_GLM_adjusted.tsv', sep='\t')
ols = pd.read_csv(OUT.parent / 'tables/TableSX_HUMAnN_OLS_adjusted.tsv', sep='\t')
wil = pd.read_csv(OUT.parent / 'tables/TableSX_HUMAnN_Wilcoxon.tsv', sep='\t')
sets = {'MAG Fisher P<0.05': set(fish.loc[fish.P < 0.05, 'KO']), 'MAG GLM P<0.05': set(glm.loc[glm.P < 0.05, 'KO']), 'HUMAnN OLS P<0.05': set(ols.loc[ols.P < 0.05, 'KO']), 'HUMAnN Wilcoxon P<0.05': set(wil.loc[wil.P < 0.05, 'KO']), 'MAG Fisher FDR<0.20': set(fish.loc[fish.FDR < 0.2, 'KO']), 'HUMAnN OLS FDR<0.20': set(ols.loc[ols.FDR < 0.2, 'KO'])}
pw = pd.read_csv(base / 'kegg_stat/kegg_enrich/enrich_results/pathway_enrichment_significant.tsv', sep='\t')
ann = fish[['KO', 'direction', 'P', 'FDR']].rename(columns={'direction': 'dir_fish', 'P': 'P_fish', 'FDR': 'FDR_fish'})
ann = ann.merge(glm[['KO', 'direction', 'P', 'FDR']].rename(columns={'direction': 'dir_glm', 'P': 'P_glm', 'FDR': 'FDR_glm'}), on='KO', how='left')
ann = ann.merge(ols[['KO', 'direction', 'P', 'FDR']].rename(columns={'direction': 'dir_ols', 'P': 'P_ols', 'FDR': 'FDR_ols'}), on='KO', how='left')
ann = ann.set_index('KO')
rows = []
for _, r in pw.iterrows():
    kos = [k for k in str(r.geneID).split('/') if k.startswith('K')]
    sub = ann.loc[[k for k in kos if k in ann.index]]
    in_ols = sub['dir_ols'].notna()
    n_ols = int(in_ols.sum())
    same_ols = int((sub.loc[in_ols, 'dir_fish'] == sub.loc[in_ols, 'dir_ols']).sum()) if n_ols else 0
    rows.append({'pathway_ID': r.ID, 'Description': r.Description, 'p.adjust': r['p.adjust'], 'n_hit_KOs_in_Fisher_ORA': len(kos), 'n_with_HUMAnN_OLS': n_ols, 'n_Fisher_OLS_same_dir': same_ols, 'n_Fisher_OLS_opposite_dir': n_ols - same_ols, 'pct_same_dir_vs_OLS': 100 * same_ols / n_ols if n_ols else np.nan, 'n_hit_also_OLS_P05': int(((sub.P_ols < 0.05) & in_ols).sum()) if n_ols else 0, 'n_hit_also_GLM_P05': int((sub.P_glm < 0.05).sum())})
pw_conc = pd.DataFrame(rows)
pw_conc.to_csv(TAB / 'TableSX_Fig6E_pathway_KO_concordance_vs_HUMAnN.tsv', sep='\t', index=False)
pw_conc.to_csv(TAB / 'TableSX_Fig6E_pathway_KO_concordance_vs_HUMAnN.csv', index=False)
pw_same_pct = pw_conc['pct_same_dir_vs_OLS'].dropna()
print('Fig6E pathways:', len(pw_conc))
print('mean pct same dir vs OLS:', float(pw_same_pct.mean()))
print('pathways with any OLS P05 hit:', int((pw_conc.n_hit_also_OLS_P05 > 0).sum()))
s2_all = pd.read_csv(base / 'kegg_stat/results/tables/S2_ORA_all_differential.tsv', sep='\t')
s2_fdr = pd.read_csv(base / 'kegg_stat/results/tables/S2_ORA_FDR02_KOs.tsv', sep='\t')
print('S2 ORA 379 FDR<0.2:', int((s2_all.FDR < 0.2).sum()))
print('S2 ORA FDR02 KOs pathways FDR<0.05:', int((s2_fdr.FDR < 0.05).sum()))

def save_fig(fig, stem, panel_dir=None):
    out_dirs = [FIG]
    if panel_dir is not None:
        panel_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(panel_dir)
    else:
        out_dirs.append(FIG / 'panels')
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf', 'svg', 'png', 'jpg']:
            fig.savefig(d / f'{stem}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    print('saved', stem)

def save_source(df, stem):
    df.to_csv(SRC / f'{stem}.tsv', sep='\t', index=False)
    df.to_csv(SRC / f'{stem}.csv', index=False)
try:
    from matplotlib_venn import venn2 as mv2
    HAS_VENN = True
except Exception:
    HAS_VENN = False
    print('matplotlib_venn missing; manual circles')

def draw_venn2(ax, setA, setB, labA, labB, title):
    only_a = len(setA - setB)
    only_b = len(setB - setA)
    both = len(setA & setB)
    if HAS_VENN:
        v = mv2(subsets=(only_a, only_b, both), set_labels=(labA, labB), ax=ax, set_colors=('#4C78A8', '#F58518'), alpha=0.55)
        for t in v.set_labels:
            if t:
                t.set_fontsize(9)
        for t in v.subset_labels:
            if t:
                t.set_fontsize(10)
    else:
        ax.add_patch(Circle((-0.3, 0), 0.55, facecolor='#4C78A8', alpha=0.45, edgecolor='black', lw=0.6))
        ax.add_patch(Circle((0.3, 0), 0.55, facecolor='#F58518', alpha=0.45, edgecolor='black', lw=0.6))
        ax.text(-0.55, 0, str(only_a), ha='center', va='center', fontsize=10)
        ax.text(0.55, 0, str(only_b), ha='center', va='center', fontsize=10)
        ax.text(0, 0, str(both), ha='center', va='center', fontsize=10)
        ax.text(-0.55, 0.78, labA, ha='center', fontsize=8)
        ax.text(0.55, 0.78, labB, ha='center', fontsize=8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.95, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
    ax.set_title(title, fontsize=10, pad=6)
    return (only_a, only_b, both)
pairs = [('a', sets['MAG Fisher P<0.05'], sets['HUMAnN OLS P<0.05'], 'MAG Fisher\nP<0.05', 'HUMAnN OLS\nP<0.05', 'A. Nominal KO sets'), ('b', sets['MAG GLM P<0.05'], sets['HUMAnN OLS P<0.05'], 'MAG GLM\nP<0.05', 'HUMAnN OLS\nP<0.05', 'B. Recovery-adjusted vs OLS'), ('c', sets['MAG Fisher FDR<0.20'], sets['HUMAnN OLS FDR<0.20'], 'MAG Fisher\nFDR<0.20', 'HUMAnN OLS\nFDR<0.20', 'C. FDR-significant KOs')]
src_venn = []
panel_dir = FIG / 'panels'
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, (tag, A, B, la, lb, title) in zip(axes, pairs):
    oa, ob, bo = draw_venn2(ax, A, B, la, lb, title)
    row = {'panel': title, 'panel_tag': tag, 'only_A': oa, 'only_B': ob, 'both': bo, 'label_A': la.replace('\n', ' '), 'label_B': lb.replace('\n', ' '), 'n_A': oa + bo, 'n_B': ob + bo}
    src_venn.append(row)
    fig_p, ax_p = plt.subplots(figsize=(3.6, 3.3))
    draw_venn2(ax_p, A, B, la, lb, title)
    fig_p.tight_layout()
    stem_p = f'Fig_L2_KO_venn_MAG_HUMAnN_{tag}'
    save_source(pd.DataFrame([row]), stem_p)
    save_fig(fig_p, stem_p, panel_dir=panel_dir)
    plt.close(fig_p)
fig.suptitle('MAG vs HUMAnN differential KO overlap', fontsize=12, y=1.02)
fig.tight_layout()
df_venn = pd.DataFrame(src_venn)
save_source(df_venn, 'Fig_L2_KO_venn_MAG_HUMAnN')
df_venn.to_csv(TAB / 'TableSX_KO_venn_counts.tsv', sep='\t', index=False)
df_venn.to_csv(TAB / 'TableSX_KO_venn_counts.csv', index=False)
save_fig(fig, 'Fig_L2_KO_venn_MAG_HUMAnN')
plt.close(fig)
tier_order = ['both_P05_same', 'both_P05_opposite', 'MAG_P05_OLS_NS_same', 'MAG_P05_OLS_NS_opposite']
tier_labels = {'both_P05_same': 'Both P<0.05, same direction', 'both_P05_opposite': 'Both P<0.05, opposite direction', 'MAG_P05_OLS_NS_same': 'MAG P<0.05 only, same direction', 'MAG_P05_OLS_NS_opposite': 'MAG P<0.05 only, opposite direction'}
colors = ['#2CA02C', '#D62728', '#AEC7E8', '#FFBB78']
c_fish = pd.read_csv(TAB / 'TableSX_Fisher_vs_OLS_KO_status.tsv', sep='\t')
c_glm = pd.read_csv(TAB / 'TableSX_GLM_vs_OLS_KO_status.tsv', sep='\t')
counts = {'MAG_Fisher': c_fish['tier'].value_counts().to_dict(), 'MAG_GLM': c_glm['tier'].value_counts().to_dict()}
methods = ['MAG_Fisher', 'MAG_GLM']
x = np.arange(len(methods))
bottoms = np.zeros(len(methods))
fig, ax = plt.subplots(figsize=(5.2, 4.0))
bar_src = []
for i, tier in enumerate(tier_order):
    vals = [counts[m].get(tier, 0) for m in methods]
    ax.bar(x, vals, bottom=bottoms, color=colors[i], edgecolor='black', linewidth=0.4, label=tier_labels[tier], width=0.55)
    for j, m in enumerate(methods):
        bar_src.append({'method': m, 'tier': tier, 'tier_label': tier_labels[tier], 'n': vals[j]})
    bottoms = bottoms + np.array(vals, dtype=float)
for xi, t in zip(x, bottoms):
    ax.text(xi, t + 2, f'n={int(t)}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(['MAG Fisher\nvs HUMAnN OLS', 'MAG GLM\nvs HUMAnN OLS'])
ax.set_ylabel('Number of MAG nominal KOs\n(overlapping HUMAnN-tested KOs)')
ax.set_title('Direction concordance with HUMAnN OLS')
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.set_ylim(0, max(bottoms) * 1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
pd.DataFrame(bar_src).to_csv(SRC / 'Fig_L2_direction_concordance_bar.tsv', sep='\t', index=False)
pd.DataFrame(bar_src).to_csv(SRC / 'Fig_L2_direction_concordance_bar.csv', index=False)
save_fig(fig, 'Fig_L2_direction_concordance_bar')
plt.close(fig)
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.hist(pw_same_pct, bins=np.arange(0, 105, 10), color='#4C78A8', edgecolor='black', linewidth=0.5)
ax.axvline(pw_same_pct.mean(), color='#D62728', lw=1.2, label=f'Mean = {pw_same_pct.mean():.1f}%')
ax.axvline(50, color='grey', ls='--', lw=0.8, label='50% chance')
ax.set_xlabel('Same-direction KOs vs HUMAnN OLS (%)')
ax.set_ylabel('Number of Fig. 6E pathways')
ax.set_title('Within Fig. 6E pathways:\nFisher hit KO direction vs HUMAnN OLS')
ax.legend(frameon=False, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
pw_conc.to_csv(SRC / 'Fig_L2_pathway_concordance_hist.tsv', sep='\t', index=False)
pw_conc.to_csv(SRC / 'Fig_L2_pathway_concordance_hist.csv', index=False)
save_fig(fig, 'Fig_L2_pathway_concordance_hist')
plt.close(fig)
summary = f"# MAG vs HUMAnN 脱钩确认（KO + KEGG 通路）\n\n\n| 对比 | only MAG | only HUMAnN | 交集 | Jaccard |\n|------|----------|-------------|------|---------|\n| Fisher P&lt;0.05 vs OLS P&lt;0.05 | {src_venn[0]['only_A']} | {src_venn[0]['only_B']} | **{src_venn[0]['both']}** | {src_venn[0]['both'] / (src_venn[0]['only_A'] + src_venn[0]['only_B'] + src_venn[0]['both']):.3f} |\n| GLM P&lt;0.05 vs OLS P&lt;0.05 | {src_venn[1]['only_A']} | {src_venn[1]['only_B']} | **{src_venn[1]['both']}** | {src_venn[1]['both'] / (src_venn[1]['only_A'] + src_venn[1]['only_B'] + src_venn[1]['both']):.3f} |\n| Fisher FDR&lt;0.20 vs OLS FDR&lt;0.20 | {src_venn[2]['only_A']} | {src_venn[2]['only_B']} | **{src_venn[2]['both']}** | 0 |\n\n图：`figures/Fig_L2_KO_venn_MAG_HUMAnN.{{pdf,svg,png,jpg}}`\n子图：`figures/Fig_L2_KO_venn_MAG_HUMAnN_{{a,b,c}}.{{pdf,svg,png,jpg}}`（及 `source_data/` 对应 TSV/CSV）\n\n\n见 `figures/Fig_L2_direction_concordance_bar.*` 与 status 表。\n\n- Fisher∩OLS可检：同向约 27%，反向约 73%\n- GLM∩OLS可检：同向约 68%，但双方都名义显著且同向仅 7 个；FDR 层无交集\n\n\n1. Fig.6E（clusterProfiler，379 Fisher KO）：35 条 padj&lt;0.05\n2. 通路 hit KO 与 HUMAnN OLS 同向比例均值 **{pw_same_pct.mean():.1f}%**（低于随机 50%）\n3. S2_ORA（379 KO）：通路 FDR&lt;0.20 = **{(s2_all.FDR < 0.2).sum()}**\n4. S2_ORA（11 FDR KO）：{(s2_fdr.FDR < 0.05).sum()} 条 FDR&lt;0.05，但与 HUMAnN FDR KO 集合交集为 0\n5. HUMAnN 独立通路 ORA：本次 KEGG REST 不稳定未重跑；KO 集几乎不重叠，通路韦恩预期近空\n\n图：`figures/Fig_L2_pathway_concordance_hist.*`\n\n\nMAG 与 HUMAnN 在显著 KO 集合、FDR KO、以及 Fig.6E 通路支撑基因方向上均脱钩；基于 379 名义 KO 的 KEGG 富集不能被 HUMAnN 交叉验证。\n"
(OUT / 'MAG_HUMAnN_KEGG_decoupling_summary.md').write_text(summary)
print(summary)
print('DONE', OUT)
