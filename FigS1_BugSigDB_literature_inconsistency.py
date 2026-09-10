from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
ROOT = PKG_ROOT / 'output' / 'FigS1_BugSigDB'
TAB = SHARED / 'BugSigDB'
FIG = ROOT / 'figures'
PLOT = FIG / 'plotdata'
FONT = ''
FONT_IT = ''
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
STEM = 'Fig_BugSigDB_literature_inconsistency'
C = {'down': '#4C72B0', 'up': '#C44E52', 'unchanged': '#ADB5BD', 'na': '#E9ECEF'}
PROP = None
PROP_IT = None

def setup():
    global PROP, PROP_IT
    for d in (FIG / 'A', FIG / 'B', PLOT / 'A', PLOT / 'B'):
        d.mkdir(parents=True, exist_ok=True)
    for fp in (FONT, FONT_IT):
        if fp:
            font_manager.fontManager.addfont(fp)
    PROP = font_manager.FontProperties(fname=FONT) if FONT else font_manager.FontProperties(family='Times New Roman')
    PROP_IT = font_manager.FontProperties(fname=FONT_IT) if FONT_IT else font_manager.FontProperties(family='Times New Roman', style='italic')
    mpl.rcParams.update({'font.family': 'Times New Roman', 'font.serif': ['Times New Roman', 'Times'], 'font.weight': 'normal', 'axes.labelweight': 'normal', 'axes.titleweight': 'normal', 'font.size': 8, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

def save_plotdata(df: pd.DataFrame, stem: str, sub: str | None=None):
    out = PLOT / sub if sub else PLOT
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'{stem}.tsv', sep='\t', index=False)
    df.to_csv(out / f'{stem}.csv', index=False)

def save_fig(fig, stem: str, sub: str | None=None):
    out = FIG / sub if sub else FIG
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png', 'jpg'):
        fig.savefig(out / f'{stem}.{ext}')

def load_alpha() -> pd.DataFrame:
    return pd.read_csv(TAB / 'TableE01_alpha_diversity_by_experiment.tsv', sep='\t')

def load_genus_conflict() -> pd.DataFrame:
    cons = pd.read_csv(TAB / 'Table05_BugSigDB_genus_consensus.tsv', sep='\t')
    sub = cons.loc[(cons['n_signatures'] >= 4) & (cons['n_PCOS_up'] > 0) & (cons['n_PCOS_down'] > 0)].copy()
    sub = sub.sort_values(['n_signatures', 'vote_margin', 'taxon'], ascending=[False, True, True])
    sub['display'] = sub['taxon'].str.capitalize()
    sub['conflict_share'] = sub[['n_PCOS_up', 'n_PCOS_down']].min(axis=1) / sub['n_signatures']
    return sub.reset_index(drop=True)

def draw_alpha(ax, dat: pd.DataFrame, title: str):
    metrics = ['Shannon', 'Chao1', 'Richness', 'Simpson', 'Pielou']
    cats = ['decreased', 'unchanged', 'increased', 'not_reported']
    colors = [C['down'], C['unchanged'], C['up'], C['na']]
    x = np.arange(len(metrics))
    bottoms = np.zeros(len(metrics))
    for cat, col in zip(cats, colors):
        vals = [int(dat.loc[(dat.metric == m) & (dat.direction == cat), 'n_experiments'].iloc[0]) for m in metrics]
        ax.bar(x, vals, bottom=bottoms, color=col, width=0.72, label=cat.replace('_', ' '))
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottoms[i] + v / 2, str(v), ha='center', va='center', fontsize=7.5, color='#222', fontproperties=PROP)
        bottoms += np.array(vals, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontproperties=PROP)
    ax.set_ylabel('Experiments (n = 17)', fontproperties=PROP)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(PROP)
    ax.set_ylim(0, 17.8)
    ax.set_title(title, fontsize=9, loc='left', pad=2, y=1.2, fontproperties=PROP)
    shan = dat.loc[dat.metric == 'Shannon'].set_index('direction')['n_experiments']
    ax.legend(frameon=False, fontsize=7, ncol=4, loc='lower left', bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0, handlelength=1.15, columnspacing=0.85, prop=PROP)
    ax.text(0.0, -0.15, f"Shannon: decreased {int(shan['decreased'])}, unchanged {int(shan['unchanged'])}, increased {int(shan['increased'])}", transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#444', fontproperties=PROP, clip_on=False)

def draw_genus(ax, g: pd.DataFrame, title: str):
    d = g.copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, -d['n_PCOS_down'], color=C['down'], height=0.72, label='PCOS_down')
    ax.barh(y, d['n_PCOS_up'], color=C['up'], height=0.72, label='PCOS_up')
    ax.axvline(0, color='black', lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(list(d['display']), fontproperties=PROP_IT, fontsize=7.5)
    xmax = float(max(d['n_PCOS_up'].max(), d['n_PCOS_down'].max())) + 2.6
    ax.set_xlim(-xmax, xmax)
    for i, r in d.iterrows():
        ax.text(xmax - 0.1, i, f'up:down {int(r.n_PCOS_up)}:{int(r.n_PCOS_down)}  n={int(r.n_signatures)}', ha='right', va='center', fontsize=6.5, color='#555', fontproperties=PROP)
    ax.set_xlabel('Signature votes (left = PCOS_down; right = PCOS_up)', fontproperties=PROP)
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(PROP)
    ax.set_title(title, fontsize=9, loc='left', pad=2, y=1.18, fontproperties=PROP)
    n_margin1 = int((g['vote_margin'] <= 1).sum())
    ax.text(0.0, -0.14, f'n_signatures >= 4 with both directions; margin <= 1: {n_margin1}/{len(g)}', transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#444', fontproperties=PROP, clip_on=False)
    ax.legend(handles=[Patch(facecolor=C['down'], label='PCOS_down'), Patch(facecolor=C['up'], label='PCOS_up')], frameon=False, fontsize=7, ncol=2, loc='lower left', bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0, handlelength=1.15, columnspacing=1.0, prop=PROP)

def main():
    setup()
    alpha = load_alpha()
    genus = load_genus_conflict()
    for m in alpha['metric'].unique():
        assert int(alpha.loc[alpha.metric == m, 'n_experiments'].sum()) == 17
    assert len(genus) >= 5
    save_plotdata(alpha, f'{STEM}_A', sub='A')
    save_plotdata(genus, f'{STEM}_B', sub='B')
    save_plotdata(alpha, f'{STEM}_A')
    save_plotdata(genus, f'{STEM}_B')
    fig_a, ax_a = plt.subplots(figsize=(4.8, 4.0))
    draw_alpha(ax_a, alpha, 'A  Alpha-diversity directions are inconsistent')
    fig_a.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.18)
    save_fig(fig_a, f'{STEM}_A', sub='A')
    plt.close(fig_a)
    fig_b, ax_b = plt.subplots(figsize=(5.4, 5.4))
    draw_genus(ax_b, genus, 'B  Genus markers show opposing votes')
    fig_b.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.14)
    save_fig(fig_b, f'{STEM}_B', sub='B')
    plt.close(fig_b)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), gridspec_kw={'width_ratios': [1.0, 1.25]})
    draw_alpha(axes[0], alpha, 'A  Alpha-diversity directions are inconsistent')
    draw_genus(axes[1], genus, 'B  Genus markers show opposing votes')
    fig.suptitle('BugSigDB primary PCOS signatures: cross-study inconsistency', fontsize=10, fontproperties=PROP, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.14, wspace=0.28)
    save_fig(fig, STEM)
    plt.close(fig)
    print(f'Wrote {STEM} (+ A/B) to {FIG}')
    print(f'Panel B genera: {len(genus)}')
    print(genus[['taxon', 'n_signatures', 'n_PCOS_up', 'n_PCOS_down', 'vote_margin']].to_string(index=False))
if __name__ == '__main__':
    main()
ROOT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
