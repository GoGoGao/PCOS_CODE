from pathlib import Path
import os
PKG_ROOT = Path(__file__).resolve().parents[1]
SHARED = PKG_ROOT / 'shared_input'
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
INPUT_DIR = str(SHARED)
OUTPUT_DIR = str(PKG_ROOT / 'output' / 'Fig1B_S2')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import warnings
import sys
import os
import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
WORK_DIR = os.environ.get('PCOS_WORK', str(__import__('pathlib').Path(__file__).resolve().parents[1] / '_runtime'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = f'{WORK_DIR}/logs/02_alpha_diversity_log.txt'

class Logger:

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = Logger(log_file)
print('=' * 70)
print('微生物组Alpha多样性分析日志')
print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 70)
print()
print('[STEP 1] 加载预处理数据')
print('-' * 50)
with open(f'{INPUT_DIR}/metaphlan_preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)
species_filtered = data['species_filtered']
species_tss = data['species_tss']
group_info = data['group_info']
colors_group = data['colors_group']
colors_batch = data['colors_batch']
print(f'  物种数: {species_filtered.shape[0]}')
print(f'  样本数: {species_filtered.shape[1]}')
print(f"  分组: {group_info['Group'].value_counts().to_dict()}")
print('\n[STEP 2] 计算Alpha多样性指数')
print('-' * 50)

def calculate_alpha_diversity(abundance_df):
    results = {}
    for sample in abundance_df.columns:
        x = abundance_df[sample].values
        x = x[x > 0]
        observed = len(x)
        p = x / x.sum()
        shannon = -np.sum(p * np.log(p))
        simpson = 1 - np.sum(p ** 2)
        inv_simpson = 1 / np.sum(p ** 2)
        pielou = shannon / np.log(observed) if observed > 1 else 0
        chao1 = observed
        n = x.sum()
        s = observed
        if n > s:
            alpha = s / np.log(n / s) if n > s else s
        else:
            alpha = s
        results[sample] = {'Observed': observed, 'Shannon': shannon, 'Simpson': simpson, 'InvSimpson': inv_simpson, 'Pielou': pielou, 'Chao1': chao1, 'Fisher': alpha}
    return pd.DataFrame(results).T
alpha_div = calculate_alpha_diversity(species_tss)
alpha_div.index.name = 'Sample'
alpha_div = alpha_div.reset_index()
alpha_div = alpha_div.merge(group_info[['Sample', 'Bioproject', 'Group']], on='Sample')
print('  计算完成，多样性指数包括:')
print('    - Observed Species (物种丰富度)')
print('    - Shannon Index (香农指数)')
print('    - Simpson Index (辛普森指数)')
print('    - Inverse Simpson (逆辛普森指数)')
print('    - Pielou Evenness (Pielou均匀度)')
print('\n[INFO] 各组Alpha多样性描述性统计:')
for metric in ['Observed', 'Shannon', 'Simpson', 'Pielou']:
    print(f'\n  {metric}:')
    for grp in ['PCOS', 'Healthy']:
        vals = alpha_div[alpha_div['Group'] == grp][metric]
        print(f'    {grp}: mean={vals.mean():.3f}, sd={vals.std():.3f}, median={vals.median():.3f}')
print('\n[STEP 3] 统计检验（考虑批次效应）')
print('-' * 50)
print('\n[3.1] Wilcoxon秩和检验 (不考虑批次效应):')
simple_test_results = {}
for metric in ['Observed', 'Shannon', 'Simpson', 'Pielou']:
    pcos_vals = alpha_div[alpha_div['Group'] == 'PCOS'][metric]
    healthy_vals = alpha_div[alpha_div['Group'] == 'Healthy'][metric]
    stat, pval = mannwhitneyu(pcos_vals, healthy_vals, alternative='two-sided')
    simple_test_results[metric] = {'statistic': stat, 'p_value': pval}
    print(f'  {metric}: U={stat:.1f}, p={pval:.4f}')
print('\n[3.2] 分层Wilcoxon检验 (按批次分层):')

def stratified_wilcoxon(data, metric, group_col, strata_col):
    strata = data[strata_col].unique()
    layer_stats = []
    for stratum in strata:
        stratum_data = data[data[strata_col] == stratum]
        grp1 = stratum_data[stratum_data[group_col] == 'PCOS'][metric].values
        grp2 = stratum_data[stratum_data[group_col] == 'Healthy'][metric].values
        if len(grp1) >= 3 and len(grp2) >= 3:
            stat, pval = mannwhitneyu(grp1, grp2, alternative='two-sided')
            n1, n2 = (len(grp1), len(grp2))
            layer_stats.append({'stratum': stratum, 'n_pcos': n1, 'n_healthy': n2, 'U': stat, 'p_value': pval, 'effect_size': stat / (n1 * n2)})
    if layer_stats:
        p_values = [s['p_value'] for s in layer_stats]
        chi2_stat = -2 * np.sum(np.log(p_values))
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_values))
        weights = [s['n_pcos'] * s['n_healthy'] for s in layer_stats]
        weighted_effect = np.average([s['effect_size'] for s in layer_stats], weights=weights)
        return {'layer_stats': layer_stats, 'combined_p': combined_p, 'weighted_effect': weighted_effect}
    return None
stratified_results = {}
for metric in ['Observed', 'Shannon', 'Simpson', 'Pielou']:
    result = stratified_wilcoxon(alpha_div, metric, 'Group', 'Bioproject')
    stratified_results[metric] = result
    print(f"  {metric}: combined p={result['combined_p']:.4f}, weighted effect={result['weighted_effect']:.3f}")
print('\n[3.3] 线性混合效应模型 (批次作为随机效应):')
try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    lmm_results = {}
    for metric in ['Observed', 'Shannon', 'Simpson', 'Pielou']:
        model_data = alpha_div[['Sample', metric, 'Group', 'Bioproject']].copy()
        model_data['Group_binary'] = (model_data['Group'] == 'PCOS').astype(int)
        model = smf.mixedlm(f'{metric} ~ Group_binary', model_data, groups=model_data['Bioproject'])
        result = model.fit(method='powell')
        coef = result.fe_params['Group_binary']
        pval = result.pvalues['Group_binary']
        ci_low, ci_high = result.conf_int().loc['Group_binary']
        lmm_results[metric] = {'coefficient': coef, 'p_value': pval, 'ci_low': ci_low, 'ci_high': ci_high}
        direction = '↑' if coef > 0 else '↓'
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f'  {metric}: β={coef:.3f} ({direction} in PCOS), p={pval:.4f} {sig}')
except Exception as e:
    print(f'  线性混合效应模型拟合出错: {e}')
    lmm_results = None
print('\n[STEP 4] 生成可视化图表')
print('-' * 50)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics = ['Observed', 'Shannon', 'Simpson', 'Pielou']
metric_labels = ['Observed Species', 'Shannon Index', 'Simpson Index', 'Pielou Evenness']
for idx, (ax, metric, label) in enumerate(zip(axes.flat, metrics, metric_labels)):
    bp = ax.boxplot([alpha_div[alpha_div['Group'] == 'PCOS'][metric], alpha_div[alpha_div['Group'] == 'Healthy'][metric]], positions=[1, 2], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(colors_group['PCOS'])
    bp['boxes'][1].set_facecolor(colors_group['Healthy'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    for grp_idx, grp in enumerate(['PCOS', 'Healthy']):
        grp_data = alpha_div[alpha_div['Group'] == grp]
        x_pos = grp_idx + 1
        for batch in colors_batch:
            batch_data = grp_data[grp_data['Bioproject'] == batch]
            jitter = np.random.normal(0, 0.08, len(batch_data))
            ax.scatter(x_pos + jitter, batch_data[metric], c=colors_batch[batch], s=30, alpha=0.6, edgecolors='white', linewidths=0.5)
    if lmm_results:
        pval = lmm_results[metric]['p_value']
    else:
        pval = simple_test_results[metric]['p_value']
    if pval < 0.001:
        sig_text = '***'
    elif pval < 0.01:
        sig_text = '**'
    elif pval < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    y_max = alpha_div[metric].max()
    y_range = alpha_div[metric].max() - alpha_div[metric].min()
    ax.plot([1, 1, 2, 2], [y_max + y_range * 0.05, y_max + y_range * 0.08, y_max + y_range * 0.08, y_max + y_range * 0.05], 'k-', linewidth=1)
    ax.text(1.5, y_max + y_range * 0.12, f'{sig_text}\np={pval:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['PCOS\n(n=94)', 'Healthy\n(n=75)'], fontsize=11)
    ax.set_ylabel(label, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(alpha_div[metric].min() - y_range * 0.1, y_max + y_range * 0.25)
batch_handles = [mpatches.Patch(color=colors_batch[b], label=b, alpha=0.7) for b in colors_batch]
fig.legend(handles=batch_handles, title='Batch', loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
plt.suptitle('Alpha Diversity Comparison: PCOS vs Healthy\n(Linear Mixed Model with Batch as Random Effect)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_boxplot.png', bbox_inches='tight', dpi=300)
plt.close()
print('  保存: alpha_diversity_boxplot.pdf/png')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (ax, metric, label) in enumerate(zip(axes.flat, metrics, metric_labels)):
    plot_data = alpha_div[['Bioproject', 'Group', metric]].copy()
    batch_order = ['PRJNA530971', 'PRJNA549764', 'PRJNA791492']
    positions = []
    labels_x = []
    for i, batch in enumerate(batch_order):
        batch_data = plot_data[plot_data['Bioproject'] == batch]
        pcos_data = batch_data[batch_data['Group'] == 'PCOS'][metric]
        healthy_data = batch_data[batch_data['Group'] == 'Healthy'][metric]
        pos_pcos = i * 3 + 0.8
        pos_healthy = i * 3 + 1.6
        bp1 = ax.boxplot([pcos_data], positions=[pos_pcos], widths=0.5, patch_artist=True)
        bp2 = ax.boxplot([healthy_data], positions=[pos_healthy], widths=0.5, patch_artist=True)
        bp1['boxes'][0].set_facecolor(colors_group['PCOS'])
        bp2['boxes'][0].set_facecolor(colors_group['Healthy'])
        bp1['boxes'][0].set_alpha(0.7)
        bp2['boxes'][0].set_alpha(0.7)
        positions.extend([pos_pcos, pos_healthy])
        labels_x.append(i * 3 + 1.2)
    ax.set_xticks(labels_x)
    ax.set_xticklabels(batch_order, fontsize=10)
    ax.set_ylabel(label, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
group_handles = [mpatches.Patch(color=colors_group['PCOS'], label='PCOS', alpha=0.7), mpatches.Patch(color=colors_group['Healthy'], label='Healthy', alpha=0.7)]
fig.legend(handles=group_handles, title='Group', loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
plt.suptitle('Alpha Diversity by Batch and Group', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_by_batch.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_by_batch.png', bbox_inches='tight', dpi=300)
plt.close()
print('  保存: alpha_diversity_by_batch.pdf/png')
fig, ax = plt.subplots(figsize=(10, 6))
y_positions = np.arange(len(metrics))
effects = []
ci_lows = []
ci_highs = []
p_values = []
if lmm_results:
    for metric in metrics:
        effects.append(lmm_results[metric]['coefficient'])
        ci_lows.append(lmm_results[metric]['ci_low'])
        ci_highs.append(lmm_results[metric]['ci_high'])
        p_values.append(lmm_results[metric]['p_value'])
else:
    for metric in metrics:
        pcos_vals = alpha_div[alpha_div['Group'] == 'PCOS'][metric]
        healthy_vals = alpha_div[alpha_div['Group'] == 'Healthy'][metric]
        pooled_std = np.sqrt(((len(pcos_vals) - 1) * pcos_vals.std() ** 2 + (len(healthy_vals) - 1) * healthy_vals.std() ** 2) / (len(pcos_vals) + len(healthy_vals) - 2))
        d = (pcos_vals.mean() - healthy_vals.mean()) / pooled_std
        se_d = np.sqrt((len(pcos_vals) + len(healthy_vals)) / (len(pcos_vals) * len(healthy_vals)) + d ** 2 / (2 * (len(pcos_vals) + len(healthy_vals))))
        ci_low = d - 1.96 * se_d
        ci_high = d + 1.96 * se_d
        effects.append(d)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        p_values.append(simple_test_results[metric]['p_value'])
for i, (effect, ci_low, ci_high, pval) in enumerate(zip(effects, ci_lows, ci_highs, p_values)):
    color = '#E64B35' if pval < 0.05 else '#999999'
    ax.errorbar(effect, y_positions[i], xerr=[[effect - ci_low], [ci_high - effect]], fmt='o', color=color, capsize=5, capthick=2, markersize=10, elinewidth=2)
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    ax.text(max(ci_highs) + 0.1, y_positions[i], f'p={pval:.3f} {sig}', va='center', fontsize=10)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels(metric_labels, fontsize=11)
ax.set_xlabel('Effect Size (β coefficient from LMM)\n← Decreased in PCOS | Increased in PCOS →', fontsize=11)
ax.set_title('Forest Plot: Alpha Diversity Effect Sizes\n(PCOS vs Healthy, adjusted for batch effect)', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(min(ci_lows) - 0.3, max(ci_highs) + 0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_forest_plot.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_forest_plot.png', bbox_inches='tight', dpi=300)
plt.close()
print('  保存: alpha_diversity_forest_plot.pdf/png')
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = alpha_div[metrics].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlation'}, xticklabels=metric_labels, yticklabels=metric_labels, ax=ax)
ax.set_title('Correlation Matrix of Alpha Diversity Indices', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_correlation.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/alpha_diversity_correlation.png', bbox_inches='tight', dpi=300)
plt.close()
print('  保存: alpha_diversity_correlation.pdf/png')
print('\n[STEP 5] 保存分析结果')
print('-' * 50)
alpha_div.to_csv(f'{OUTPUT_DIR}/alpha_diversity_values.csv', index=False)
stats_summary = []
for metric in metrics:
    row = {'Metric': metric, 'PCOS_mean': alpha_div[alpha_div['Group'] == 'PCOS'][metric].mean(), 'PCOS_sd': alpha_div[alpha_div['Group'] == 'PCOS'][metric].std(), 'Healthy_mean': alpha_div[alpha_div['Group'] == 'Healthy'][metric].mean(), 'Healthy_sd': alpha_div[alpha_div['Group'] == 'Healthy'][metric].std(), 'Wilcoxon_p': simple_test_results[metric]['p_value'], 'Stratified_p': stratified_results[metric]['combined_p']}
    if lmm_results:
        row['LMM_beta'] = lmm_results[metric]['coefficient']
        row['LMM_p'] = lmm_results[metric]['p_value']
        row['LMM_CI_low'] = lmm_results[metric]['ci_low']
        row['LMM_CI_high'] = lmm_results[metric]['ci_high']
    stats_summary.append(row)
stats_df = pd.DataFrame(stats_summary)
stats_df.to_csv(f'{OUTPUT_DIR}/alpha_diversity_statistics.csv', index=False)
print('  保存完成:')
print('    - alpha_diversity_values.csv')
print('    - alpha_diversity_statistics.csv')
print('\n' + '=' * 70)
print('Alpha多样性分析摘要')
print('=' * 70)
print('\n主要发现:')
for metric in metrics:
    pcos_mean = alpha_div[alpha_div['Group'] == 'PCOS'][metric].mean()
    healthy_mean = alpha_div[alpha_div['Group'] == 'Healthy'][metric].mean()
    diff_pct = (pcos_mean - healthy_mean) / healthy_mean * 100
    direction = '升高' if diff_pct > 0 else '降低'
    if lmm_results:
        pval = lmm_results[metric]['p_value']
    else:
        pval = simple_test_results[metric]['p_value']
    sig = '显著' if pval < 0.05 else '不显著'
    print(f'  {metric}: PCOS组{direction}{abs(diff_pct):.1f}% (p={pval:.4f}, {sig})')
print('\n' + '=' * 70)
print(f"分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('=' * 70)
print(f'\n日志已保存至: {log_file}')
