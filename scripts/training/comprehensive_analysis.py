"""
Comprehensive Analysis with Multiple Metrics
- C-index comparison
- Fold variance analysis
- Dataset efficiency (performance per patient)
- Complexity vs Performance
- Statistical significance heatmap
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
sns.set_palette("husl")

print("=" * 80)
print("Comprehensive Multi-Metric Analysis")
print("=" * 80)

# ============================================================
# 1. Load All Results
# ============================================================

print("\n[1] Loading all results...")

results_files = {
    'Image-Only': 'results/image_only/cv_results.json',
    'RNA-Only': 'results/rnaseq_only/cv_results.json',
    'Partial\nModality': 'results/partial_modality/cv_results.json',
    'SimMLM': 'results/simmim/cv_results.json',
    'MMsurv': 'results/mmsurv/cv_results.json',
    'Simple\nFusion': 'results/simple_fusion/cv_results.json',
    'Flexible\nMM': 'results/flexible_multimodal/cv_results.json',
}

all_results = {}
for model_name, filepath in results_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

            if 'c_index_mean' in data:
                all_results[model_name] = {
                    'mean': data['c_index_mean'],
                    'std': data['c_index_std'],
                    'fold_values': [f['best_c_index'] for f in data['fold_results']],
                    'n_patients': data.get('dataset_size', None)
                }
            else:
                # Image-only format
                c_indices = [f['best_c_index'] for f in data['fold_results']]
                all_results[model_name] = {
                    'mean': np.mean(c_indices),
                    'std': np.std(c_indices),
                    'fold_values': c_indices,
                    'n_patients': None
                }
        print(f"  ✓ {model_name}: {all_results[model_name]['mean']:.4f} ± {all_results[model_name]['std']:.4f}")
    else:
        print(f"  ✗ {model_name}: Not found")

print(f"\n  Total: {len(all_results)} models loaded")

# Dataset sizes
dataset_sizes = {
    'Image-Only': 142,
    'RNA-Only': 264,
    'Partial\nModality': 608,
    'SimMLM': 348,
    'MMsurv': 88,
    'Simple\nFusion': 68,
    'Flexible\nMM': 348
}

# Update from loaded data
for model_name, result in all_results.items():
    if result['n_patients'] is not None:
        dataset_sizes[model_name] = result['n_patients']

os.makedirs('results/comprehensive_analysis', exist_ok=True)

# ============================================================
# 2. Figure 1: C-index Comparison (Enhanced)
# ============================================================

print("\n[2] Generating C-index comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

models = list(all_results.keys())
means = [all_results[m]['mean'] for m in models]
stds = [all_results[m]['std'] for m in models]

# Sort by performance
sorted_indices = np.argsort(means)[::-1]
models_sorted = [models[i] for i in sorted_indices]
means_sorted = [means[i] for i in sorted_indices]
stds_sorted = [stds[i] for i in sorted_indices]

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))

# (A) Bar plot with error bars
bars = axes[0, 0].barh(models_sorted, means_sorted, xerr=stds_sorted,
                        color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, mean, std) in enumerate(zip(bars, means_sorted, stds_sorted)):
    axes[0, 0].text(mean + std + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{mean:.4f}\n±{std:.4f}',
                    va='center', fontsize=10, fontweight='bold')

axes[0, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random')
axes[0, 0].axvline(x=0.6, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good')
axes[0, 0].set_xlabel('C-index', fontsize=13, fontweight='bold')
axes[0, 0].set_title('(A) Model Performance Ranking', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(alpha=0.3, axis='x')
axes[0, 0].set_xlim([0.45, max(means_sorted) + max(stds_sorted) + 0.05])

# (B) Violin plot with individual points
fold_data = []
model_labels = []
for model in models_sorted:
    fold_data.extend(all_results[model]['fold_values'])
    model_labels.extend([model] * len(all_results[model]['fold_values']))

df_violin = pd.DataFrame({'Model': model_labels, 'C-index': fold_data})

parts = axes[0, 1].violinplot([all_results[m]['fold_values'] for m in models_sorted],
                               positions=range(len(models_sorted)),
                               vert=False,
                               showmeans=True,
                               showextrema=True,
                               widths=0.7)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Overlay scatter
for i, model in enumerate(models_sorted):
    y_pos = [i] * len(all_results[model]['fold_values'])
    y_jitter = np.random.normal(0, 0.04, len(y_pos))
    axes[0, 1].scatter(all_results[model]['fold_values'],
                       np.array(y_pos) + y_jitter,
                       alpha=0.8, s=50, color='black', edgecolor='white', linewidth=1)

axes[0, 1].set_yticks(range(len(models_sorted)))
axes[0, 1].set_yticklabels(models_sorted)
axes[0, 1].set_xlabel('C-index', fontsize=13, fontweight='bold')
axes[0, 1].set_title('(B) Distribution Across Folds', fontsize=14, fontweight='bold')
axes[0, 1].axvline(x=0.6, color='green', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 1].grid(alpha=0.3, axis='x')

# (C) Coefficient of Variation (Stability)
cv_values = [(all_results[m]['std'] / all_results[m]['mean']) * 100 for m in models_sorted]

bars_cv = axes[1, 0].barh(models_sorted, cv_values, color=colors, alpha=0.8,
                           edgecolor='black', linewidth=1.5)

for bar, cv in zip(bars_cv, cv_values):
    axes[1, 0].text(cv + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{cv:.2f}%',
                    va='center', fontsize=10, fontweight='bold')

axes[1, 0].set_xlabel('Coefficient of Variation (%)', fontsize=13, fontweight='bold')
axes[1, 0].set_title('(C) Model Stability (Lower = More Stable)', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='x')
axes[1, 0].invert_yaxis()

# (D) 95% Confidence Intervals
ci_lower = [all_results[m]['mean'] - 1.96 * all_results[m]['std'] for m in models_sorted]
ci_upper = [all_results[m]['mean'] + 1.96 * all_results[m]['std'] for m in models_sorted]

y_pos = np.arange(len(models_sorted))
# Create error bars as 2xN array: [lower_errors, upper_errors]
xerr_lower = [means_sorted[i] - ci_lower[i] for i in range(len(models_sorted))]
xerr_upper = [ci_upper[i] - means_sorted[i] for i in range(len(models_sorted))]
xerr = np.array([xerr_lower, xerr_upper])  # Shape should be (2, n)

axes[1, 1].errorbar(means_sorted, y_pos,
                    xerr=xerr.reshape(2, len(models_sorted)),
                    fmt='o', markersize=10, linewidth=2, capsize=8, capthick=2,
                    color='black', elinewidth=3, alpha=0.8)

axes[1, 1].set_yticks(y_pos)
axes[1, 1].set_yticklabels(models_sorted)
axes[1, 1].set_xlabel('C-index', fontsize=13, fontweight='bold')
axes[1, 1].set_title('(D) 95% Confidence Intervals', fontsize=14, fontweight='bold')
axes[1, 1].axvline(x=0.6, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (0.6)')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/comprehensive_analysis/01_cindex_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 01_cindex_comparison.png")
plt.close()

# ============================================================
# 3. Figure 2: Statistical Significance Matrix
# ============================================================

print("\n[3] Generating statistical significance matrix...")

n_models = len(models)
p_matrix = np.ones((n_models, n_models))

for i, model_i in enumerate(models):
    for j, model_j in enumerate(models):
        if i != j and len(all_results[model_i]['fold_values']) == len(all_results[model_j]['fold_values']):
            t_stat, p_value = stats.ttest_rel(all_results[model_i]['fold_values'],
                                              all_results[model_j]['fold_values'])
            p_matrix[i, j] = p_value

fig, ax = plt.subplots(figsize=(12, 10))

# Create significance annotations
annot = np.empty((n_models, n_models), dtype=object)
for i in range(n_models):
    for j in range(n_models):
        if i == j:
            annot[i, j] = '—'
        else:
            p = p_matrix[i, j]
            if p < 0.001:
                annot[i, j] = '***'
            elif p < 0.01:
                annot[i, j] = '**'
            elif p < 0.05:
                annot[i, j] = '*'
            else:
                annot[i, j] = 'ns'

# Heatmap
sns.heatmap(p_matrix, annot=annot, fmt='', cmap='RdYlGn_r',
            xticklabels=models, yticklabels=models,
            cbar_kws={'label': 'p-value'},
            vmin=0, vmax=0.1, ax=ax, linewidths=1, linecolor='white')

ax.set_title('Statistical Significance Matrix (Paired t-test)\n*** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05',
             fontsize=14, fontweight='bold', pad=20)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('results/comprehensive_analysis/02_significance_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ 02_significance_matrix.png")
plt.close()

# ============================================================
# 4. Figure 3: Efficiency Analysis
# ============================================================

print("\n[4] Generating efficiency analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (A) Performance vs Dataset Size
sizes = [dataset_sizes.get(m, 0) for m in models]
performance = means

scatter = axes[0].scatter(sizes, performance, s=300, c=means, cmap='RdYlGn',
                          alpha=0.8, edgecolor='black', linewidth=2)

for i, model in enumerate(models):
    axes[0].annotate(model, (sizes[i], performance[i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

axes[0].set_xlabel('Dataset Size (# Patients)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('C-index', fontsize=13, fontweight='bold')
axes[0].set_title('(A) Performance vs Dataset Size', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='C-index')

# (B) Efficiency Score (C-index / log(dataset_size))
efficiency = [means[i] / np.log10(sizes[i] + 1) if sizes[i] > 0 else 0
              for i in range(len(models))]

sorted_eff_indices = np.argsort(efficiency)[::-1]
models_eff_sorted = [models[i] for i in sorted_eff_indices]
efficiency_sorted = [efficiency[i] for i in sorted_eff_indices]

bars_eff = axes[1].barh(models_eff_sorted, efficiency_sorted,
                        color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models))),
                        alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, eff in zip(bars_eff, efficiency_sorted):
    axes[1].text(eff + 0.005, bar.get_y() + bar.get_height()/2,
                f'{eff:.4f}',
                va='center', fontsize=10, fontweight='bold')

axes[1].set_xlabel('Efficiency Score (C-index / log₁₀(N))', fontsize=13, fontweight='bold')
axes[1].set_title('(B) Data Efficiency Ranking', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/comprehensive_analysis/03_efficiency_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ 03_efficiency_analysis.png")
plt.close()

# ============================================================
# 5. Figure 4: Model Complexity Analysis
# ============================================================

print("\n[5] Generating complexity analysis...")

# Define complexity scores (1-5 scale)
complexity_scores = {
    'Image-Only': 3,       # DenseNet121
    'RNA-Only': 1,         # Simple MLP
    'Partial\nModality': 3,  # Gating network
    'SimMLM': 5,           # DMoME + MoFe, two-stage
    'MMsurv': 4,           # CBP + Transformer
    'Simple\nFusion': 2,   # Late fusion
    'Flexible\nMM': 2      # Late fusion + learnable bias
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (A) Complexity vs Performance
complexity = [complexity_scores.get(m, 0) for m in models]

scatter2 = axes[0].scatter(complexity, performance, s=300, c=means, cmap='RdYlGn',
                           alpha=0.8, edgecolor='black', linewidth=2)

for i, model in enumerate(models):
    axes[0].annotate(model, (complexity[i], performance[i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

axes[0].set_xlabel('Model Complexity (1=Simple, 5=Very Complex)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('C-index', fontsize=13, fontweight='bold')
axes[0].set_title('(A) Complexity vs Performance\n(Simple Models Preferred)', fontsize=14, fontweight='bold')
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_xticklabels(['Simple', 'Low', 'Medium', 'High', 'Very High'])
axes[0].axhline(y=0.6, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (0.6)')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
plt.colorbar(scatter2, ax=axes[0], label='C-index')

# (B) Performance/Complexity Ratio
perf_complexity_ratio = [means[i] / (complexity[i] + 0.1) for i in range(len(models))]

sorted_pc_indices = np.argsort(perf_complexity_ratio)[::-1]
models_pc_sorted = [models[i] for i in sorted_pc_indices]
pc_ratio_sorted = [perf_complexity_ratio[i] for i in sorted_pc_indices]

bars_pc = axes[1].barh(models_pc_sorted, pc_ratio_sorted,
                       color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models))),
                       alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, ratio in zip(bars_pc, pc_ratio_sorted):
    axes[1].text(ratio + 0.005, bar.get_y() + bar.get_height()/2,
                f'{ratio:.4f}',
                va='center', fontsize=10, fontweight='bold')

axes[1].set_xlabel('Performance/Complexity Ratio', fontsize=13, fontweight='bold')
axes[1].set_title('(B) Best Performance per Unit Complexity', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/comprehensive_analysis/04_complexity_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ 04_complexity_analysis.png")
plt.close()

# ============================================================
# 6. Figure 5: Fold-wise Performance
# ============================================================

print("\n[6] Generating fold-wise performance analysis...")

fig, ax = plt.subplots(figsize=(14, 8))

n_folds = max([len(all_results[m]['fold_values']) for m in models])
x = np.arange(n_folds)
width = 0.1

for i, model in enumerate(models):
    fold_vals = all_results[model]['fold_values']
    # Pad if fewer folds
    while len(fold_vals) < n_folds:
        fold_vals.append(0)

    offset = (i - len(models)/2) * width
    bars = ax.bar(x + offset, fold_vals, width, label=model, alpha=0.8)

ax.set_xlabel('Fold', fontsize=13, fontweight='bold')
ax.set_ylabel('C-index', fontsize=13, fontweight='bold')
ax.set_title('Fold-wise Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
ax.axhline(y=0.6, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (0.6)')
ax.legend(fontsize=10, ncol=2, loc='lower right')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/comprehensive_analysis/05_fold_performance.png', dpi=300, bbox_inches='tight')
print("  ✓ 05_fold_performance.png")
plt.close()

# ============================================================
# 7. Summary Statistics Table
# ============================================================

print("\n[7] Generating summary table...")

summary_data = []
for model in models_sorted:
    summary_data.append({
        'Model': model.replace('\n', ' '),
        'Mean C-index': f"{all_results[model]['mean']:.4f}",
        'Std': f"{all_results[model]['std']:.4f}",
        'CV (%)': f"{(all_results[model]['std'] / all_results[model]['mean']) * 100:.2f}",
        'Min': f"{min(all_results[model]['fold_values']):.4f}",
        'Max': f"{max(all_results[model]['fold_values']):.4f}",
        'N Patients': dataset_sizes.get(model, 'N/A'),
        'Complexity': complexity_scores.get(model, 'N/A')
    })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('results/comprehensive_analysis/summary_statistics.csv', index=False)
print("  ✓ summary_statistics.csv")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 80)
print("Comprehensive Analysis Complete!")
print("=" * 80)
print("\n📁 Generated Files:")
print("  1. 01_cindex_comparison.png - 4-panel C-index analysis")
print("  2. 02_significance_matrix.png - Statistical significance heatmap")
print("  3. 03_efficiency_analysis.png - Performance vs dataset size")
print("  4. 04_complexity_analysis.png - Model complexity trade-offs")
print("  5. 05_fold_performance.png - Fold-wise comparison")
print("  6. summary_statistics.csv - Detailed statistics table")
print("\n📊 Key Insights:")
print(f"  - Best model: {models_sorted[0].replace(chr(10), ' ')} ({means_sorted[0]:.4f})")
print(f"  - Most stable: {models_sorted[np.argmin(cv_values)].replace(chr(10), ' ')} (CV: {min(cv_values):.2f}%)")
print(f"  - Most efficient: {models_eff_sorted[0].replace(chr(10), ' ')} (Efficiency: {efficiency_sorted[0]:.4f})")
print("=" * 80)
