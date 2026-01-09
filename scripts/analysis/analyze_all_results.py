"""
전체 모델 결과 분석 및 시각화
- CV 결과 비교
- 통계적 유의성 검증
- 시각화: Box plot, Bar chart, Fold별 비교
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

print("=" * 80)
print("전체 모델 결과 분석")
print("=" * 80)

# ============================================================
# 1. 결과 로드
# ============================================================

results_dir = 'results'
models = {
    'Image-Only': 'image_only',
    'Multimodal\n(Complete)': 'multimodal',
    'Partial\nModality': 'partial_modality',
    'SimMLM': 'simmim'
}

data = {}
for model_name, folder in models.items():
    json_path = f'{results_dir}/{folder}/cv_results.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data[model_name] = json.load(f)
        print(f"✓ {model_name:25s} 로드 완료")
    else:
        print(f"✗ {model_name:25s} 결과 없음")

print()

# ============================================================
# 2. 통계 요약
# ============================================================

print("=" * 80)
print("C-index 통계 요약")
print("=" * 80)

summary_data = []

for model_name, result in data.items():
    if 'c_index_mean' in result:
        mean = result['c_index_mean']
        std = result['c_index_std']
    else:
        # Image-only 형식
        c_indices = [f['best_c_index'] for f in result['fold_results']]
        mean = np.mean(c_indices)
        std = np.std(c_indices)

    fold_results = [f['best_c_index'] for f in result['fold_results']]

    summary_data.append({
        'Model': model_name,
        'Mean': mean,
        'Std': std,
        'Min': np.min(fold_results),
        'Max': np.max(fold_results),
        'Median': np.median(fold_results),
        'CV (%)': (std / mean * 100) if mean > 0 else 0
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Mean', ascending=False)

print()
print(summary_df.to_string(index=False))
print()

# ============================================================
# 3. Fold별 상세 결과
# ============================================================

print("=" * 80)
print("Fold별 C-index")
print("=" * 80)

fold_data = []
for model_name, result in data.items():
    for fold_result in result['fold_results']:
        fold_data.append({
            'Model': model_name,
            'Fold': fold_result['fold'],
            'C-index': fold_result['best_c_index']
        })

fold_df = pd.DataFrame(fold_data)
fold_pivot = fold_df.pivot(index='Fold', columns='Model', values='C-index')

print()
print(fold_pivot.to_string())
print()

# ============================================================
# 4. 통계적 유의성 검정 (Paired t-test)
# ============================================================

print("=" * 80)
print("통계적 유의성 검정 (Paired t-test)")
print("=" * 80)

model_names = list(data.keys())
model_c_indices = {}

for model_name, result in data.items():
    model_c_indices[model_name] = [f['best_c_index'] for f in result['fold_results']]

print()
print("P-value Matrix (낮을수록 유의미한 차이):")
print("-" * 80)

p_value_matrix = []
for i, model1 in enumerate(model_names):
    row = []
    for j, model2 in enumerate(model_names):
        if i == j:
            row.append('-')
        else:
            c1 = model_c_indices[model1]
            c2 = model_c_indices[model2]

            # Paired t-test (같은 fold끼리 비교)
            t_stat, p_value = stats.ttest_rel(c1, c2)

            if p_value < 0.001:
                row.append(f'{p_value:.4f}***')
            elif p_value < 0.01:
                row.append(f'{p_value:.4f}**')
            elif p_value < 0.05:
                row.append(f'{p_value:.4f}*')
            else:
                row.append(f'{p_value:.4f}')
    p_value_matrix.append(row)

p_df = pd.DataFrame(p_value_matrix, index=model_names, columns=model_names)
print(p_df.to_string())
print()
print("***: p<0.001, **: p<0.01, *: p<0.05")
print()

# ============================================================
# 5. 시각화
# ============================================================

print("=" * 80)
print("시각화 생성 중...")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(16, 12))

# ============================================================
# Plot 1: Box Plot (분포 비교)
# ============================================================

ax1 = plt.subplot(2, 3, 1)

box_data = []
box_labels = []
for model_name in model_names:
    box_data.append(model_c_indices[model_name])
    box_labels.append(model_name)

bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                  notch=True, showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

# Color boxes
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_ylabel('C-index', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Distribution', fontsize=14, fontweight='bold')
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.5)')
ax1.axhline(y=0.6, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (0.6)')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')

# ============================================================
# Plot 2: Bar Chart with Error Bars
# ============================================================

ax2 = plt.subplot(2, 3, 2)

means = [summary_df[summary_df['Model'] == m]['Mean'].values[0] for m in model_names]
stds = [summary_df[summary_df['Model'] == m]['Std'].values[0] for m in model_names]

x_pos = np.arange(len(model_names))
bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
               color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
             f'{mean:.4f}\n±{std:.4f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names)
ax2.set_ylabel('C-index', fontsize=12, fontweight='bold')
ax2.set_title('Mean C-index with Std Dev', fontsize=14, fontweight='bold')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(y=0.6, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_ylim([0.4, 0.7])
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

# ============================================================
# Plot 3: Fold별 성능 비교 (Line Plot)
# ============================================================

ax3 = plt.subplot(2, 3, 3)

for i, model_name in enumerate(model_names):
    c_indices = model_c_indices[model_name]
    folds = list(range(1, len(c_indices) + 1))
    ax3.plot(folds, c_indices, marker='o', linewidth=2, markersize=8,
             label=model_name, color=plt.cm.tab10(i))

ax3.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax3.set_ylabel('C-index', fontsize=12, fontweight='bold')
ax3.set_title('C-index Across Folds', fontsize=14, fontweight='bold')
ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.axhline(y=0.6, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, 6))
ax3.set_ylim([0.4, 0.7])

# ============================================================
# Plot 4: Violin Plot (분포 상세)
# ============================================================

ax4 = plt.subplot(2, 3, 4)

# Prepare data for violin plot
violin_data = []
violin_positions = []
for i, model_name in enumerate(model_names):
    for c_index in model_c_indices[model_name]:
        violin_data.append({'Model': model_name, 'C-index': c_index})

violin_df = pd.DataFrame(violin_data)

sns.violinplot(data=violin_df, x='Model', y='C-index', ax=ax4, palette=colors, inner='box')
ax4.set_ylabel('C-index', fontsize=12, fontweight='bold')
ax4.set_xlabel('')
ax4.set_title('Distribution Comparison (Violin Plot)', fontsize=14, fontweight='bold')
ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.axhline(y=0.6, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax4.grid(True, alpha=0.3, axis='y')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')

# ============================================================
# Plot 5: Improvement over Baseline
# ============================================================

ax5 = plt.subplot(2, 3, 5)

baseline_name = 'Image-Only'
baseline_mean = summary_df[summary_df['Model'] == baseline_name]['Mean'].values[0]

improvements = []
improvement_labels = []
improvement_colors = []

for model_name in model_names:
    if model_name == baseline_name:
        continue

    model_mean = summary_df[summary_df['Model'] == model_name]['Mean'].values[0]
    improvement = ((model_mean - baseline_mean) / baseline_mean) * 100
    improvements.append(improvement)
    improvement_labels.append(model_name)
    improvement_colors.append('green' if improvement > 0 else 'red')

bars = ax5.barh(improvement_labels, improvements, color=improvement_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    width = bar.get_width()
    ax5.text(width + 0.3 if width > 0 else width - 0.3,
             bar.get_y() + bar.get_height()/2.,
             f'{imp:+.2f}%',
             ha='left' if width > 0 else 'right',
             va='center', fontsize=10, fontweight='bold')

ax5.axvline(x=0, color='black', linewidth=2)
ax5.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
ax5.set_title(f'Relative Improvement over {baseline_name}', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# ============================================================
# Plot 6: Statistical Summary Table
# ============================================================

ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary table
table_data = []
for model_name in model_names:
    row_data = summary_df[summary_df['Model'] == model_name].iloc[0]
    table_data.append([
        model_name.replace('\n', ' '),
        f"{row_data['Mean']:.4f}",
        f"{row_data['Std']:.4f}",
        f"[{row_data['Min']:.3f}, {row_data['Max']:.3f}]",
        f"{row_data['CV (%)']:.2f}%"
    ])

table = ax6.table(cellText=table_data,
                  colLabels=['Model', 'Mean', 'Std', 'Range', 'CV'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(table_data) + 1):
    for j in range(5):
        table[(i, j)].set_facecolor(colors[(i-1) % len(colors)])

ax6.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)

# ============================================================
# Save figure
# ============================================================

plt.tight_layout()
output_path = 'results/model_comparison_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ 시각화 저장: {output_path}")

# ============================================================
# 6. 랭킹 및 권장사항
# ============================================================

print("\n" + "=" * 80)
print("모델 랭킹 및 권장사항")
print("=" * 80)

print("\n📊 성능 랭킹 (C-index 기준):")
for i, (idx, row) in enumerate(summary_df.iterrows(), 1):
    print(f"  {i}. {row['Model']:25s} - {row['Mean']:.4f} ± {row['Std']:.4f}")

print("\n🎯 핵심 발견:")
print(f"  • 최고 성능: {summary_df.iloc[0]['Model']} (C-index: {summary_df.iloc[0]['Mean']:.4f})")
print(f"  • 가장 안정적: {summary_df.sort_values('Std').iloc[0]['Model']} (Std: {summary_df.sort_values('Std').iloc[0]['Std']:.4f})")
print(f"  • 베이스라인 대비 최대 향상: {improvements[np.argmax(improvements)]:+.2f}% ({improvement_labels[np.argmax(improvements)]})")

print("\n💡 권장사항:")
best_model = summary_df.iloc[0]['Model']
best_c_index = summary_df.iloc[0]['Mean']

if best_c_index >= 0.70:
    print("  ✅ Excellent: C-index ≥ 0.70 달성! 논문 수준의 성능입니다.")
elif best_c_index >= 0.65:
    print("  ✅ Very Good: C-index ≥ 0.65 달성! 추가 튜닝으로 0.70+ 가능합니다.")
elif best_c_index >= 0.60:
    print("  ✅ Good: C-index ≥ 0.60 달성! 추가 개선 여지가 있습니다.")
elif best_c_index >= 0.55:
    print("  ⚠️  Fair: C-index ≥ 0.55. Hyperparameter 튜닝을 권장합니다.")
else:
    print("  ⚠️  Needs improvement: C-index < 0.55. 모델/데이터 재검토 필요.")

print(f"\n  • 권장 모델: {best_model}")
print(f"  • 다음 단계:")
print(f"    1. {best_model} 모델로 Hyperparameter tuning")
print(f"    2. Ensemble methods (여러 fold 모델 결합)")
print(f"    3. Feature importance 분석")
print(f"    4. External validation")

# ============================================================
# 7. 상세 보고서 저장
# ============================================================

report_path = 'results/analysis_report.txt'
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("모델 성능 분석 보고서\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. C-index 통계 요약\n")
    f.write("-" * 80 + "\n")
    f.write(summary_df.to_string(index=False) + "\n\n")

    f.write("2. Fold별 C-index\n")
    f.write("-" * 80 + "\n")
    f.write(fold_pivot.to_string() + "\n\n")

    f.write("3. 통계적 유의성 (P-value Matrix)\n")
    f.write("-" * 80 + "\n")
    f.write(p_df.to_string() + "\n")
    f.write("***: p<0.001, **: p<0.01, *: p<0.05\n\n")

    f.write("4. 베이스라인 대비 개선율\n")
    f.write("-" * 80 + "\n")
    for label, imp in zip(improvement_labels, improvements):
        f.write(f"  {label:25s}: {imp:+.2f}%\n")
    f.write("\n")

    f.write("5. 결론 및 권장사항\n")
    f.write("-" * 80 + "\n")
    f.write(f"  최고 성능 모델: {best_model}\n")
    f.write(f"  C-index: {best_c_index:.4f}\n")

print(f"✓ 상세 보고서 저장: {report_path}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print(f"\n생성된 파일:")
print(f"  - {output_path}")
print(f"  - {report_path}")
print()
