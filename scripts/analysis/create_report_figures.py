"""
보고서용 핵심 시각화
- 데이터 샘플 이미지 (2-3개)
- 모달리티 분포
- 생존 곡선
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("보고서용 시각화 생성")
print("=" * 60)

os.makedirs('results/report_figures', exist_ok=True)

# ============================================================
# 1. 샘플 이미지 (3개)
# ============================================================

print("\n[1] 샘플 이미지 생성")

matching_table = pd.read_csv('data/processed/full_matching_table.csv')
imaging_patients = matching_table[matching_table['has_imaging'] == True]

if len(imaging_patients) >= 3:
    sample_patients = imaging_patients.sample(n=3, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (_, patient_row) in enumerate(sample_patients.iterrows()):
        patient_id = patient_row['patient_id']
        nifti_path = patient_row['nifti_path']

        if pd.notna(nifti_path) and os.path.exists(nifti_path):
            try:
                img = sitk.ReadImage(str(nifti_path))
                img_np = sitk.GetArrayFromImage(img)

                # 중간 슬라이스
                D, H, W = img_np.shape
                mid_slice = img_np[D // 2, :, :]

                axes[idx].imshow(mid_slice, cmap='gray')
                axes[idx].set_title(f'Patient {idx+1}\n{patient_id}\nShape: {img_np.shape}', fontsize=11)
                axes[idx].axis('off')

                print(f"  ✓ {patient_id}")

            except Exception as e:
                axes[idx].text(0.5, 0.5, 'Load Error', ha='center', va='center')
                axes[idx].axis('off')
                print(f"  ✗ {patient_id}: {e}")

    plt.tight_layout()
    plt.savefig('results/report_figures/sample_images.png', dpi=300, bbox_inches='tight')
    print("✓ 저장: results/report_figures/sample_images.png")
    plt.close()

# ============================================================
# 2. 데이터 분포 요약
# ============================================================

print("\n[2] 데이터 분포 차트")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (1) 모달리티 분포
modality_counts = {
    'Imaging': matching_table['has_imaging'].sum(),
    'RNA-seq': matching_table['has_rnaseq'].sum(),
    'Clinical': matching_table['has_clinical'].sum(),
    'Survival': matching_table['has_survival'].sum()
}

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = axes[0].bar(modality_counts.keys(), modality_counts.values(),
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# 값 표시
for bar, (name, count) in zip(bars, modality_counts.items()):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 10,
                 f'{count}\n({count/len(matching_table)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
axes[0].set_title('Data Availability by Modality', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, len(matching_table) * 1.15])
axes[0].grid(alpha=0.3, axis='y')

# (2) 생존 상태
survival_data = matching_table[matching_table['has_survival'] == True]
if len(survival_data) > 0:
    status_counts = survival_data['survival_status'].value_counts()

    labels = ['Alive (Censored)', 'Dead (Event)']
    sizes = [status_counts.get(0, 0), status_counts.get(1, 0)]
    colors_pie = ['lightgreen', 'lightcoral']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = axes[1].pie(sizes, labels=labels, autopct='%1.1f%%',
                                             colors=colors_pie, startangle=90, explode=explode,
                                             shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})

    axes[1].set_title('Survival Status Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('results/report_figures/data_distribution.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results/report_figures/data_distribution.png")
plt.close()

# ============================================================
# 3. Kaplan-Meier 곡선 (전체 코호트)
# ============================================================

print("\n[3] Kaplan-Meier 생존 곡선")

try:
    from lifelines import KaplanMeierFitter

    if len(survival_data) > 0:
        kmf = KaplanMeierFitter()

        # 전체 코호트
        kmf.fit(survival_data['survival_time'],
                survival_data['survival_status'],
                label='All Patients (n={})'.format(len(survival_data)))

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        kmf.plot_survival_function(ax=ax, ci_show=True, color='steelblue', linewidth=2.5)

        ax.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title('Kaplan-Meier Survival Curve\nTCGA-OV Cohort', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11, loc='best')

        # 통계 정보 추가
        median_survival = kmf.median_survival_time_
        textstr = f'Median Survival: {median_survival:.1f} days\nTotal Events: {survival_data["survival_status"].sum()}/{len(survival_data)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('results/report_figures/kaplan_meier_curve.png', dpi=300, bbox_inches='tight')
        print("✓ 저장: results/report_figures/kaplan_meier_curve.png")
        plt.close()

except ImportError:
    print("✗ lifelines 패키지 필요 (pip install lifelines)")

# ============================================================
# 4. 모델 성능 비교 (기존 결과 활용)
# ============================================================

print("\n[4] 모델 성능 비교")

results_files = {
    'Image-Only': 'results/image_only/cv_results.json',
    'Multimodal': 'results/multimodal/cv_results.json',
    'Partial\nModality': 'results/partial_modality/cv_results.json',
    'SimMLM': 'results/simmim/cv_results.json',
    'MMsurv': 'results/mmsurv/cv_results.json'
}

import json

model_results = {}
for model_name, filepath in results_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            if 'c_index_mean' in data:
                model_results[model_name] = {
                    'mean': data['c_index_mean'],
                    'std': data['c_index_std']
                }
            else:
                # Image-only 형식
                c_indices = [f['best_c_index'] for f in data['fold_results']]
                model_results[model_name] = {
                    'mean': np.mean(c_indices),
                    'std': np.std(c_indices)
                }

if model_results:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = list(model_results.keys())
    means = [model_results[m]['mean'] for m in models]
    stds = [model_results[m]['std'] for m in models]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(models, means, yerr=stds, capsize=10,
                   color=colors[:len(models)], alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # 값 표시
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('C-index', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (Cross-Validation)', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Random (0.5)')
    ax.axhline(y=0.6, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (0.6)')
    ax.set_ylim([0.4, max(means) + max(stds) + 0.1])
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3, axis='y')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('results/report_figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 저장: results/report_figures/model_comparison.png")
    plt.close()

# ============================================================
# 요약
# ============================================================

print("\n" + "=" * 60)
print("생성된 보고서용 시각화")
print("=" * 60)
print("\n📁 results/report_figures/")
print("  1. sample_images.png - 샘플 CT 이미지 (3개)")
print("  2. data_distribution.png - 모달리티 분포 & 생존 상태")
print("  3. kaplan_meier_curve.png - 전체 코호트 생존 곡선")
print("  4. model_comparison.png - 모델 성능 비교")

print("\n💡 이 파일들을 논문/보고서에 직접 사용할 수 있습니다.")
print("=" * 60)
