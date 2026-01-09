"""
전체 608명 환자의 통합 Matching Table 생성
- 부분 모달리티 지원
- Age, Survival 라벨 포함
"""

import pandas as pd
import os
from pathlib import Path

print("=" * 60)
print("Full Matching Table 생성 (608명)")
print("=" * 60)

# ============================================================
# Clinical 데이터 로드
# ============================================================

clinical_file = 'data/clinical/tcga_ov_multimodal_clinical.csv'
df_clinical = pd.read_csv(clinical_file)

print(f"\nClinical 데이터: {len(df_clinical)}명")
print(f"컬럼: {list(df_clinical.columns)}")

# ============================================================
# Age 추출
# ============================================================

# Age at index (years)
df_clinical['age'] = df_clinical['demographic.age_at_index']

# 또는 days_to_birth로부터 계산
if df_clinical['age'].isna().any():
    # days_to_birth는 음수 (태어난 날로부터 진단까지 일수)
    df_clinical['age'] = df_clinical['age'].fillna(
        -df_clinical['demographic.days_to_birth'] / 365.25
    )

print(f"\nAge 통계:")
print(f"  평균: {df_clinical['age'].mean():.1f}")
print(f"  범위: {df_clinical['age'].min():.1f} ~ {df_clinical['age'].max():.1f}")
print(f"  결측: {df_clinical['age'].isna().sum()}개")

# ============================================================
# Survival 라벨 추출
# ============================================================

# Vital status
df_clinical['vital_status'] = df_clinical['demographic.vital_status']

# Days to death
df_clinical['days_to_death'] = df_clinical['demographic.days_to_death']

# Survival time and status
df_clinical['survival_time'] = df_clinical['days_to_death']
df_clinical['survival_status'] = (df_clinical['vital_status'] == 'Dead').astype(int)

# Censored 환자는 days_to_death가 NaN -> 마지막 follow-up으로 대체해야 함
# 여기서는 일단 NaN으로 남김 (필요시 추가 처리)

print(f"\nSurvival 통계:")
print(f"  생존 시간 있음: {df_clinical['survival_time'].notna().sum()}명")
print(f"  Event (death): {df_clinical['survival_status'].sum()}명")
print(f"  Censored: {(df_clinical['survival_status'] == 0).sum()}명")

# ============================================================
# Imaging 경로 확인
# ============================================================

imaging_dir = Path('data/imaging/nifti')
patient_imaging = {}

if imaging_dir.exists():
    for patient_dir in imaging_dir.iterdir():
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            # 첫 번째 series 찾기
            nifti_files = list(patient_dir.glob('*.nii.gz'))
            if nifti_files:
                patient_imaging[patient_id] = str(nifti_files[0])

print(f"\nImaging 데이터:")
print(f"  환자 수: {len(patient_imaging)}명")

# ============================================================
# RNA-seq 확인
# ============================================================

rnaseq_file = 'data/processed/rnaseq_normalized_mapped.csv'
if os.path.exists(rnaseq_file):
    df_rna = pd.read_csv(rnaseq_file, index_col=0)
    rnaseq_patients = set(df_rna.index)
    print(f"\nRNA-seq 데이터:")
    print(f"  환자 수: {len(rnaseq_patients)}명")
else:
    rnaseq_patients = set()
    print(f"\n경고: RNA-seq 파일 없음")

# ============================================================
# Matching Table 생성
# ============================================================

matching_data = []

for idx, row in df_clinical.iterrows():
    patient_id = row['submitter_id']

    # Imaging
    has_imaging = patient_id in patient_imaging
    nifti_path = patient_imaging.get(patient_id, None)

    # RNA-seq
    has_rnaseq = patient_id in rnaseq_patients

    # Clinical (age)
    has_clinical = pd.notna(row['age'])
    age = row['age'] if has_clinical else None

    # Survival
    has_survival = pd.notna(row['survival_time'])
    survival_time = row['survival_time'] if has_survival else None
    survival_status = row['survival_status']

    matching_data.append({
        'patient_id': patient_id,
        'nifti_path': nifti_path,
        'has_imaging': has_imaging,
        'has_rnaseq': has_rnaseq,
        'has_clinical': has_clinical,
        'age': age,
        'survival_time': survival_time,
        'survival_status': survival_status,
        'has_survival': has_survival
    })

df_matching = pd.DataFrame(matching_data)

# ============================================================
# 통계
# ============================================================

print(f"\n{'=' * 60}")
print("Matching Table 통계")
print(f"{'=' * 60}")

print(f"\n전체 환자: {len(df_matching)}명")
print(f"\n모달리티 분포:")
print(f"  Imaging: {df_matching['has_imaging'].sum()}명")
print(f"  RNA-seq: {df_matching['has_rnaseq'].sum()}명")
print(f"  Clinical (age): {df_matching['has_clinical'].sum()}명")
print(f"  Survival 라벨: {df_matching['has_survival'].sum()}명")

print(f"\n멀티모달 조합:")
print(f"  Imaging + RNA-seq: {((df_matching['has_imaging']) & (df_matching['has_rnaseq'])).sum()}명")
print(f"  Imaging + Clinical: {((df_matching['has_imaging']) & (df_matching['has_clinical'])).sum()}명")
print(f"  RNA-seq + Clinical: {((df_matching['has_rnaseq']) & (df_matching['has_clinical'])).sum()}명")
print(f"  완전 멀티모달 (3개 모두): {((df_matching['has_imaging']) & (df_matching['has_rnaseq']) & (df_matching['has_clinical'])).sum()}명")

print(f"\n생존 라벨 있는 환자:")
print(f"  전체: {df_matching['has_survival'].sum()}명")
print(f"  Imaging 있음: {((df_matching['has_survival']) & (df_matching['has_imaging'])).sum()}명")
print(f"  RNA-seq 있음: {((df_matching['has_survival']) & (df_matching['has_rnaseq'])).sum()}명")
print(f"  완전 멀티모달: {((df_matching['has_survival']) & (df_matching['has_imaging']) & (df_matching['has_rnaseq']) & (df_matching['has_clinical'])).sum()}명")

# ============================================================
# 저장
# ============================================================

output_file = 'data/processed/full_matching_table.csv'
df_matching.to_csv(output_file, index=False)

print(f"\n{'=' * 60}")
print(f"저장 완료: {output_file}")
print(f"  Shape: {df_matching.shape}")
print(f"{'=' * 60}")
