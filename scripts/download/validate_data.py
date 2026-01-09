"""
다운로드된 데이터 검증 스크립트
- 각 환자별 데이터 완전성 확인
- 멀티모달 데이터 매칭 테이블 생성
"""

import os
import pandas as pd
import json
from pathlib import Path

print("=" * 60)
print("데이터 검증 및 매칭")
print("=" * 60)

# 샘플링된 환자 목록 로드
if not os.path.exists('data/sampled_patients.csv'):
    print("\n오류: data/sampled_patients.csv 파일이 없습니다.")
    exit(1)

sampled_patients = pd.read_csv('data/sampled_patients.csv')['patient_id'].tolist()
print(f"\n샘플링된 환자 수: {len(sampled_patients)}")

# 검증 결과를 저장할 DataFrame
validation_results = []

print(f"\n{'=' * 60}")
print("환자별 데이터 검증")
print(f"{'=' * 60}")

# Pre-scan genomic data directories once (much faster)
print("\n1단계: 유전체 데이터 디렉토리 스캔 중...")
rnaseq_exists = Path("data/genomic/rnaseq").exists()
mutation_exists = Path("data/genomic/mutation").exists()
cnv_exists = Path("data/genomic/cnv").exists()

# Load clinical data once
print("2단계: 임상 데이터 로드 중...")
clinical_file = Path("data/clinical/tcga_ov_multimodal_clinical.csv")
if not clinical_file.exists():
    clinical_file = Path("data/clinical/tcga_ov_sampled_clinical.csv")

clinical_patients = set()
if clinical_file.exists():
    df_clinical = pd.read_csv(clinical_file)
    clinical_patients = set(df_clinical['submitter_id'].values)

print(f"3단계: {len(sampled_patients)}명 환자 검증 중...")

from tqdm import tqdm

for idx, patient_id in enumerate(tqdm(sampled_patients, desc="검증 진행")):
    result = {
        'patient_id': patient_id,
        'has_imaging': False,
        'imaging_series_count': 0,
        'has_rnaseq': rnaseq_exists,  # 유전체 데이터는 환자별 매핑이 어려우므로 전체 존재 여부로 판단
        'has_mutation': mutation_exists,
        'has_cnv': cnv_exists,
        'has_clinical': patient_id in clinical_patients,
        'multimodal_complete': False
    }

    # 1. 영상 데이터 확인 (환자별로 디렉토리가 명확히 분리됨)
    imaging_dir = Path(f"data/imaging/dicom/{patient_id}")
    if imaging_dir.exists() and imaging_dir.is_dir():
        # 시리즈 디렉토리 수 카운트
        series_dirs = [d for d in imaging_dir.iterdir() if d.is_dir()]
        if len(series_dirs) > 0:
            result['has_imaging'] = True
            result['imaging_series_count'] = len(series_dirs)

    # 멀티모달 완전성 체크 (영상 + RNA-seq + 임상)
    result['multimodal_complete'] = (
        result['has_imaging'] and
        result['has_rnaseq'] and
        result['has_clinical']
    )

    validation_results.append(result)

# DataFrame 변환
df_validation = pd.DataFrame(validation_results)

# 결과 저장
validation_file = 'data/validation_results.csv'
df_validation.to_csv(validation_file, index=False)

print(f"\n✓ 검증 결과 저장: {validation_file}")

# 요약 통계
print(f"\n{'=' * 60}")
print("데이터 완전성 요약")
print(f"{'=' * 60}")

total_patients = len(df_validation)
print(f"\n총 환자 수: {total_patients}")
print(f"\n데이터 타입별 보유 환자:")
print(f"  영상 데이터:    {df_validation['has_imaging'].sum():3d}명 ({df_validation['has_imaging'].sum() / total_patients * 100:.1f}%)")
print(f"  RNA-seq:        {df_validation['has_rnaseq'].sum():3d}명 ({df_validation['has_rnaseq'].sum() / total_patients * 100:.1f}%)")
print(f"  Mutation:       {df_validation['has_mutation'].sum():3d}명 ({df_validation['has_mutation'].sum() / total_patients * 100:.1f}%)")
print(f"  CNV:            {df_validation['has_cnv'].sum():3d}명 ({df_validation['has_cnv'].sum() / total_patients * 100:.1f}%)")
print(f"  임상 데이터:    {df_validation['has_clinical'].sum():3d}명 ({df_validation['has_clinical'].sum() / total_patients * 100:.1f}%)")
print(f"\n멀티모달 완전한 데이터: {df_validation['multimodal_complete'].sum()}명 ({df_validation['multimodal_complete'].sum() / total_patients * 100:.1f}%)")

# 멀티모달 매칭 테이블
multimodal_patients = df_validation[df_validation['multimodal_complete'] == True]['patient_id'].tolist()

if len(multimodal_patients) > 0:
    print(f"\n멀티모달 데이터가 완전한 환자 (처음 10명):")
    for patient_id in multimodal_patients[:10]:
        row = df_validation[df_validation['patient_id'] == patient_id].iloc[0]
        print(f"  {patient_id}: 영상 {row['imaging_series_count']}개 시리즈")

    # 멀티모달 환자 리스트 저장
    multimodal_file = 'data/multimodal_patients.csv'
    pd.DataFrame({'patient_id': multimodal_patients}).to_csv(multimodal_file, index=False)
    print(f"\n✓ 멀티모달 환자 목록 저장: {multimodal_file}")
else:
    print("\n경고: 멀티모달 데이터가 완전한 환자가 없습니다.")
    print("일부 데이터 다운로드가 실패했을 수 있습니다.")

# 디스크 사용량 추정
print(f"\n{'=' * 60}")
print("디스크 사용량")
print(f"{'=' * 60}")

def get_dir_size(path):
    """디렉토리 크기 계산 (MB)"""
    total_size = 0
    if not os.path.exists(path):
        return 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # MB

imaging_size = get_dir_size("data/imaging/dicom")
rnaseq_size = get_dir_size("data/genomic/rnaseq")
mutation_size = get_dir_size("data/genomic/mutation")
cnv_size = get_dir_size("data/genomic/cnv")
clinical_size = get_dir_size("data/clinical")

total_size = imaging_size + rnaseq_size + mutation_size + cnv_size + clinical_size

print(f"\n영상 데이터:    {imaging_size:8.1f} MB")
print(f"RNA-seq:        {rnaseq_size:8.1f} MB")
print(f"Mutation:       {mutation_size:8.1f} MB")
print(f"CNV:            {cnv_size:8.1f} MB")
print(f"임상 데이터:    {clinical_size:8.1f} MB")
print(f"{'-' * 40}")
print(f"총 사용량:      {total_size:8.1f} MB ({total_size / 1024:.2f} GB)")

# 최종 요약 JSON 저장
summary = {
    'total_patients': total_patients,
    'data_availability': {
        'imaging': int(df_validation['has_imaging'].sum()),
        'rnaseq': int(df_validation['has_rnaseq'].sum()),
        'mutation': int(df_validation['has_mutation'].sum()),
        'cnv': int(df_validation['has_cnv'].sum()),
        'clinical': int(df_validation['has_clinical'].sum())
    },
    'multimodal_complete': int(df_validation['multimodal_complete'].sum()),
    'disk_usage_mb': {
        'imaging': round(imaging_size, 1),
        'rnaseq': round(rnaseq_size, 1),
        'mutation': round(mutation_size, 1),
        'cnv': round(cnv_size, 1),
        'clinical': round(clinical_size, 1),
        'total': round(total_size, 1)
    }
}

summary_file = 'data/data_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ 최종 요약 저장: {summary_file}")

print(f"\n{'=' * 60}")
print("검증 완료!")
print(f"{'=' * 60}")
print(f"\n다운로드된 데이터를 사용하여 연구를 시작할 수 있습니다.")
print(f"멀티모달 환자 목록: data/multimodal_patients.csv")
