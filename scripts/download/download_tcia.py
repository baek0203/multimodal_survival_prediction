"""
TCIA-OV 영상 데이터 다운로드 스크립트
- 샘플링된 환자의 의료 영상만 다운로드
- tcia-utils 사용
"""

import pandas as pd
import os
import json
from tqdm import tqdm

print("=" * 60)
print("TCIA-OV 영상 데이터 다운로드")
print("=" * 60)

# tcia-utils 설치 확인
try:
    from tcia_utils import nbia
    print("\n✓ tcia-utils 로드 완료")
except ImportError:
    print("\n오류: tcia-utils가 설치되지 않았습니다.")
    print("다음 명령어로 설치하세요:")
    print("  pip install tcia-utils")
    exit(1)

# 샘플링된 환자 목록 로드
if not os.path.exists('data/sampled_patients.csv'):
    print("\n오류: data/sampled_patients.csv 파일이 없습니다.")
    print("먼저 01_sample_patients.py를 실행하세요.")
    exit(1)

sampled_patients = pd.read_csv('data/sampled_patients.csv')['patient_id'].tolist()
print(f"\n샘플링된 환자 수: {len(sampled_patients)}")
print(f"환자 ID (처음 3개): {sampled_patients[:3]}")

# 디렉토리 생성
os.makedirs('data/imaging/dicom', exist_ok=True)
os.makedirs('data/imaging/metadata', exist_ok=True)

# 1. 전체 TCGA-OV 시리즈 메타데이터 가져오기
print(f"\n{'=' * 60}")
print("TCGA-OV 영상 메타데이터 수집 중...")
print(f"{'=' * 60}")

try:
    all_series = nbia.getSeries(collection="TCGA-OV", format="df")
    print(f"\n✓ 전체 시리즈 수: {len(all_series)}")
except Exception as e:
    print(f"\n오류: TCIA 연결 실패: {e}")
    print("네트워크 연결을 확인하거나 나중에 다시 시도하세요.")
    exit(1)

# 2. 샘플링된 환자의 시리즈만 필터링
print(f"\n샘플링된 환자의 시리즈 필터링 중...")
sampled_series = all_series[all_series['PatientID'].isin(sampled_patients)]

print(f"\n✓ 샘플링된 환자의 시리즈 수: {len(sampled_series)}")
print(f"  환자당 평균 시리즈: {len(sampled_series) / len(sampled_patients):.1f}개")

# 메타데이터 저장
metadata_file = 'data/imaging/metadata/tcia_ov_sampled_metadata.csv'
sampled_series.to_csv(metadata_file, index=False)
print(f"\n✓ 메타데이터 저장: {metadata_file}")

# 3. 환자별 시리즈 요약
print(f"\n{'=' * 60}")
print("환자별 영상 시리즈 요약")
print(f"{'=' * 60}")

patient_summary = sampled_series.groupby('PatientID').agg({
    'SeriesInstanceUID': 'count',
    'Modality': lambda x: ', '.join(x.unique()),
    'BodyPartExamined': lambda x: ', '.join(x.dropna().unique())
}).rename(columns={'SeriesInstanceUID': 'SeriesCount'})

print(f"\n환자별 시리즈 수:")
for idx, (patient_id, row) in enumerate(patient_summary.head(10).iterrows()):
    print(f"  {patient_id}: {row['SeriesCount']}개 시리즈 (Modality: {row['Modality']})")

if len(patient_summary) > 10:
    print(f"  ... 외 {len(patient_summary) - 10}명")

# 요약 정보 저장
summary_file = 'data/imaging/metadata/patient_series_summary.csv'
patient_summary.to_csv(summary_file)
print(f"\n✓ 환자별 요약 저장: {summary_file}")

# 4. 총 다운로드 용량 추정
print(f"\n{'=' * 60}")
print("다운로드 용량 추정")
print(f"{'=' * 60}")

# ImageCount는 시리즈당 이미지 수
if 'ImageCount' in sampled_series.columns:
    total_images = sampled_series['ImageCount'].sum()
    avg_image_size_mb = 0.5  # 평균 DICOM 이미지 크기 (MB)
    estimated_size_gb = (total_images * avg_image_size_mb) / 1024

    print(f"\n총 이미지 수: {int(total_images):,}")
    print(f"예상 용량: {estimated_size_gb:.2f} GB")
else:
    estimated_size_gb = len(sampled_series) * 0.2  # 시리즈당 평균 200MB
    print(f"\n예상 용량: {estimated_size_gb:.2f} GB")

# 5. 다운로드 여부 확인
print(f"\n{'=' * 60}")
print("다운로드 시작")
print(f"{'=' * 60}")

print(f"\n다운로드할 시리즈: {len(sampled_series)}개")
print(f"예상 용량: {estimated_size_gb:.2f} GB")
print(f"저장 경로: data/imaging/dicom/")

response = input("\n다운로드를 시작하시겠습니까? (y/n): ")

if response.lower() != 'y':
    print("\n다운로드 취소됨")
    print("메타데이터는 저장되었습니다:")
    print(f"  - {metadata_file}")
    print(f"  - {summary_file}")
    print("\n나중에 다시 실행하시거나, 수동으로 다운로드하세요.")
    exit(0)

# 6. 시리즈 다운로드
print(f"\n영상 다운로드 시작...")

download_summary = {
    'total_series': len(sampled_series),
    'downloaded': 0,
    'failed': 0,
    'failed_series': []
}

# tqdm으로 진행률 표시
for idx in tqdm(range(len(sampled_series)), desc="다운로드 중"):
    series = sampled_series.iloc[idx]
    patient_id = series['PatientID']
    series_uid = series['SeriesInstanceUID']

    output_dir = f"data/imaging/dicom/{patient_id}"

    try:
        # 시리즈 다운로드 (tcia-utils 최신 API)
        # downloadSeries는 리스트로 UID를 받음
        nbia.downloadSeries(
            [series_uid],  # 리스트로 전달
            path=output_dir,
            format="dicom"
        )
        download_summary['downloaded'] += 1

    except Exception as e:
        print(f"\n오류: {patient_id} - {series_uid} 다운로드 실패")
        print(f"  사유: {e}")
        download_summary['failed'] += 1
        download_summary['failed_series'].append({
            'patient_id': patient_id,
            'series_uid': series_uid,
            'error': str(e)
        })

# 7. 다운로드 요약
print(f"\n{'=' * 60}")
print("다운로드 완료!")
print(f"{'=' * 60}")
print(f"\n총 시리즈: {download_summary['total_series']}")
print(f"성공: {download_summary['downloaded']}")
print(f"실패: {download_summary['failed']}")

if download_summary['failed'] > 0:
    print(f"\n실패한 시리즈 목록:")
    for item in download_summary['failed_series'][:5]:
        print(f"  - {item['patient_id']}: {item['error']}")
    if len(download_summary['failed_series']) > 5:
        print(f"  ... 외 {len(download_summary['failed_series']) - 5}건")

# 요약 정보 저장
summary_json = 'data/imaging/download_summary.json'
with open(summary_json, 'w') as f:
    json.dump(download_summary, f, indent=2)
print(f"\n✓ 다운로드 요약 저장: {summary_json}")

print(f"\n다음 단계: 04_validate_data.py 실행하여 데이터 검증")
