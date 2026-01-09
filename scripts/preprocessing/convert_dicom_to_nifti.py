"""
DICOM to NIfTI 변환 스크립트
- 142명 멀티모달 환자의 DICOM 영상을 NIfTI 형식으로 변환
- 메타데이터 추출 및 저장
- 품질 검사 수행
"""

import os
import pandas as pd
import json
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DICOM → NIfTI 변환")
print("=" * 60)

# 멀티모달 완전 환자 목록 로드
if not os.path.exists('data/multimodal_patients.csv'):
    print("\n오류: data/multimodal_patients.csv 파일이 없습니다.")
    print("먼저 04_validate_data.py를 실행하세요.")
    exit(1)

multimodal_patients = pd.read_csv('data/multimodal_patients.csv')['patient_id'].tolist()
print(f"\n멀티모달 환자 수: {len(multimodal_patients)}명")

# 출력 디렉토리 생성
os.makedirs('data/imaging/nifti', exist_ok=True)
os.makedirs('data/imaging/nifti_metadata', exist_ok=True)

# 변환 결과 저장
conversion_results = []

print(f"\n{'=' * 60}")
print("DICOM 시리즈 변환 시작")
print(f"{'=' * 60}\n")

def convert_dicom_series_to_nifti(dicom_dir, output_file, metadata_file):
    """
    DICOM 시리즈를 NIfTI로 변환하고 메타데이터 저장

    Args:
        dicom_dir: DICOM 파일이 있는 디렉토리
        output_file: NIfTI 출력 파일 경로
        metadata_file: 메타데이터 JSON 파일 경로

    Returns:
        bool: 성공 여부
        dict: 메타데이터
    """
    try:
        # DICOM 시리즈 읽기
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))

        if len(dicom_names) == 0:
            return False, {"error": "No DICOM files found"}

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # 메타데이터 추출
        metadata = {
            'num_slices': len(dicom_names),
            'size': list(image.GetSize()),
            'spacing': list(image.GetSpacing()),
            'origin': list(image.GetOrigin()),
            'direction': [float(x) for x in image.GetDirection()],
            'pixel_type': str(image.GetPixelIDTypeAsString()),
        }

        # DICOM 태그에서 추가 정보 추출 (첫 번째 슬라이스)
        reader_single = sitk.ImageFileReader()
        reader_single.SetFileName(dicom_names[0])
        reader_single.LoadPrivateTagsOn()
        reader_single.ReadImageInformation()

        # 주요 DICOM 태그
        tags_to_extract = {
            'Modality': '0008|0060',
            'SeriesDescription': '0008|103e',
            'StudyDate': '0008|0020',
            'SliceThickness': '0018|0050',
            'PixelSpacing': '0028|0030',
            'Manufacturer': '0008|0070',
            'ManufacturerModelName': '0008|1090',
        }

        for key, tag in tags_to_extract.items():
            try:
                value = reader_single.GetMetaData(tag)
                metadata[key] = value
            except:
                metadata[key] = None

        # NIfTI로 저장
        sitk.WriteImage(image, str(output_file))

        # 메타데이터 저장
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True, metadata

    except Exception as e:
        return False, {"error": str(e)}

# 환자별 DICOM 변환
for patient_id in tqdm(multimodal_patients, desc="환자별 변환"):
    patient_dicom_dir = Path(f"data/imaging/dicom/{patient_id}")

    if not patient_dicom_dir.exists():
        conversion_results.append({
            'patient_id': patient_id,
            'status': 'failed',
            'reason': 'DICOM directory not found',
            'num_series': 0
        })
        continue

    # 환자별 시리즈 디렉토리 탐색
    series_dirs = [d for d in patient_dicom_dir.iterdir() if d.is_dir()]

    patient_result = {
        'patient_id': patient_id,
        'num_series': len(series_dirs),
        'converted_series': 0,
        'failed_series': 0,
        'series_info': []
    }

    # 환자별 출력 디렉토리
    patient_nifti_dir = Path(f"data/imaging/nifti/{patient_id}")
    patient_nifti_dir.mkdir(parents=True, exist_ok=True)

    # 각 시리즈 변환
    for idx, series_dir in enumerate(series_dirs):
        series_name = series_dir.name
        output_nifti = patient_nifti_dir / f"series_{idx:02d}.nii.gz"
        output_metadata = Path(f"data/imaging/nifti_metadata/{patient_id}_series_{idx:02d}.json")

        success, metadata = convert_dicom_series_to_nifti(
            series_dir,
            output_nifti,
            output_metadata
        )

        if success:
            patient_result['converted_series'] += 1
            patient_result['series_info'].append({
                'series_index': idx,
                'series_name': series_name,
                'nifti_file': str(output_nifti),
                'metadata_file': str(output_metadata),
                'modality': metadata.get('Modality', 'Unknown'),
                'num_slices': metadata.get('num_slices', 0),
                'size': metadata.get('size', []),
                'spacing': metadata.get('spacing', [])
            })
        else:
            patient_result['failed_series'] += 1

    patient_result['status'] = 'success' if patient_result['converted_series'] > 0 else 'failed'
    conversion_results.append(patient_result)

# 결과를 DataFrame으로 변환
df_results = pd.DataFrame([
    {
        'patient_id': r['patient_id'],
        'status': r['status'],
        'num_series': r.get('num_series', 0),
        'converted_series': r.get('converted_series', 0),
        'failed_series': r.get('failed_series', 0)
    }
    for r in conversion_results
])

# 결과 저장
results_file = 'data/imaging/conversion_results.csv'
df_results.to_csv(results_file, index=False)

# 상세 결과 JSON 저장
detailed_results_file = 'data/imaging/conversion_results_detailed.json'
with open(detailed_results_file, 'w') as f:
    json.dump(conversion_results, f, indent=2)

# 요약 통계
print(f"\n{'=' * 60}")
print("변환 완료 요약")
print(f"{'=' * 60}")

total_patients = len(conversion_results)
successful = len([r for r in conversion_results if r.get('status') == 'success'])
total_series = sum([r.get('num_series', 0) for r in conversion_results])
converted_series = sum([r.get('converted_series', 0) for r in conversion_results])
failed_series = sum([r.get('failed_series', 0) for r in conversion_results])

print(f"\n총 환자 수: {total_patients}명")
print(f"성공한 환자: {successful}명 ({successful/total_patients*100:.1f}%)")
print(f"\n총 시리즈 수: {total_series}개")
print(f"변환 성공: {converted_series}개")
print(f"변환 실패: {failed_series}개")
print(f"성공률: {converted_series/total_series*100:.1f}%")

print(f"\n✓ 결과 저장: {results_file}")
print(f"✓ 상세 결과: {detailed_results_file}")

# 디스크 사용량
nifti_dir = Path('data/imaging/nifti')
nifti_size_mb = sum(f.stat().st_size for f in nifti_dir.rglob('*.nii.gz')) / (1024 * 1024)
print(f"\nNIfTI 총 용량: {nifti_size_mb:.1f} MB ({nifti_size_mb/1024:.2f} GB)")

print(f"\n{'=' * 60}")
print("다음 단계: 유전체 데이터 전처리 (06_preprocess_genomic.py)")
print(f"{'=' * 60}")
