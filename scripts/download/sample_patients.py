"""
TCGA-OV 환자 샘플링 스크립트
- TCIA와 TCGA 모두에 데이터가 있는 환자만 선택
- 샘플링된 환자 ID 리스트 저장
"""

import requests
import json
import pandas as pd
import random
import os

# 디렉토리 생성
os.makedirs('data', exist_ok=True)

print("=" * 60)
print("TCGA-OV 환자 샘플링 시작")
print("=" * 60)

# 1. TCGA에서 전체 환자 목록 가져오기
print("\n[1/3] TCGA에서 환자 목록 가져오는 중...")

cases_endpt = "https://api.gdc.cancer.gov/cases"

filters = {
    "op": "in",
    "content": {
        "field": "project.project_id",
        "value": ["TCGA-OV"]
    }
}

params = {
    "filters": json.dumps(filters),
    "fields": "submitter_id",
    "format": "JSON",
    "size": "1000"
}

response = requests.get(cases_endpt, params=params)
tcga_data = json.loads(response.content.decode("utf-8"))

tcga_patient_ids = [case['submitter_id'] for case in tcga_data['data']['hits']]
print(f"   TCGA 전체 환자 수: {len(tcga_patient_ids)}")

# 2. TCIA에서 영상이 있는 환자 목록 가져오기
print("\n[2/3] TCIA에서 영상이 있는 환자 목록 가져오는 중...")

try:
    from tcia_utils import nbia

    # TCGA-OV 컬렉션의 환자 목록
    tcia_patients = nbia.getPatient(collection="TCGA-OV", format="df")
    tcia_patient_ids = tcia_patients['PatientID'].tolist()
    print(f"   TCIA 영상이 있는 환자 수: {len(tcia_patient_ids)}")

except ImportError:
    print("   경고: tcia-utils가 설치되지 않음. TCIA 확인 생략.")
    print("   pip install tcia-utils 실행 후 다시 시도하세요.")
    tcia_patient_ids = tcga_patient_ids  # TCGA 리스트 사용
except Exception as e:
    print(f"   경고: TCIA 연결 실패 ({e}). TCGA 환자만 사용.")
    tcia_patient_ids = tcga_patient_ids

# 3. 공통 환자 찾기 (TCIA와 TCGA 모두에 있는 환자)
print("\n[3/3] TCIA와 TCGA 공통 환자 찾는 중...")

tcga_set = set(tcga_patient_ids)
tcia_set = set(tcia_patient_ids)
common_patients = list(tcga_set.intersection(tcia_set))

print(f"   공통 환자 수: {len(common_patients)}")

if len(common_patients) == 0:
    print("\n경고: 공통 환자가 없습니다. TCGA 환자만 사용합니다.")
    common_patients = tcga_patient_ids

# 4. 랜덤 샘플링
print("\n" + "=" * 60)
print("환자 샘플링")
print("=" * 60)

SAMPLE_SIZE = 609  # 원하는 샘플 수 (조정 가능)
random.seed(42)  # 재현성을 위한 시드

if len(common_patients) < SAMPLE_SIZE:
    print(f"\n경고: 요청한 샘플 수({SAMPLE_SIZE})가 공통 환자 수({len(common_patients)})보다 많습니다.")
    SAMPLE_SIZE = len(common_patients)
    print(f"샘플 크기를 {SAMPLE_SIZE}로 조정합니다.")

sampled_patients = random.sample(common_patients, SAMPLE_SIZE)

print(f"\n샘플링 완료: {len(sampled_patients)}명")
print(f"샘플 환자 ID (처음 5개): {sampled_patients[:5]}")

# 5. 결과 저장
print("\n" + "=" * 60)
print("결과 저장")
print("=" * 60)

# 샘플링된 환자 ID 저장
df_sampled = pd.DataFrame({
    'patient_id': sampled_patients
})
df_sampled.to_csv('data/sampled_patients.csv', index=False)
print(f"\n✓ 샘플링된 환자 ID 저장: data/sampled_patients.csv")

# 전체 환자 정보도 저장 (참고용)
df_all = pd.DataFrame({
    'patient_id': common_patients,
    'sampled': [pid in sampled_patients for pid in common_patients]
})
df_all.to_csv('data/all_common_patients.csv', index=False)
print(f"✓ 전체 공통 환자 목록 저장: data/all_common_patients.csv")

# 요약 정보 저장
summary = {
    'total_tcga_patients': len(tcga_patient_ids),
    'total_tcia_patients': len(tcia_patient_ids),
    'common_patients': len(common_patients),
    'sampled_patients': len(sampled_patients),
    'sample_rate': f"{len(sampled_patients) / len(common_patients) * 100:.1f}%"
}

with open('data/sampling_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ 샘플링 요약 저장: data/sampling_summary.json")

print("\n" + "=" * 60)
print("샘플링 완료!")
print("=" * 60)
print(f"\nTCGA 전체 환자: {summary['total_tcga_patients']}")
print(f"TCIA 전체 환자: {summary['total_tcia_patients']}")
print(f"공통 환자: {summary['common_patients']}")
print(f"샘플링 환자: {summary['sampled_patients']} ({summary['sample_rate']})")
print("\n다음 단계: 02_download_tcga.py 실행")
