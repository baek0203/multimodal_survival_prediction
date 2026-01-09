"""
GDC API를 통해 RNA-seq case UUID → submitter_id 매핑
"""

import requests
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("=" * 60)
print("GDC API를 통한 RNA-seq case UUID 매핑")
print("=" * 60)

# RNA-seq 디렉토리에서 case UUID 목록 가져오기
rnaseq_dir = Path('data/genomic/rnaseq')
case_uuids = [d.name for d in rnaseq_dir.iterdir() if d.is_dir()]

print(f"\nRNA-seq case UUID 수: {len(case_uuids)}")
print(f"샘플 (처음 5개):")
for i, uuid in enumerate(case_uuids[:5]):
    print(f"  {i+1}. {uuid}")

# GDC API로 file UUID → submitter_id 매핑
print(f"\nGDC API로 매핑 중 (files endpoint)...")

files_endpt = "https://api.gdc.cancer.gov/files"

uuid_to_patient = {}
batch_size = 100  # API 요청은 batch로

for i in tqdm(range(0, len(case_uuids), batch_size), desc="API 요청"):
    batch_uuids = case_uuids[i:i+batch_size]

    filters = {
        "op": "in",
        "content": {
            "field": "file_id",
            "value": batch_uuids
        }
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,cases.submitter_id",
        "format": "JSON",
        "size": str(batch_size)
    }

    try:
        response = requests.get(files_endpt, params=params, timeout=30)
        data = json.loads(response.content.decode("utf-8"))

        for file_info in data['data']['hits']:
            file_id = file_info['file_id']
            cases = file_info.get('cases', [])

            if len(cases) > 0:
                # 첫 번째 case의 submitter_id 사용
                submitter_id = cases[0]['submitter_id']
                uuid_to_patient[file_id] = submitter_id

    except Exception as e:
        print(f"  경고: Batch {i//batch_size + 1} 실패 ({e})")
        continue

print(f"\n매핑 결과:")
print(f"  전체 case UUID: {len(case_uuids)}개")
print(f"  매핑 성공: {len(uuid_to_patient)}개")
print(f"  매핑 실패: {len(case_uuids) - len(uuid_to_patient)}개")

if len(uuid_to_patient) > 0:
    print(f"\n샘플 매핑 (처음 5개):")
    for i, (uuid, pid) in enumerate(list(uuid_to_patient.items())[:5]):
        print(f"  {uuid} → {pid}")

    # RNA-seq 데이터프레임 로드
    print(f"\nRNA-seq 데이터프레임 업데이트 중...")
    rnaseq_file = 'data/processed/rnaseq_normalized.csv'
    df_rna = pd.read_csv(rnaseq_file, index_col=0)

    print(f"  원본 shape: {df_rna.shape}")
    print(f"  원본 인덱스 타입: {df_rna.index[0]}")

    # 인덱스 매핑
    mapped_indices = []
    unmapped_count = 0

    for uuid in df_rna.index:
        if uuid in uuid_to_patient:
            mapped_indices.append(uuid_to_patient[uuid])
        else:
            # 매핑 안 되는 경우 원본 UUID 유지
            mapped_indices.append(uuid)
            unmapped_count += 1

    df_rna.index = mapped_indices

    print(f"\n최종 인덱스 매핑:")
    print(f"  매핑된 submitter_id: {len(uuid_to_patient)}개")
    print(f"  미매핑 UUID: {unmapped_count}개")

    # 중복 인덱스 확인
    duplicates = df_rna.index.duplicated().sum()
    if duplicates > 0:
        print(f"\n경고: 중복 인덱스 {duplicates}개 발견!")
        print("  중복 제거: 첫 번째 샘플만 유지")
        df_rna = df_rna[~df_rna.index.duplicated(keep='first')]

    # 저장
    output_file = 'data/processed/rnaseq_normalized_mapped.csv'
    df_rna.to_csv(output_file)

    print(f"\n✓ 저장: {output_file}")
    print(f"  Shape: {df_rna.shape}")
    print(f"  인덱스 샘플 (처음 5개):")
    for i, idx in enumerate(df_rna.index[:5]):
        print(f"    {i+1}. {idx}")

    # 매핑 테이블 저장
    with open('data/processed/case_uuid_to_patient.json', 'w') as f:
        json.dump(uuid_to_patient, f, indent=2)

    print(f"\n✓ 매핑 테이블 저장: data/processed/case_uuid_to_patient.json")

    print(f"\n{'=' * 60}")
    print("완료!")
    print(f"{'=' * 60}")

else:
    print(f"\n오류: API 매핑 실패")
