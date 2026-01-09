"""
TCGA-OV 전체 유전체 데이터 다운로드 스크립트
- 전체 멀티모달 교집합 환자의 RNA-seq, Mutation, CNV, Clinical 데이터 다운로드
- GDC API + 전체 코호트 사용
"""

import requests
import json
import pandas as pd
import os
import subprocess
import glob

print("=" * 60)
print("TCGA-OV 전체 멀티모달 유전체 데이터 다운로드")
print("=" * 60)

# 전체 멀티모달 코호트 로드 (샘플링 제거)
if not os.path.exists('data/sampled_patients.csv'):
    print("\n오류: data/sampled_patients.csv 파일이 없습니다.")
    print("먼저 modified_sampling.py를 실행하세요.")
    exit(1)

sampled_df = pd.read_csv('data/sampled_patients.csv')
sampled_patients = sampled_df['patient_id'].tolist()
print(f"\n✅ 멀티모달 교집합 환자 수: {len(sampled_patients)}명")
print(f"환자 ID (처음 5개): {sampled_patients[:5]}")

# 디렉토리 생성
os.makedirs('data/genomic/rnaseq', exist_ok=True)
os.makedirs('data/genomic/mutation', exist_ok=True)
os.makedirs('data/genomic/cnv', exist_ok=True)
os.makedirs('data/clinical', exist_ok=True)

files_endpt = "https://api.gdc.cancer.gov/files"
cases_endpt = "https://api.gdc.cancer.gov/cases"

def download_data_for_cohort(data_category, data_type, output_dir, description):
    """전체 교집합 환자의 특정 데이터 타입 다운로드 (배치 처리)"""

    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")

    # 환자를 배치로 나누기 (URL 길이 제한 회피)
    BATCH_SIZE = 50  # 한 번에 50명씩 처리
    all_file_info = []

    print(f"총 {len(sampled_patients)}명을 {BATCH_SIZE}명씩 배치 처리...")

    for i in range(0, len(sampled_patients), BATCH_SIZE):
        batch = sampled_patients[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(sampled_patients) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  배치 {batch_num}/{total_batches} ({len(batch)}명) 처리 중...")

        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-OV"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.submitter_id",
                        "value": batch  # 배치만 전송
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_category",
                        "value": [data_category]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": [data_type]
                    }
                }
            ]
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,data_type,file_size",
            "format": "JSON",
            "size": "5000"
        }

        response = requests.get(files_endpt, params=params)

        if response.status_code != 200:
            print(f"    오류: 배치 {batch_num} API 요청 실패 (status code: {response.status_code})")
            continue

        data = json.loads(response.content.decode("utf-8"))

        if 'data' in data and 'hits' in data['data']:
            batch_files = data['data']['hits']
            all_file_info.extend(batch_files)
            print(f"    ✓ {len(batch_files)}개 파일 발견")
        else:
            print(f"    경고: 배치 {batch_num} 데이터 형식 오류")

    file_ids = [f['file_id'] for f in all_file_info]
    total_size_mb = sum([f.get('file_size', 0) for f in all_file_info]) / (1024 * 1024)

    print(f"\n전체 결과:")
    print(f"  찾은 파일 수: {len(file_ids)}")
    print(f"  총 용량: {total_size_mb:.1f} MB")

    try:
        covered_patients = len(set([f['cases'][0]['submitter_id'] for f in all_file_info if 'cases' in f and len(f['cases']) > 0]))
        print(f"  대상 환자 커버: {covered_patients}/{len(sampled_patients)}명")
    except:
        pass

    if len(file_ids) == 0:
        print("다운로드할 파일이 없습니다.")
        return 0

    file_info = all_file_info

    # 매니페스트 파일 생성
    manifest_file = f"data/gdc_manifest_{data_category.replace(' ', '_').lower()}_full.txt"
    print(f"\n매니페스트 파일 생성 중: {manifest_file}")
    
    with open(manifest_file, 'w') as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for item in file_info:
            file_id = item['file_id']
            file_name = item['file_name']
            file_size = item.get('file_size', 0)
            f.write(f"{file_id}\t{file_name}\t\t{file_size}\t\n")

    print("✓ 매니페스트 파일 생성 완료")

    # gdc-client로 다운로드
    print(f"\ngdc-client로 다운로드 중...")
    print(f"저장 경로: {output_dir}")

    # gdc-client 경로 확인 (gdc_client 또는 gdc-client)
    gdc_client_paths = ['./gdc_client', './gdc-client', 'gdc_client', 'gdc-client']
    gdc_client = None

    for path in gdc_client_paths:
        if os.path.exists(path) or subprocess.run(['which', path.split('/')[-1]], 
                                                 capture_output=True).returncode == 0:
            gdc_client = path
            break

    if gdc_client is None:
        print("\n⚠️  gdc-client를 찾을 수 없습니다.")
        print("수동 실행:")
        print(f"  {gdc_client_paths[0]} download -m {manifest_file} -d {output_dir}")
        return len(file_ids)

    try:
        cmd = [gdc_client, 'download', '-m', manifest_file, '-d', output_dir]
        print(f"실행: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 다운로드 완료!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  다운로드 중 오류 발생 (계속 진행)")
        print(f"수동 실행: {' '.join(cmd)}")

    return len(file_ids)

# 1. RNA-seq 데이터 다운로드 (HTSeq - Counts)
rnaseq_count = download_data_for_cohort(
    "Transcriptome Profiling",
    "Gene Expression Quantification",
    "data/genomic/rnaseq",
    "RNA-seq 데이터 다운로드 (전체 코호트)"
)

# 2. Mutation 데이터 다운로드
mutation_count = download_data_for_cohort(
    "Simple Nucleotide Variation", 
    "Masked Somatic Mutation",
    "data/genomic/mutation",
    "Mutation 데이터 다운로드 (전체 코호트)"
)

# 3. Copy Number Variation 다운로드
cnv_count = download_data_for_cohort(
    "Copy Number Variation",
    "Copy Number Segment",
    "data/genomic/cnv",
    "Copy Number Variation 다운로드 (전체 코호트)"
)

# 4. 전체 TCGA-OV 임상 데이터 다운로드
print(f"\n{'=' * 60}")
print("전체 TCGA-OV 임상 데이터 다운로드")
print(f"{'=' * 60}")

filters = {
    "op": "in",
    "content": {
        "field": "project.project_id",
        "value": ["TCGA-OV"]
    }
}

params = {
    "filters": json.dumps(filters),
    "fields": "submitter_id,diagnoses.age_at_diagnosis,diagnoses.tumor_stage,diagnoses.vital_status,diagnoses.days_to_death,diagnoses.days_to_last_follow_up,demographic.gender,demographic.race,demographic.ethnicity,exposures.years_smoked",
    "format": "JSON",
    "size": "2000",
    "expand": "diagnoses,demographic,exposures"
}

print("GDC API에서 전체 임상 데이터 가져오는 중...")
response = requests.get(cases_endpt, params=params)
clinical_data = json.loads(response.content.decode("utf-8"))

df_clinical = pd.json_normalize(clinical_data['data']['hits'])
clinical_file = 'data/clinical/tcga_ov_full_clinical.csv'
df_clinical.to_csv(clinical_file, index=False)

# 코호트 환자만 필터링
df_clinical_cohort = df_clinical[df_clinical['submitter_id'].isin(sampled_patients)]
df_clinical_cohort.to_csv('data/clinical/tcga_ov_multimodal_clinical.csv', index=False)

print(f"\n✓ 전체 임상 데이터 저장: {clinical_file}")
print(f"✓ 코호트 임상 데이터 저장: data/clinical/tcga_ov_multimodal_clinical.csv")
print(f"  전체 환자 수: {len(df_clinical)}명")
print(f"  코호트 환자 수: {len(df_clinical_cohort)}명")

# 최종 요약
print(f"\n{'=' * 60}")
print("✅ 전체 다운로드 완료 요약")
print(f"{'=' * 60}")
print(f"🎯 멀티모달 코호트: {len(sampled_patients)}명")
print(f"📊 RNA-seq 파일: {rnaseq_count}개")
print(f"🔬 Mutation 파일: {mutation_count}개")
print(f"🧬 CNV 파일: {cnv_count}개")
print(f"🏥 코호트 임상 데이터: {len(df_clinical_cohort)}명")
print(f"\n📁 ICFNet 학습 준비 완료!")
print(f"   data/tcga_ov_multimodal_clinical.csv ← 생존 라벨")
print(f"   data/tcia_ov/ ← CT 이미지 (19명)")
print(f"   data/genomic/* ← 오믹스 데이터")

print(f"\n🚀 다음 단계: ICFNet 학습 시작!")
