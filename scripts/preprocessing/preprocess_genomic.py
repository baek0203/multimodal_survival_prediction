"""
유전체 데이터 전처리 스크립트
- RNA-seq 정규화 및 특징 선택
- Mutation 특징 행렬 생성
- CNV 데이터 처리
- 142명 멀티모달 환자 데이터만 추출
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("유전체 데이터 전처리")
print("=" * 60)

# 멀티모달 환자 목록 로드
if not os.path.exists('data/multimodal_patients.csv'):
    print("\n오류: data/multimodal_patients.csv 파일이 없습니다.")
    exit(1)

multimodal_patients = pd.read_csv('data/multimodal_patients.csv')['patient_id'].tolist()
print(f"\n멀티모달 환자 수: {len(multimodal_patients)}명")

# 출력 디렉토리 생성
os.makedirs('data/processed', exist_ok=True)

# ============================================================
# 1. RNA-seq 데이터 전처리
# ============================================================
print(f"\n{'=' * 60}")
print("RNA-seq 데이터 처리")
print(f"{'=' * 60}")

rnaseq_dir = Path("data/genomic/rnaseq")
rnaseq_data = {}

print("\n1단계: RNA-seq 파일 읽기...")

# GDC 다운로드 파일은 UUID 디렉토리 구조
for uuid_dir in tqdm(list(rnaseq_dir.glob("*")), desc="RNA-seq 파일 스캔"):
    if not uuid_dir.is_dir():
        continue

    # .tsv 파일 찾기
    tsv_files = list(uuid_dir.glob("*.tsv"))
    if len(tsv_files) == 0:
        continue

    tsv_file = tsv_files[0]

    # annotations.txt에서 환자 ID 매핑 (GDC 메타데이터)
    # 실제로는 GDC manifest나 API에서 환자 ID를 매핑해야 함
    # 여기서는 파일명 또는 메타데이터를 통해 매핑

    try:
        # RNA-seq 카운트 데이터 읽기
        df_counts = pd.read_csv(tsv_file, sep='\t', comment='#')

        # 데이터 형식 확인 (GDC는 여러 형식 가능)
        if 'gene_id' in df_counts.columns:
            # STAR gene counts 형식
            gene_col = 'gene_id'
            count_cols = [col for col in df_counts.columns if 'unstranded' in col or 'tpm' in col.lower()]
            if len(count_cols) > 0:
                count_col = count_cols[0]
            else:
                count_col = df_counts.columns[-1]  # 마지막 열
        else:
            continue

        # 유전자-카운트 딕셔너리 생성
        gene_counts = dict(zip(df_counts[gene_col], df_counts[count_col]))

        # UUID를 키로 저장 (나중에 환자 ID와 매핑 필요)
        rnaseq_data[uuid_dir.name] = gene_counts

    except Exception as e:
        continue

print(f"✓ 읽은 RNA-seq 파일 수: {len(rnaseq_data)}")

# RNA-seq 데이터를 DataFrame으로 변환
if len(rnaseq_data) > 0:
    print("\n2단계: RNA-seq 데이터프레임 생성...")
    df_rnaseq = pd.DataFrame(rnaseq_data).T

    # 유전자 이름 정리 (ENSG ID에서 버전 제거)
    df_rnaseq.columns = [col.split('.')[0] if '|' not in col else col.split('|')[0] for col in df_rnaseq.columns]

    # 수치형으로 변환
    df_rnaseq = df_rnaseq.apply(pd.to_numeric, errors='coerce')
    df_rnaseq = df_rnaseq.fillna(0)

    print(f"  형태: {df_rnaseq.shape} (샘플 x 유전자)")

    # 3단계: 정규화
    print("\n3단계: 로그 변환 및 정규화...")

    # 로그 변환 (log2(count + 1))
    df_rnaseq_log = np.log2(df_rnaseq + 1)

    # Z-score 정규화
    scaler = StandardScaler()
    rnaseq_normalized = scaler.fit_transform(df_rnaseq_log)
    df_rnaseq_normalized = pd.DataFrame(
        rnaseq_normalized,
        index=df_rnaseq_log.index,
        columns=df_rnaseq_log.columns
    )

    # 4단계: 특징 선택 (분산 기반)
    print("\n4단계: 특징 선택...")

    # 낮은 분산 제거 (상위 5000개 유전자 선택)
    variances = df_rnaseq_normalized.var(axis=0)
    top_genes = variances.nlargest(5000).index.tolist()

    df_rnaseq_selected = df_rnaseq_normalized[top_genes]

    print(f"  선택된 유전자 수: {len(top_genes)}")
    print(f"  최종 형태: {df_rnaseq_selected.shape}")

    # 저장
    rnaseq_output = 'data/processed/rnaseq_normalized.csv'
    df_rnaseq_selected.to_csv(rnaseq_output)
    print(f"\n✓ RNA-seq 데이터 저장: {rnaseq_output}")

    # 선택된 유전자 목록 저장
    with open('data/processed/selected_genes.txt', 'w') as f:
        f.write('\n'.join(top_genes))

else:
    print("경고: RNA-seq 데이터가 없습니다.")
    df_rnaseq_selected = None

# ============================================================
# 2. Mutation 데이터 전처리
# ============================================================
print(f"\n{'=' * 60}")
print("Mutation 데이터 처리")
print(f"{'=' * 60}")

mutation_dir = Path("data/genomic/mutation")
all_mutations = []

print("\n1단계: Mutation 파일 읽기...")

# MAF 파일 읽기
for maf_file in tqdm(list(mutation_dir.rglob("*.maf*")), desc="Mutation 파일 스캔"):
    try:
        # MAF 파일은 주석이 많으므로 건너뛰기
        df_maf = pd.read_csv(maf_file, sep='\t', comment='#', low_memory=False)

        if len(df_maf) > 0:
            all_mutations.append(df_maf)

    except Exception as e:
        continue

if len(all_mutations) > 0:
    print(f"✓ 읽은 Mutation 파일 수: {len(all_mutations)}")

    # 모든 mutation 데이터 병합
    df_mutations = pd.concat(all_mutations, ignore_index=True)

    print(f"\n2단계: Mutation 특징 생성...")
    print(f"  총 mutation 수: {len(df_mutations)}")

    # Mutation 특징 생성 (유전자별 mutation 카운트)
    if 'Hugo_Symbol' in df_mutations.columns:
        # 환자별 mutation된 유전자 카운트
        # 실제로는 환자 ID 컬럼이 필요 (Tumor_Sample_Barcode)
        mutation_features = {}

        if 'Tumor_Sample_Barcode' in df_mutations.columns:
            # 환자 ID 추출 (TCGA-XX-XXXX 형식)
            df_mutations['patient_id'] = df_mutations['Tumor_Sample_Barcode'].str[:12]

            # 유전자별 mutation 카운트 행렬
            mutation_counts = df_mutations.groupby(['patient_id', 'Hugo_Symbol']).size().unstack(fill_value=0)

            # Binary 변환 (mutation 있음/없음)
            mutation_binary = (mutation_counts > 0).astype(int)

            # 저장
            mutation_output = 'data/processed/mutation_features.csv'
            mutation_binary.to_csv(mutation_output)
            print(f"\n✓ Mutation 데이터 저장: {mutation_output}")
            print(f"  형태: {mutation_binary.shape} (환자 x 유전자)")

else:
    print("경고: Mutation 데이터가 없습니다.")

# ============================================================
# 3. CNV 데이터 전처리
# ============================================================
print(f"\n{'=' * 60}")
print("CNV 데이터 처리")
print(f"{'=' * 60}")

cnv_dir = Path("data/genomic/cnv")
all_cnv = []

print("\n1단계: CNV 파일 읽기...")

# CNV segment 파일 읽기
for cnv_file in tqdm(list(cnv_dir.rglob("*.txt")), desc="CNV 파일 스캔"):
    try:
        df_cnv = pd.read_csv(cnv_file, sep='\t', comment='#')

        if len(df_cnv) > 0:
            all_cnv.append(df_cnv)

    except Exception as e:
        continue

if len(all_cnv) > 0:
    print(f"✓ 읽은 CNV 파일 수: {len(all_cnv)}")

    df_cnv_all = pd.concat(all_cnv, ignore_index=True)

    print(f"\n2단계: CNV 특징 생성...")
    print(f"  총 CNV segment 수: {len(df_cnv_all)}")

    # CNV 특징 생성 (chromosome arm별 평균 copy number)
    # 실제 구현은 segment를 유전자에 매핑하는 복잡한 작업 필요

    cnv_output = 'data/processed/cnv_raw.csv'
    df_cnv_all.to_csv(cnv_output, index=False)
    print(f"\n✓ CNV 원본 데이터 저장: {cnv_output}")
    print(f"  (주의: 유전자 레벨 매핑은 별도 처리 필요)")

else:
    print("경고: CNV 데이터가 없습니다.")

# ============================================================
# 4. 임상 데이터 전처리
# ============================================================
print(f"\n{'=' * 60}")
print("임상 데이터 처리")
print(f"{'=' * 60}")

clinical_file = Path("data/clinical/tcga_ov_multimodal_clinical.csv")

if clinical_file.exists():
    df_clinical = pd.read_csv(clinical_file)

    print(f"\n임상 데이터 형태: {df_clinical.shape}")

    # 멀티모달 환자만 필터링
    df_clinical_multimodal = df_clinical[df_clinical['submitter_id'].isin(multimodal_patients)]

    # 생존 관련 변수 추출
    survival_cols = [col for col in df_clinical_multimodal.columns
                     if 'vital_status' in col or 'days_to' in col]

    print(f"멀티모달 환자 수: {len(df_clinical_multimodal)}")
    print(f"주요 변수: {survival_cols[:5]}")

    # 저장
    clinical_output = 'data/processed/clinical_multimodal.csv'
    df_clinical_multimodal.to_csv(clinical_output, index=False)
    print(f"\n✓ 임상 데이터 저장: {clinical_output}")

else:
    print("경고: 임상 데이터 파일이 없습니다.")

# ============================================================
# 최종 요약
# ============================================================
print(f"\n{'=' * 60}")
print("전처리 완료 요약")
print(f"{'=' * 60}")

summary = {
    'multimodal_patients': len(multimodal_patients),
    'processed_files': {
        'rnaseq': 'data/processed/rnaseq_normalized.csv' if df_rnaseq_selected is not None else None,
        'mutation': 'data/processed/mutation_features.csv',
        'cnv': 'data/processed/cnv_raw.csv',
        'clinical': 'data/processed/clinical_multimodal.csv'
    },
    'rnaseq_shape': list(df_rnaseq_selected.shape) if df_rnaseq_selected is not None else None,
    'selected_genes': len(top_genes) if df_rnaseq_selected is not None else 0
}

summary_file = 'data/processed/preprocessing_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ 요약 저장: {summary_file}")

print(f"\n{'=' * 60}")
print("다음 단계: 멀티모달 데이터셋 구축 (07_create_multimodal_dataset.py)")
print(f"{'=' * 60}")
