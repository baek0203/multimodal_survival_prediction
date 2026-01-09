"""
멀티모달 데이터셋 구축 스크립트
- PyTorch Dataset 클래스 구현
- Train/Validation/Test 분할
- 데이터 로더 생성
- 142명 멀티모달 환자 데이터 통합
"""

import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from tqdm import tqdm

print("=" * 60)
print("멀티모달 데이터셋 구축")
print("=" * 60)

# ============================================================
# 1. 데이터 로드
# ============================================================
print(f"\n{'=' * 60}")
print("데이터 로드")
print(f"{'=' * 60}")

# 멀티모달 환자 목록
multimodal_patients = pd.read_csv('data/multimodal_patients.csv')['patient_id'].tolist()
print(f"\n멀티모달 환자 수: {len(multimodal_patients)}명")

# NIfTI 변환 결과 로드
conversion_results_file = 'data/imaging/conversion_results_detailed.json'
if os.path.exists(conversion_results_file):
    with open(conversion_results_file, 'r') as f:
        conversion_results = json.load(f)
else:
    print("경고: conversion_results_detailed.json 파일이 없습니다.")
    print("먼저 05_convert_dicom_to_nifti.py를 실행하세요.")
    conversion_results = []

# RNA-seq 데이터 로드
rnaseq_file = 'data/processed/rnaseq_normalized.csv'
if os.path.exists(rnaseq_file):
    df_rnaseq = pd.read_csv(rnaseq_file, index_col=0)
    print(f"RNA-seq 데이터: {df_rnaseq.shape}")
else:
    print("경고: RNA-seq 전처리 데이터가 없습니다.")
    df_rnaseq = None

# 임상 데이터 로드
clinical_file = 'data/processed/clinical_multimodal.csv'
if os.path.exists(clinical_file):
    df_clinical = pd.read_csv(clinical_file)
    print(f"임상 데이터: {df_clinical.shape}")
else:
    print("경고: 임상 데이터가 없습니다.")
    df_clinical = None

# ============================================================
# 2. 데이터 매칭 테이블 생성
# ============================================================
print(f"\n{'=' * 60}")
print("데이터 매칭 테이블 생성")
print(f"{'=' * 60}")

matching_table = []

for patient_id in multimodal_patients:
    # NIfTI 파일 경로
    nifti_dir = Path(f"data/imaging/nifti/{patient_id}")
    nifti_files = list(nifti_dir.glob("*.nii.gz")) if nifti_dir.exists() else []

    # 첫 번째 시리즈 사용 (또는 가장 큰 시리즈 선택 가능)
    nifti_file = nifti_files[0] if len(nifti_files) > 0 else None

    # RNA-seq 인덱스 (UUID 매핑 필요 - 실제로는 GDC manifest 사용)
    # 여기서는 간단히 환자 ID로 매칭 (실제 구현 시 수정 필요)
    has_rnaseq = df_rnaseq is not None

    # 임상 데이터
    clinical_row = df_clinical[df_clinical['submitter_id'] == patient_id] if df_clinical is not None else None
    has_clinical = clinical_row is not None and len(clinical_row) > 0

    # 생존 라벨 추출
    survival_time = None
    survival_status = None

    if has_clinical and len(clinical_row) > 0:
        # 생존 시간 (days_to_death 또는 days_to_last_follow_up)
        for col in clinical_row.columns:
            if 'days_to_death' in col and not clinical_row[col].isna().all():
                days_col = clinical_row[col].iloc[0]
                if isinstance(days_col, (int, float)) and not pd.isna(days_col):
                    survival_time = float(days_col)
                    survival_status = 1  # 사망
                    break

        if survival_time is None:
            for col in clinical_row.columns:
                if 'days_to_last_follow_up' in col and not clinical_row[col].isna().all():
                    days_col = clinical_row[col].iloc[0]
                    if isinstance(days_col, (int, float)) and not pd.isna(days_col):
                        survival_time = float(days_col)
                        survival_status = 0  # 생존 중
                        break

    matching_table.append({
        'patient_id': patient_id,
        'nifti_path': str(nifti_file) if nifti_file else None,
        'has_imaging': nifti_file is not None,
        'has_rnaseq': has_rnaseq,
        'has_clinical': has_clinical,
        'survival_time': survival_time,
        'survival_status': survival_status,
        'complete': nifti_file is not None and has_rnaseq and has_clinical and survival_time is not None
    })

df_matching = pd.DataFrame(matching_table)

# 완전한 데이터만 필터링
df_complete = df_matching[df_matching['complete'] == True]

print(f"\n완전한 데이터 환자 수: {len(df_complete)}명")
print(f"  - 영상: {df_complete['has_imaging'].sum()}명")
print(f"  - RNA-seq: {df_complete['has_rnaseq'].sum()}명")
print(f"  - 임상: {df_complete['has_clinical'].sum()}명")
print(f"  - 생존 라벨: {df_complete['survival_time'].notna().sum()}명")

# 매칭 테이블 저장
matching_file = 'data/processed/multimodal_matching_table.csv'
df_complete.to_csv(matching_file, index=False)
print(f"\n✓ 매칭 테이블 저장: {matching_file}")

# ============================================================
# 3. Train/Val/Test 분할
# ============================================================
print(f"\n{'=' * 60}")
print("데이터 분할 (Train/Val/Test)")
print(f"{'=' * 60}")

complete_patients = df_complete['patient_id'].tolist()

# Stratified split (생존 상태 기준)
train_patients, test_patients = train_test_split(
    complete_patients,
    test_size=0.15,
    random_state=42,
    stratify=df_complete['survival_status'].tolist() if len(set(df_complete['survival_status'])) > 1 else None
)

train_patients, val_patients = train_test_split(
    train_patients,
    test_size=0.15 / 0.85,  # 15% of total
    random_state=42,
    stratify=[df_complete[df_complete['patient_id'] == p]['survival_status'].iloc[0]
              for p in train_patients] if len(set(df_complete['survival_status'])) > 1 else None
)

print(f"\nTrain: {len(train_patients)}명")
print(f"Val:   {len(val_patients)}명")
print(f"Test:  {len(test_patients)}명")

# 분할 정보 저장
splits = {
    'train': train_patients,
    'val': val_patients,
    'test': test_patients
}

splits_file = 'data/processed/data_splits.json'
with open(splits_file, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\n✓ 분할 정보 저장: {splits_file}")

# ============================================================
# 4. PyTorch Dataset 클래스
# ============================================================
print(f"\n{'=' * 60}")
print("PyTorch Dataset 클래스 생성")
print(f"{'=' * 60}")

class MultimodalOvarianCancerDataset(Dataset):
    """
    멀티모달 난소암 데이터셋
    - 영상 (NIfTI)
    - RNA-seq
    - 임상 변수
    - 생존 라벨
    """

    def __init__(self, patient_ids, matching_table, rnaseq_data=None,
                 clinical_data=None, transform=None, target_size=(128, 128, 64)):
        """
        Args:
            patient_ids: 환자 ID 리스트
            matching_table: 매칭 테이블 DataFrame
            rnaseq_data: RNA-seq DataFrame
            clinical_data: 임상 데이터 DataFrame
            transform: 영상 augmentation
            target_size: 영상 리샘플 크기 (H, W, D)
        """
        self.patient_ids = patient_ids
        self.matching_table = matching_table.set_index('patient_id')
        self.rnaseq_data = rnaseq_data
        self.clinical_data = clinical_data
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        row = self.matching_table.loc[patient_id]

        # 1. 영상 로드
        nifti_path = row['nifti_path']
        if pd.notna(nifti_path) and os.path.exists(nifti_path):
            image = self.load_nifti(nifti_path)
        else:
            image = torch.zeros((1, *self.target_size))

        # 2. RNA-seq 로드
        if self.rnaseq_data is not None:
            # UUID 매핑 필요 (여기서는 간소화)
            # 실제로는 patient_id -> UUID 매핑 테이블 필요
            rnaseq = torch.zeros(5000)  # Placeholder
        else:
            rnaseq = torch.zeros(5000)

        # 3. 임상 변수 (나이, 병기 등)
        clinical = torch.tensor([0.0])  # Placeholder

        # 4. 라벨 (생존 시간, 상태)
        survival_time = row['survival_time'] if pd.notna(row['survival_time']) else 0.0
        survival_status = row['survival_status'] if pd.notna(row['survival_status']) else 0

        label = torch.tensor([survival_time, survival_status], dtype=torch.float32)

        return {
            'patient_id': patient_id,
            'image': image,
            'rnaseq': rnaseq,
            'clinical': clinical,
            'label': label
        }

    def load_nifti(self, nifti_path):
        """NIfTI 파일 로드 및 전처리"""
        try:
            image = sitk.ReadImage(str(nifti_path))

            # NumPy 배열로 변환
            image_np = sitk.GetArrayFromImage(image)  # (D, H, W)

            # 정규화 (0-1)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

            # 리샘플링 (간단한 resize)
            # 실제로는 SimpleITK의 Resample 사용 권장
            from scipy.ndimage import zoom

            D, H, W = image_np.shape
            zoom_factors = (
                self.target_size[2] / D,
                self.target_size[0] / H,
                self.target_size[1] / W
            )

            image_resized = zoom(image_np, zoom_factors, order=1)

            # Torch tensor로 변환 (1, H, W, D)
            image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0)

            # (1, D, H, W) 형태로 변경
            image_tensor = image_tensor.permute(0, 1, 2, 3)

            return image_tensor

        except Exception as e:
            print(f"영상 로드 실패 ({nifti_path}): {e}")
            return torch.zeros((1, *self.target_size))


# Dataset 생성 예제
print("\n데이터셋 인스턴스 생성...")

train_dataset = MultimodalOvarianCancerDataset(
    patient_ids=train_patients,
    matching_table=df_complete,
    rnaseq_data=df_rnaseq,
    clinical_data=df_clinical
)

val_dataset = MultimodalOvarianCancerDataset(
    patient_ids=val_patients,
    matching_table=df_complete,
    rnaseq_data=df_rnaseq,
    clinical_data=df_clinical
)

test_dataset = MultimodalOvarianCancerDataset(
    patient_ids=test_patients,
    matching_table=df_complete,
    rnaseq_data=df_rnaseq,
    clinical_data=df_clinical
)

print(f"\nTrain dataset: {len(train_dataset)} 샘플")
print(f"Val dataset:   {len(val_dataset)} 샘플")
print(f"Test dataset:  {len(test_dataset)} 샘플")

# 데이터 로더 생성 예제
print("\n데이터 로더 생성 예제...")

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0  # Windows에서는 0 권장
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print("✓ 데이터 로더 생성 완료")

# ============================================================
# 5. Dataset 클래스 코드 저장
# ============================================================
dataset_code_file = 'data/processed/multimodal_dataset.py'

dataset_code = '''"""
멀티모달 난소암 데이터셋 (PyTorch)
이 파일을 모델 학습 스크립트에서 import하여 사용하세요.

사용 예:
    from multimodal_dataset import MultimodalOvarianCancerDataset
    from torch.utils.data import DataLoader

    dataset = MultimodalOvarianCancerDataset(patient_ids, matching_table)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage import zoom


class MultimodalOvarianCancerDataset(Dataset):
    """멀티모달 난소암 데이터셋"""

    def __init__(self, patient_ids, matching_table, rnaseq_data=None,
                 clinical_data=None, transform=None, target_size=(128, 128, 64)):
        self.patient_ids = patient_ids
        self.matching_table = matching_table.set_index('patient_id')
        self.rnaseq_data = rnaseq_data
        self.clinical_data = clinical_data
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        row = self.matching_table.loc[patient_id]

        # 영상 로드
        nifti_path = row['nifti_path']
        if pd.notna(nifti_path) and os.path.exists(nifti_path):
            image = self.load_nifti(nifti_path)
        else:
            image = torch.zeros((1, *self.target_size))

        # RNA-seq (placeholder - 실제 구현 필요)
        rnaseq = torch.zeros(5000)

        # 임상 변수
        clinical = torch.tensor([0.0])

        # 라벨
        survival_time = row['survival_time'] if pd.notna(row['survival_time']) else 0.0
        survival_status = row['survival_status'] if pd.notna(row['survival_status']) else 0
        label = torch.tensor([survival_time, survival_status], dtype=torch.float32)

        return {
            'patient_id': patient_id,
            'image': image,
            'rnaseq': rnaseq,
            'clinical': clinical,
            'label': label
        }

    def load_nifti(self, nifti_path):
        """NIfTI 로드 및 전처리"""
        try:
            image = sitk.ReadImage(str(nifti_path))
            image_np = sitk.GetArrayFromImage(image)

            # 정규화
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

            # 리샘플링
            D, H, W = image_np.shape
            zoom_factors = (
                self.target_size[2] / D,
                self.target_size[0] / H,
                self.target_size[1] / W
            )
            image_resized = zoom(image_np, zoom_factors, order=1)

            # Tensor 변환
            image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0)

            return image_tensor

        except Exception as e:
            return torch.zeros((1, *self.target_size))
'''

with open(dataset_code_file, 'w') as f:
    f.write(dataset_code)

print(f"\n✓ Dataset 클래스 저장: {dataset_code_file}")

# ============================================================
# 최종 요약
# ============================================================
print(f"\n{'=' * 60}")
print("데이터셋 구축 완료")
print(f"{'=' * 60}")

summary = {
    'total_multimodal_patients': len(multimodal_patients),
    'complete_data_patients': len(df_complete),
    'data_splits': {
        'train': len(train_patients),
        'val': len(val_patients),
        'test': len(test_patients)
    },
    'files_created': {
        'matching_table': matching_file,
        'splits': splits_file,
        'dataset_class': dataset_code_file
    }
}

summary_file = 'data/processed/dataset_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ 요약 저장: {summary_file}")
print(f"\n{'=' * 60}")
print("다음 단계: 모델 학습 또는 EDA 수행")
print(f"{'=' * 60}")
