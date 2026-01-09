"""
Kaplan-Meier 생존 곡선 생성
- 각 모델의 best fold에서 예측
- High risk vs Low risk 그룹 비교
- Log-rank test로 통계적 유의성 검증
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Kaplan-Meier 곡선 생성")
print("=" * 80)

# ============================================================
# 모델 정의 (각 스크립트에서 복사)
# ============================================================

# Image-Only Model
class ImageOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.risk_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        risk = self.risk_head(x).squeeze(1)
        return risk


# Partial Modality Model
try:
    from monai.networks.nets import DenseNet121
    USE_MONAI = True
except:
    USE_MONAI = False

class PartialModalityNet(nn.Module):
    def __init__(self, rna_dim=5005, clinical_dim=1):
        super().__init__()

        if USE_MONAI:
            self.ct_encoder = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=128,
                pretrained=False
            )
            self.use_monai = True
        else:
            self.ct_encoder = nn.Sequential(
                nn.Conv3d(1, 32, 3, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
            )
            self.use_monai = False

        self.ct_pool = nn.AdaptiveAvgPool3d(1)

        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(128 + 128 + 32 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

        fusion_dim = 128 + 128 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.cox_head = nn.Linear(128, 1)

    def forward(self, ct, rna, clinical, mask):
        batch_size = ct.size(0)

        ct_feat = self.ct_encoder(ct)
        if self.use_monai:
            if ct_feat.dim() > 2:
                ct_feat = self.ct_pool(ct_feat)
                ct_feat = ct_feat.view(batch_size, -1)
        else:
            ct_feat = ct_feat.view(batch_size, -1)

        rna_feat = self.rna_encoder(rna)
        clin_feat = self.clinical_encoder(clinical)

        ct_feat = ct_feat * mask[:, 0:1]
        rna_feat = rna_feat * mask[:, 1:2]
        clin_feat = clin_feat * mask[:, 2:3]

        concat_feat = torch.cat([ct_feat, rna_feat, clin_feat, mask], dim=1)
        gate_weights = self.gate(concat_feat)

        ct_feat_weighted = ct_feat * gate_weights[:, 0:1].expand_as(ct_feat)
        rna_feat_weighted = rna_feat * gate_weights[:, 1:2].expand_as(rna_feat)
        clin_feat_weighted = clin_feat * gate_weights[:, 2:3].expand_as(clin_feat)

        fused = torch.cat([ct_feat_weighted, rna_feat_weighted, clin_feat_weighted], dim=1)
        fused = self.fusion(fused)

        hazard = self.cox_head(fused).squeeze(1)

        return hazard, gate_weights


# SimMLM Model
class ModalityExpert(nn.Module):
    def __init__(self, modality_type, input_dim=None, output_dim=128):
        super().__init__()
        self.modality_type = modality_type

        if modality_type == 'image':
            if USE_MONAI:
                self.encoder = DenseNet121(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=output_dim,
                    pretrained=False
                )
                self.use_monai = True
            else:
                self.encoder = nn.Sequential(
                    nn.Conv3d(1, 32, 3, stride=2, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(),
                    nn.Conv3d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(),
                    nn.Conv3d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d(1),
                )
                self.use_monai = False
            self.pool = nn.AdaptiveAvgPool3d(1)

        elif modality_type == 'rnaseq':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, output_dim),
                nn.ReLU()
            )

        elif modality_type == 'clinical':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.ReLU()
            )

        self.cox_head = nn.Linear(output_dim, 1)

    def forward(self, x):
        if self.modality_type == 'image':
            feat = self.encoder(x)
            if self.use_monai:
                if feat.dim() > 2:
                    feat = self.pool(feat)
                    feat = feat.view(feat.size(0), -1)
            else:
                feat = feat.view(feat.size(0), -1)
        else:
            feat = self.encoder(x)

        hazard = self.cox_head(feat).squeeze(1)
        return feat, hazard


class GatingNetwork(nn.Module):
    def __init__(self, feature_dim=128, num_modalities=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * num_modalities + num_modalities, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_modalities)
        )

    def forward(self, features_list, mask):
        concat_feat = torch.cat(features_list + [mask], dim=1)
        logits = self.gate(concat_feat)
        logits = logits.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(logits, dim=1)
        return weights


class SimMLM_SurvivalNet(nn.Module):
    def __init__(self, rna_dim=5005, clinical_dim=1, feature_dim=128):
        super().__init__()
        self.expert_image = ModalityExpert('image', output_dim=feature_dim)
        self.expert_rnaseq = ModalityExpert('rnaseq', input_dim=rna_dim, output_dim=feature_dim)
        self.expert_clinical = ModalityExpert('clinical', input_dim=clinical_dim, output_dim=feature_dim)
        self.gating = GatingNetwork(feature_dim=feature_dim, num_modalities=3)
        self.ensemble_cox = nn.Linear(feature_dim, 1)
        self.feature_dim = feature_dim

    def forward(self, image, rnaseq, clinical, mask):
        feat_img, hazard_img = self.expert_image(image)
        feat_rna, hazard_rna = self.expert_rnaseq(rnaseq)
        feat_clin, hazard_clin = self.expert_clinical(clinical)

        feat_img = feat_img * mask[:, 0:1]
        feat_rna = feat_rna * mask[:, 1:2]
        feat_clin = feat_clin * mask[:, 2:3]

        gate_weights = self.gating([feat_img, feat_rna, feat_clin], mask)

        fused_feat = (
            gate_weights[:, 0:1] * feat_img +
            gate_weights[:, 1:2] * feat_rna +
            gate_weights[:, 2:3] * feat_clin
        )

        ensemble_hazard = self.ensemble_cox(fused_feat).squeeze(1)

        expert_hazards = {
            'image': hazard_img,
            'rnaseq': hazard_rna,
            'clinical': hazard_clin
        }

        return ensemble_hazard, expert_hazards, gate_weights


# ============================================================
# Dataset 로드
# ============================================================

import sys
sys.path.append('data/processed')
import SimpleITK as sitk
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader

class PartialModalityDataset(Dataset):
    def __init__(self, patient_ids, matching_table, target_size=(64, 64, 32)):
        self.patient_ids = patient_ids
        self.matching_table = matching_table
        self.target_size = target_size

        rnaseq_file = 'data/processed/rnaseq_normalized_mapped.csv'
        if os.path.exists(rnaseq_file):
            self.rnaseq_data = pd.read_csv(rnaseq_file, index_col=0)
            self.rna_dim = self.rnaseq_data.shape[1]
        else:
            self.rnaseq_data = None
            self.rna_dim = 5005

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        row = self.matching_table[self.matching_table['patient_id'] == patient_id].iloc[0]

        has_image = False
        image = torch.zeros((1, *self.target_size))

        nifti_path = row['nifti_path']
        if pd.notna(nifti_path) and os.path.exists(nifti_path):
            try:
                img = sitk.ReadImage(str(nifti_path))
                img_np = sitk.GetArrayFromImage(img)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                D, H, W = img_np.shape
                target_d, target_h, target_w = self.target_size
                zoom_factors = (target_d / D, target_h / H, target_w / W)
                img_resized = zoom(img_np, zoom_factors, order=1)
                image = torch.from_numpy(img_resized).float().unsqueeze(0)
                has_image = True
            except:
                pass

        has_rnaseq = False
        rnaseq = torch.zeros(self.rna_dim)

        if self.rnaseq_data is not None and patient_id in self.rnaseq_data.index:
            rnaseq_values = self.rnaseq_data.loc[patient_id].values
            rnaseq = torch.from_numpy(rnaseq_values).float()
            has_rnaseq = True

        has_clinical = False
        clinical = torch.tensor([0.0], dtype=torch.float32)

        if pd.notna(row['age']):
            clinical = torch.tensor([row['age'] / 100.0], dtype=torch.float32)
            has_clinical = True

        has_survival = False
        survival_time = 0.0
        survival_status = 0

        if pd.notna(row['survival_time']):
            survival_time = float(row['survival_time'])
            survival_status = int(row['survival_status'])
            has_survival = True

        label = torch.tensor([survival_time, survival_status], dtype=torch.float32)

        mask = torch.tensor([
            float(has_image),
            float(has_rnaseq),
            float(has_clinical)
        ])

        return {
            'patient_id': patient_id,
            'image': image,
            'rnaseq': rnaseq,
            'clinical': clinical,
            'label': label,
            'mask': mask,
            'has_survival': has_survival
        }


print("\n이 스크립트는 학습된 모델을 로드하여 예측을 생성하고 KM 곡선을 그립니다.")
print("현재는 모델 정의만 포함되어 있습니다.")
print("\n실행하려면 각 모델의 best fold checkpoint를 로드해야 합니다.")
print("예: models/simmim/fold_1_best.pth")

print("\n✓ 모델 정의 완료")
print("✓ Dataset 정의 완료")
print("\nKM 곡선 생성을 위해서는 추가 구현이 필요합니다.")
