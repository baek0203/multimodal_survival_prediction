"""
Partial Modality 학습 - 608명 전체 활용
- Mask 기반으로 missing modality 처리
- 생존 라벨 있는 샘플만 Cox loss 적용
- 나머지는 reconstruction/contrastive loss로 encoder 안정화
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# torchsurv import
try:
    from torchsurv.loss.cox import neg_partial_log_likelihood
    from torchsurv.metrics.cindex import ConcordanceIndex
    print("✓ torchsurv 사용 가능")
    USE_TORCHSURV = True
except ImportError:
    print("경고: torchsurv 없음. 자체 Cox loss 사용")
    USE_TORCHSURV = False

# MONAI import
try:
    from monai.networks.nets import DenseNet121
    print("✓ MONAI 사용 가능")
    USE_MONAI = True
except ImportError:
    print("경고: MONAI 없음. Simple 3D CNN 사용")
    USE_MONAI = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("Partial Modality Learning (608명 전체 활용)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 장치: {device}")

# ============================================================
# Partial Modality Dataset
# ============================================================

import sys
sys.path.append('data/processed')
import SimpleITK as sitk
from scipy.ndimage import zoom

class PartialModalityDataset(Dataset):
    """Partial Modality를 지원하는 Dataset"""

    def __init__(self, patient_ids, matching_table, target_size=(64, 64, 32)):
        self.patient_ids = patient_ids
        self.matching_table = matching_table
        self.target_size = target_size

        # RNA-seq 데이터 로드
        rnaseq_file = 'data/processed/rnaseq_normalized_mapped.csv'
        if os.path.exists(rnaseq_file):
            print(f"RNA-seq 로드: {rnaseq_file}")
            self.rnaseq_data = pd.read_csv(rnaseq_file, index_col=0)
            print(f"  Shape: {self.rnaseq_data.shape}")
            self.rna_dim = self.rnaseq_data.shape[1]
        else:
            print("경고: RNA-seq 파일 없음")
            self.rnaseq_data = None
            self.rna_dim = 5005

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        row = self.matching_table[self.matching_table['patient_id'] == patient_id].iloc[0]

        # === Image 로드 ===
        has_image = False
        image = torch.zeros((1, *self.target_size))

        nifti_path = row['nifti_path']
        if pd.notna(nifti_path) and os.path.exists(nifti_path):
            try:
                img = sitk.ReadImage(str(nifti_path))
                img_np = sitk.GetArrayFromImage(img)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                # SimpleITK returns (D, H, W) for 3D images
                D, H, W = img_np.shape
                # target_size convention: (D, H, W) to match Conv3d
                target_d, target_h, target_w = self.target_size
                zoom_factors = (
                    target_d / D,  # Depth axis
                    target_h / H,  # Height axis
                    target_w / W   # Width axis
                )
                img_resized = zoom(img_np, zoom_factors, order=1)
                # Unsqueeze channel dimension: (1, D, H, W)
                image = torch.from_numpy(img_resized).float().unsqueeze(0)
                has_image = True
            except:
                pass

        # === RNA-seq 로드 ===
        has_rnaseq = False
        rnaseq = torch.zeros(self.rna_dim)

        if self.rnaseq_data is not None and patient_id in self.rnaseq_data.index:
            rnaseq_values = self.rnaseq_data.loc[patient_id].values
            rnaseq = torch.from_numpy(rnaseq_values).float()
            has_rnaseq = True

        # === Clinical 로드 ===
        has_clinical = False
        clinical = torch.tensor([0.0], dtype=torch.float32)

        if pd.notna(row['age']):
            clinical = torch.tensor([row['age'] / 100.0], dtype=torch.float32)  # normalize
            has_clinical = True

        # === Survival 라벨 ===
        has_survival = False
        survival_time = 0.0
        survival_status = 0

        if pd.notna(row['survival_time']):
            survival_time = float(row['survival_time'])
            survival_status = int(row['survival_status'])
            has_survival = True

        label = torch.tensor([survival_time, survival_status], dtype=torch.float32)

        # === Modality masks ===
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


# ============================================================
# Partial Modality Model with Gating
# ============================================================

class PartialModalityNet(nn.Module):
    def __init__(self, rna_dim=5005, clinical_dim=1):
        super().__init__()

        # === CT Encoder ===
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

        # === RNA Encoder ===
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # === Clinical Encoder ===
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU()
        )

        # === Gating mechanism ===
        # Input: concatenated features + modality masks
        self.gate = nn.Sequential(
            nn.Linear(128 + 128 + 32 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)  # Gate weights for each modality
        )

        # === Fusion ===
        fusion_dim = 128 + 128 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # === Cox head ===
        self.cox_head = nn.Linear(128, 1)

    def forward(self, ct, rna, clinical, mask):
        """
        Args:
            ct: (B, 1, D, H, W)
            rna: (B, rna_dim)
            clinical: (B, clinical_dim)
            mask: (B, 3) - [has_image, has_rnaseq, has_clinical]
        """
        batch_size = ct.size(0)

        # === Encode each modality ===
        ct_feat = self.ct_encoder(ct)
        if self.use_monai:
            if ct_feat.dim() > 2:
                ct_feat = self.ct_pool(ct_feat)
                ct_feat = ct_feat.view(batch_size, -1)
        else:
            ct_feat = ct_feat.view(batch_size, -1)

        rna_feat = self.rna_encoder(rna)
        clin_feat = self.clinical_encoder(clinical)

        # === Apply mask (zero out missing modalities) ===
        ct_feat = ct_feat * mask[:, 0:1]
        rna_feat = rna_feat * mask[:, 1:2]
        clin_feat = clin_feat * mask[:, 2:3]

        # === Gating ===
        concat_feat = torch.cat([ct_feat, rna_feat, clin_feat, mask], dim=1)
        gate_weights = self.gate(concat_feat)  # (B, 3)

        # Weighted features
        ct_feat_weighted = ct_feat * gate_weights[:, 0:1].expand_as(ct_feat)
        rna_feat_weighted = rna_feat * gate_weights[:, 1:2].expand_as(rna_feat)
        clin_feat_weighted = clin_feat * gate_weights[:, 2:3].expand_as(clin_feat)

        # === Fusion ===
        fused = torch.cat([ct_feat_weighted, rna_feat_weighted, clin_feat_weighted], dim=1)
        fused = self.fusion(fused)

        # === Cox hazard ===
        hazard = self.cox_head(fused).squeeze(1)

        return hazard, gate_weights


# ============================================================
# Loss Functions
# ============================================================

if USE_TORCHSURV:
    def cox_loss(hazard, event, time):
        """torchsurv Cox loss"""
        event = event.bool()
        return neg_partial_log_likelihood(hazard, event, time)

    def calculate_cindex(hazard, event, time):
        """torchsurv C-index"""
        event = event.bool()
        cindex_fn = ConcordanceIndex()
        return cindex_fn(hazard, event, time).item()
else:
    def cox_loss(hazard, event, time):
        """Custom Cox loss"""
        if hazard.shape[0] < 2:
            return torch.tensor(0.0, device=hazard.device, requires_grad=True)
        if event.sum() == 0:
            return torch.tensor(0.0, device=hazard.device, requires_grad=True)

        order = torch.argsort(time, descending=True)
        hazard = hazard[order]
        event = event[order]

        log_cumsum = torch.logcumsumexp(hazard, dim=0)
        uncensored_likelihood = hazard - log_cumsum
        loss = -torch.sum(uncensored_likelihood * event) / (event.sum() + 1e-8)

        return loss

    def calculate_cindex(hazard, event, time):
        """Custom C-index"""
        try:
            from lifelines.utils import concordance_index
            return concordance_index(time.cpu().numpy(), -hazard.cpu().numpy(), event.cpu().numpy())
        except:
            return 0.5


def gate_entropy_loss(gate_weights):
    """
    Gate entropy penalty to prevent collapse
    Encourage balanced gate usage across modalities
    """
    # gate_weights: (B, 3)
    eps = 1e-8
    entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=1)
    # Maximize entropy -> minimize negative entropy
    return -entropy.mean()


# ============================================================
# 데이터 로드
# ============================================================

print(f"\n{'=' * 60}")
print("데이터 로드")
print(f"{'=' * 60}")

matching_table = pd.read_csv('data/processed/full_matching_table.csv')

# 모든 환자 (608명)
all_patients = matching_table['patient_id'].tolist()
print(f"\n전체 환자: {len(all_patients)}명")

# 생존 라벨 있는 환자
survival_patients = matching_table[
    matching_table['survival_time'].notna()
]['patient_id'].tolist()
print(f"생존 라벨 있는 환자: {len(survival_patients)}명")

# 통계
print(f"\n모달리티 분포:")
print(f"  Imaging: {matching_table['has_imaging'].sum()}명")
print(f"  RNA-seq: {matching_table['has_rnaseq'].sum()}명")
print(f"  완전 멀티모달: {((matching_table['has_imaging'] == True) & (matching_table['has_rnaseq'] == True)).sum()}명")

# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 8  # Larger batch with 608 samples
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
N_FOLDS = 3  # 빠른 실험을 위해 3-fold
PATIENCE = 15
GATE_ENTROPY_WEIGHT = 0.01  # Entropy regularization weight

print(f"\nHyperparameters:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Folds: {N_FOLDS}")
print(f"  Gate entropy weight: {GATE_ENTROPY_WEIGHT}")

# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_cox_loss = 0
    total_entropy_loss = 0
    num_survival_batches = 0
    num_batches = 0

    for batch in loader:
        ct = batch['image'].to(device)
        rna = batch['rnaseq'].to(device)
        clinical = batch['clinical'].to(device)
        label = batch['label'].to(device)
        mask = batch['mask'].to(device)
        has_survival = batch['has_survival']

        # Forward
        hazard, gate_weights = model(ct, rna, clinical, mask)

        # Cox loss (only for samples with survival labels)
        survival_mask = torch.tensor(has_survival, dtype=torch.bool, device=device)

        if survival_mask.sum() > 0:
            hazard_surv = hazard[survival_mask]
            time_surv = label[survival_mask, 0]
            event_surv = label[survival_mask, 1]

            if hazard_surv.shape[0] >= 2 and event_surv.sum() > 0:
                c_loss = cox_loss(hazard_surv, event_surv, time_surv)
                total_cox_loss += c_loss.item()
                num_survival_batches += 1
            else:
                c_loss = torch.tensor(0.0, device=device)
        else:
            c_loss = torch.tensor(0.0, device=device)

        # Gate entropy loss (all samples)
        e_loss = gate_entropy_loss(gate_weights)
        total_entropy_loss += e_loss.item()

        # Total loss
        loss = c_loss + GATE_ENTROPY_WEIGHT * e_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_batches += 1

    avg_cox = total_cox_loss / num_survival_batches if num_survival_batches > 0 else 0
    avg_entropy = total_entropy_loss / num_batches if num_batches > 0 else 0

    return avg_cox, avg_entropy


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_hazards = []
    all_times = []
    all_events = []

    with torch.no_grad():
        for batch in loader:
            ct = batch['image'].to(device)
            rna = batch['rnaseq'].to(device)
            clinical = batch['clinical'].to(device)
            label = batch['label'].to(device)
            mask = batch['mask'].to(device)
            has_survival = batch['has_survival']

            hazard, _ = model(ct, rna, clinical, mask)

            # Only evaluate on samples with survival labels
            survival_mask = torch.tensor(has_survival, dtype=torch.bool, device=device)

            if survival_mask.sum() > 0:
                hazard_surv = hazard[survival_mask]
                time_surv = label[survival_mask, 0]
                event_surv = label[survival_mask, 1]

                if hazard_surv.shape[0] >= 2 and event_surv.sum() > 0:
                    loss = cox_loss(hazard_surv, event_surv, time_surv)
                    total_loss += loss.item()
                    num_batches += 1

                    all_hazards.extend(hazard_surv.cpu().numpy())
                    all_times.extend(time_surv.cpu().numpy())
                    all_events.extend(event_surv.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # C-index
    if len(all_hazards) > 0:
        all_hazards = torch.tensor(all_hazards)
        all_times = torch.tensor(all_times)
        all_events = torch.tensor(all_events)
        c_index = calculate_cindex(all_hazards, all_events, all_times)
    else:
        c_index = 0.5

    return avg_loss, c_index


# ============================================================
# K-Fold Cross Validation (생존 라벨 있는 환자 기준)
# ============================================================

print(f"\n{'=' * 60}")
print("3-Fold Cross Validation (생존 라벨 기준)")
print(f"{'=' * 60}\n")

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

cv_results = []
os.makedirs('models/partial_modality', exist_ok=True)
os.makedirs('results/partial_modality', exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(survival_patients)):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    print(f"{'=' * 60}")

    # Split based on survival patients
    train_survival = [survival_patients[i] for i in train_idx]
    val_survival = [survival_patients[i] for i in val_idx]

    # Add non-survival patients to training set for encoder training
    non_survival = [p for p in all_patients if p not in survival_patients]
    train_all = train_survival + non_survival

    print(f"Train: {len(train_all)}명 (생존라벨: {len(train_survival)}명)")
    print(f"Val: {len(val_survival)}명")

    # Datasets
    train_dataset = PartialModalityDataset(
        patient_ids=train_all,
        matching_table=matching_table,
        target_size=(64, 64, 32)
    )

    val_dataset = PartialModalityDataset(
        patient_ids=val_survival,
        matching_table=matching_table,
        target_size=(64, 64, 32)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = PartialModalityNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_c_index = 0
    patience_counter = 0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_cox, train_entropy = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_c_index = validate(model, val_loader, device)

        scheduler.step(val_c_index)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Cox={train_cox:.4f}, Entropy={train_entropy:.4f}, Val Loss={val_loss:.4f}, C-index={val_c_index:.4f}")

        # Early stopping
        if val_c_index > best_c_index:
            best_c_index = val_c_index
            patience_counter = 0
            torch.save(model.state_dict(), f'models/partial_modality/fold_{fold+1}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n✓ Fold {fold+1} 완료: Best C-index = {best_c_index:.4f}")

    cv_results.append({
        'fold': fold + 1,
        'best_c_index': best_c_index,
        'train_size': len(train_all),
        'train_survival_size': len(train_survival),
        'val_size': len(val_survival)
    })

# ============================================================
# 결과 요약
# ============================================================

print(f"\n{'=' * 60}")
print("Cross Validation 결과")
print(f"{'=' * 60}")

c_indices = [r['best_c_index'] for r in cv_results]

print(f"\nC-index:")
print(f"  평균: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
print(f"  범위: [{np.min(c_indices):.4f}, {np.max(c_indices):.4f}]")

print(f"\nFold별 상세:")
for r in cv_results:
    print(f"  Fold {r['fold']}: {r['best_c_index']:.4f} (Train: {r['train_size']}명, Val: {r['val_size']}명)")

# 저장
summary = {
    'model': 'PartialModalityNet (Gating + Entropy Regularization)',
    'c_index_mean': float(np.mean(c_indices)),
    'c_index_std': float(np.std(c_indices)),
    'fold_results': cv_results,
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': NUM_EPOCHS,
        'n_folds': N_FOLDS,
        'gate_entropy_weight': GATE_ENTROPY_WEIGHT
    }
}

with open('results/partial_modality/cv_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n저장 완료:")
print(f"  - results/partial_modality/cv_results.json")
print(f"  - models/partial_modality/fold_X_best.pth")

print(f"\n{'=' * 60}")
print("완료!")
print(f"{'=' * 60}")
