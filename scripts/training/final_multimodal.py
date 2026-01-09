"""
최종 멀티모달 생존 예측 모델
- MONAI DenseNet (CT)
- MLP (RNA-seq)
- MLP (Clinical)
- Late Fusion
- torchsurv Cox Loss
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    print("경고: torchsurv 없음. pip install torchsurv 필요")
    print("자체 Cox loss 사용")
    USE_TORCHSURV = False

# MONAI import
try:
    from monai.networks.nets import DenseNet121
    print("✓ MONAI 사용 가능")
    USE_MONAI = True
except ImportError:
    print("경고: MONAI 없음. pip install monai 필요")
    print("Simple 3D CNN 사용")
    USE_MONAI = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("Final Multimodal Survival Prediction")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 장치: {device}")

# ============================================================
# 모델 정의
# ============================================================

class MultiModalSurvivalNet(nn.Module):
    def __init__(self, rna_dim=5005, clinical_dim=1):
        super().__init__()

        # CT encoder (MONAI DenseNet or simple CNN)
        if USE_MONAI:
            # MONAI DenseNet with custom output
            self.ct_encoder = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=128,  # 직접 128로 출력
                pretrained=False
            )
            self.use_monai = True
            ct_out_dim = 128
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
                nn.AdaptiveAvgPool3d(1),  # Pooling 여기로 이동
            )
            self.use_monai = False
            ct_out_dim = 128

        self.ct_pool = nn.AdaptiveAvgPool3d(1)  # MONAI용

        # RNA encoder (고차원 압축)
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # Clinical encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU()
        )

        # Fusion (Late Fusion)
        fusion_dim = ct_out_dim + 128 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Cox head
        self.cox_head = nn.Linear(128, 1)

    def forward(self, ct, rna, clinical):
        # CT features
        ct_feat = self.ct_encoder(ct)

        # MONAI DenseNet with out_channels returns (B, out_channels) directly
        # Simple CNN returns (B, C, D, H, W) and needs pooling
        if self.use_monai:
            # MONAI already outputs (B, 128), no pooling needed
            if ct_feat.dim() > 2:
                ct_feat = self.ct_pool(ct_feat)
                ct_feat = ct_feat.view(ct_feat.size(0), -1)
        else:
            # Simple CNN already has pooling in sequential
            ct_feat = ct_feat.view(ct_feat.size(0), -1)

        # RNA features
        rna_feat = self.rna_encoder(rna)

        # Clinical features
        clin_feat = self.clinical_encoder(clinical)

        # Fusion
        fused = torch.cat([ct_feat, rna_feat, clin_feat], dim=1)
        fused = self.fusion(fused)

        # Cox hazard (log-hazard)
        hazard = self.cox_head(fused).squeeze(1)

        return hazard


# ============================================================
# Cox Loss (torchsurv or custom)
# ============================================================

if USE_TORCHSURV:
    def cox_loss(hazard, event, time):
        """torchsurv Cox loss"""
        # torchsurv requires event to be bool type
        event = event.bool()
        return neg_partial_log_likelihood(hazard, event, time)

    def calculate_cindex(hazard, event, time):
        """torchsurv C-index"""
        # torchsurv requires event to be bool type
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


# ============================================================
# 데이터 로드
# ============================================================

print(f"\n{'=' * 60}")
print("데이터 로드")
print(f"{'=' * 60}")

matching_table = pd.read_csv('data/processed/multimodal_matching_table.csv')
complete_patients = matching_table[
    (matching_table['has_imaging'] == True) &
    (matching_table['survival_time'].notna())
]['patient_id'].tolist()

print(f"\n완전한 데이터: {len(complete_patients)}명")

import sys
sys.path.append('data/processed')
from multimodal_dataset import MultimodalOvarianCancerDataset

# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
N_FOLDS = 5
PATIENCE = 15

print(f"\nHyperparameters:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Folds: {N_FOLDS}")
print(f"  Early stopping patience: {PATIENCE}")

# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        ct = batch['image'].to(device)
        rna = batch['rnaseq'].to(device)
        clinical = batch['clinical'].to(device)
        label = batch['label'].to(device)

        time = label[:, 0]
        event = label[:, 1]

        # Forward
        hazard = model(ct, rna, clinical)
        loss = cox_loss(hazard, event, time)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


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

            time = label[:, 0]
            event = label[:, 1]

            hazard = model(ct, rna, clinical)
            loss = cox_loss(hazard, event, time)

            total_loss += loss.item()
            num_batches += 1

            all_hazards.extend(hazard.cpu().numpy())
            all_times.extend(time.cpu().numpy())
            all_events.extend(event.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # C-index
    all_hazards = torch.tensor(all_hazards)
    all_times = torch.tensor(all_times)
    all_events = torch.tensor(all_events)

    c_index = calculate_cindex(all_hazards, all_events, all_times)

    return avg_loss, c_index


# ============================================================
# K-Fold Cross Validation
# ============================================================

print(f"\n{'=' * 60}")
print("5-Fold Cross Validation")
print(f"{'=' * 60}\n")

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

cv_results = []
os.makedirs('models/final', exist_ok=True)
os.makedirs('results/final', exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(complete_patients)):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    print(f"{'=' * 60}")

    train_patients = [complete_patients[i] for i in train_idx]
    val_patients = [complete_patients[i] for i in val_idx]

    print(f"Train: {len(train_patients)}명, Val: {len(val_patients)}명")

    # Datasets
    train_dataset = MultimodalOvarianCancerDataset(
        patient_ids=train_patients,
        matching_table=matching_table,
        target_size=(64, 64, 32)
    )

    val_dataset = MultimodalOvarianCancerDataset(
        patient_ids=val_patients,
        matching_table=matching_table,
        target_size=(64, 64, 32)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = MultiModalSurvivalNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_c_index = 0
    patience_counter = 0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_c_index = validate(model, val_loader, device)

        scheduler.step(val_c_index)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, C-index={val_c_index:.4f}")

        # Early stopping
        if val_c_index > best_c_index:
            best_c_index = val_c_index
            patience_counter = 0
            torch.save(model.state_dict(), f'models/final/fold_{fold+1}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n✓ Fold {fold+1} 완료: Best C-index = {best_c_index:.4f}")

    cv_results.append({
        'fold': fold + 1,
        'best_c_index': best_c_index
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
    print(f"  Fold {r['fold']}: {r['best_c_index']:.4f}")

# 저장
summary = {
    'model': 'MultiModalSurvivalNet (Late Fusion)',
    'c_index_mean': float(np.mean(c_indices)),
    'c_index_std': float(np.std(c_indices)),
    'fold_results': cv_results,
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': NUM_EPOCHS,
        'n_folds': N_FOLDS
    }
}

with open('results/final/cv_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n저장 완료:")
print(f"  - results/final/cv_results.json")
print(f"  - models/final/fold_X_best.pth")

print(f"\n{'=' * 60}")
print("완료!")
print(f"{'=' * 60}")
