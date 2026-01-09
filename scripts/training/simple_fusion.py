"""
Simple Late Fusion: RNA-seq + Image
- RNA-seq이 강력하므로 (0.6174) 이를 base로 사용
- Image feature를 보조로 추가
- Simple concatenation + MLP
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    print("경고: MONAI 없음")
    USE_MONAI = False

import sys
sys.path.append('data/processed')
import SimpleITK as sitk
from scipy.ndimage import zoom

# Fallback implementations
if not USE_TORCHSURV:
    def neg_partial_log_likelihood(log_hazard, event, time):
        sorted_indices = torch.argsort(time, descending=True)
        log_hazard = log_hazard[sorted_indices]
        event = event[sorted_indices]

        hazard_ratio = torch.exp(log_hazard)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = log_hazard - log_risk
        censored_likelihood = uncensored_likelihood * event
        neg_likelihood = -torch.sum(censored_likelihood) / (torch.sum(event) + 1e-8)
        return neg_likelihood

    class ConcordanceIndex:
        def __call__(self, log_hazard, event, time):
            n = len(time)
            concordant = 0
            permissible = 0

            for i in range(n):
                if event[i] == 1:
                    for j in range(n):
                        if time[j] > time[i]:
                            permissible += 1
                            if log_hazard[i] > log_hazard[j]:
                                concordant += 1

            return torch.tensor(concordant / permissible if permissible > 0 else 0.5)

print("=" * 60)
print("Simple Late Fusion: RNA-seq + Image")
print("=" * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {DEVICE}")

# Config
DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/simple_fusion'
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 3
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

# ============================================================
# Dataset
# ============================================================

class SimpleFusionDataset(Dataset):
    """RNA-seq + Image dataset"""

    def __init__(self, patient_ids, matching_table, rnaseq_df, target_size=(64, 64, 32)):
        self.patient_ids = patient_ids
        self.matching_table = matching_table
        self.rnaseq_df = rnaseq_df
        self.target_size = target_size

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        row = self.matching_table[self.matching_table['patient_id'] == patient_id].iloc[0]

        # Image
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

        # RNA-seq
        has_rnaseq = False
        rnaseq = torch.zeros(self.rnaseq_df.shape[1])

        if patient_id in self.rnaseq_df.index:
            rnaseq = torch.FloatTensor(self.rnaseq_df.loc[patient_id].values)
            has_rnaseq = True

        # Survival
        has_survival = row['has_survival']
        time = float(row['survival_time']) if pd.notna(row['survival_time']) else 0.0
        event = int(row['survival_status']) if pd.notna(row['survival_status']) else 0

        return {
            'image': image,
            'rnaseq': rnaseq,
            'has_image': has_image,
            'has_rnaseq': has_rnaseq,
            'has_survival': has_survival,
            'time': torch.FloatTensor([time]),
            'event': torch.LongTensor([event])
        }

# ============================================================
# Model
# ============================================================

class SimpleFusionModel(nn.Module):
    """Simple late fusion of RNA-seq + Image"""

    def __init__(self, rna_dim=5005, img_feature_dim=128, rna_feature_dim=256):
        super().__init__()

        # RNA-seq encoder (stronger)
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, rna_feature_dim),
            nn.ReLU()
        )

        # Image encoder (lightweight)
        if USE_MONAI:
            self.image_encoder = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=img_feature_dim,
                pretrained=False
            )
            self.use_monai = True
            self.image_pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.image_encoder = nn.Sequential(
                nn.Conv3d(1, 32, 3, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, img_feature_dim, 3, stride=2, padding=1),
                nn.BatchNorm3d(img_feature_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1)
            )
            self.use_monai = False

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(rna_feature_dim + img_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, image, rnaseq):
        batch_size = image.size(0)

        # Encode
        rna_feat = self.rna_encoder(rnaseq)

        if self.use_monai:
            img_feat = self.image_encoder(image)
            if img_feat.dim() > 2:
                img_feat = self.image_pool(img_feat)
                img_feat = img_feat.view(batch_size, -1)
        else:
            img_feat = self.image_encoder(image)
            img_feat = img_feat.view(batch_size, -1)

        # Fuse
        fused = torch.cat([rna_feat, img_feat], dim=1)
        log_hazard = self.fusion(fused).squeeze(1)

        return log_hazard

# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        image = batch['image'].to(device)
        rnaseq = batch['rnaseq'].to(device)
        time = batch['time'].squeeze().to(device)
        event = batch['event'].squeeze().to(device)
        has_survival = batch['has_survival']

        # Filter survival samples
        survival_mask = torch.tensor(has_survival, dtype=torch.bool, device=device)

        if survival_mask.sum() < 2:
            continue

        optimizer.zero_grad()

        log_hazard = model(image, rnaseq)
        log_hazard_surv = log_hazard[survival_mask]
        time_surv = time[survival_mask]
        event_surv = event[survival_mask].bool()  # torchsurv requires boolean

        if event_surv.sum() == 0:
            continue

        loss = neg_partial_log_likelihood(log_hazard_surv, event_surv, time_surv)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0

def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_hazards = []
    all_events = []
    all_times = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating', leave=False):
            image = batch['image'].to(device)
            rnaseq = batch['rnaseq'].to(device)
            time = batch['time'].squeeze().to(device)
            event = batch['event'].squeeze().to(device)
            has_survival = batch['has_survival']

            survival_mask = torch.tensor(has_survival, dtype=torch.bool, device=device)

            if survival_mask.sum() < 2:
                continue

            log_hazard = model(image, rnaseq)
            log_hazard_surv = log_hazard[survival_mask]
            time_surv = time[survival_mask]
            event_surv = event[survival_mask].bool()  # torchsurv requires boolean

            if event_surv.sum() == 0:
                continue

            loss = neg_partial_log_likelihood(log_hazard_surv, event_surv, time_surv)

            total_loss += loss.item()
            n_batches += 1

            all_hazards.append(log_hazard_surv.cpu())
            all_events.append(event_surv.cpu())
            all_times.append(time_surv.cpu())

    if len(all_hazards) == 0:
        return 0.0, 0.5

    all_hazards = torch.cat(all_hazards)
    all_events = torch.cat(all_events)
    all_times = torch.cat(all_times)

    # torchsurv requires event to be boolean
    all_events = all_events.bool()

    cindex_metric = ConcordanceIndex()
    c_index = cindex_metric(all_hazards, all_events, all_times).item()

    return total_loss / n_batches if n_batches > 0 else 0.0, c_index

# ============================================================
# Load Data
# ============================================================

print("\n[1] Loading data...")

matching_table = pd.read_csv(os.path.join(DATA_DIR, 'full_matching_table.csv'))

# Image + RNA-seq + Survival 모두 있는 환자만
valid_patients = matching_table[
    (matching_table['has_imaging'] == True) &
    (matching_table['has_rnaseq'] == True) &
    (matching_table['has_survival'] == True)
]

print(f"  ✓ Complete multimodal patients: {len(valid_patients)}")

# RNA-seq 로드
rnaseq_file = os.path.join(DATA_DIR, 'rnaseq_normalized_mapped.csv')
if not os.path.exists(rnaseq_file):
    rnaseq_file = os.path.join(DATA_DIR, 'rnaseq_normalized.csv')

rnaseq_df = pd.read_csv(rnaseq_file, index_col=0)

# 환자 교집합
common_patients = list(set(valid_patients['patient_id']) & set(rnaseq_df.index))
print(f"  ✓ Final patient count: {len(common_patients)}")

# ============================================================
# Cross-Validation
# ============================================================

print(f"\n[2] Starting {N_FOLDS}-fold Cross-Validation...")

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(common_patients), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{N_FOLDS}")
    print(f"{'='*60}")

    train_ids = [common_patients[i] for i in train_idx]
    val_ids = [common_patients[i] for i in val_idx]

    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)}")

    # Datasets
    train_dataset = SimpleFusionDataset(train_ids, matching_table, rnaseq_df)
    val_dataset = SimpleFusionDataset(val_ids, matching_table, rnaseq_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = SimpleFusionModel(rna_dim=rnaseq_df.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training
    best_c_index = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_c_index = validate(model, val_loader, DEVICE)
        scheduler.step()

        if val_c_index > best_c_index:
            best_c_index = val_c_index
            best_epoch = epoch
            torch.save(model.state_dict(),
                      os.path.join(RESULTS_DIR, f'best_model_fold{fold}.pth'))

        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val C-index: {val_c_index:.4f} | "
                  f"Best: {best_c_index:.4f} (Epoch {best_epoch})")

    fold_results.append({
        'fold': fold,
        'best_c_index': best_c_index,
        'best_epoch': best_epoch,
        'train_size': len(train_ids),
        'val_size': len(val_ids)
    })

    print(f"\n  ✓ Fold {fold} Complete | Best C-index: {best_c_index:.4f}")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Cross-Validation Results Summary")
print("=" * 60)

c_indices = [result['best_c_index'] for result in fold_results]
mean_c_index = np.mean(c_indices)
std_c_index = np.std(c_indices)

print(f"\nC-index: {mean_c_index:.4f} ± {std_c_index:.4f}")
print(f"\nFold-wise Results:")
for result in fold_results:
    print(f"  Fold {result['fold']}: {result['best_c_index']:.4f} (Epoch {result['best_epoch']})")

# Save results
cv_results = {
    'model': 'Simple-Fusion (RNA+Image)',
    'n_folds': N_FOLDS,
    'num_epochs': NUM_EPOCHS,
    'c_index_mean': mean_c_index,
    'c_index_std': std_c_index,
    'fold_results': fold_results
}

with open(os.path.join(RESULTS_DIR, 'cv_results.json'), 'w') as f:
    json.dump(cv_results, f, indent=2)

print(f"\n✓ Results saved to: {RESULTS_DIR}/cv_results.json")
print("=" * 60)
