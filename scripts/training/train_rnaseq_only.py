"""
RNA-seq Only Baseline Model
- RNA-seq 데이터만 사용한 생존 예측 모델
- Image-only와 대칭되는 ablation study용 베이스라인
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
from sklearn.preprocessing import StandardScaler
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

print("=" * 60)
print("RNA-seq Only Survival Prediction")
print("=" * 60)

# ============================================================
# Fallback Cox Loss (if torchsurv not available)
# ============================================================

if not USE_TORCHSURV:
    def neg_partial_log_likelihood(log_hazard, event, time):
        """Simple Cox Partial Likelihood Loss"""
        # Sort by time (descending)
        sorted_indices = torch.argsort(time, descending=True)
        log_hazard = log_hazard[sorted_indices]
        event = event[sorted_indices]

        # Cox partial likelihood
        hazard_ratio = torch.exp(log_hazard)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = log_hazard - log_risk
        censored_likelihood = uncensored_likelihood * event
        neg_likelihood = -torch.sum(censored_likelihood) / torch.sum(event)
        return neg_likelihood

    class ConcordanceIndex:
        def __call__(self, log_hazard, event, time):
            """Simple C-index calculation"""
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

# ============================================================
# Configuration
# ============================================================

DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/rnaseq_only'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {DEVICE}")

# Hyperparameters
N_FOLDS = 3
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

# ============================================================
# Dataset
# ============================================================

class RNASeqDataset(Dataset):
    """RNA-seq only dataset"""

    def __init__(self, patient_ids, rnaseq_df, survival_df):
        self.patient_ids = patient_ids
        self.rnaseq_df = rnaseq_df
        self.survival_df = survival_df

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # RNA-seq features
        rnaseq_vec = self.rnaseq_df.loc[patient_id].values.astype(np.float32)

        # Survival data
        survival_row = self.survival_df.loc[patient_id]
        time = float(survival_row['survival_time'])
        event = int(survival_row['survival_status'])

        return {
            'rnaseq': torch.FloatTensor(rnaseq_vec),
            'time': torch.FloatTensor([time]),
            'event': torch.LongTensor([event])
        }

# ============================================================
# Model Architecture
# ============================================================

class RNASeqSurvivalModel(nn.Module):
    """Simple MLP for RNA-seq → Hazard prediction"""

    def __init__(self, input_dim=5005, hidden_dims=[1024, 512, 256]):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim

        # Output: log-hazard (single value)
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, rnaseq):
        log_hazard = self.mlp(rnaseq)  # (B, 1)
        return log_hazard

# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc='Training', leave=False):
        rnaseq = batch['rnaseq'].to(device)
        time = batch['time'].squeeze().to(device)
        event = batch['event'].squeeze().to(device)

        optimizer.zero_grad()

        log_hazard = model(rnaseq).squeeze()
        loss = neg_partial_log_likelihood(log_hazard, event, time)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    all_hazards = []
    all_events = []
    all_times = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating', leave=False):
            rnaseq = batch['rnaseq'].to(device)
            time = batch['time'].squeeze().to(device)
            event = batch['event'].squeeze().to(device)

            log_hazard = model(rnaseq).squeeze()
            loss = neg_partial_log_likelihood(log_hazard, event, time)

            total_loss += loss.item()

            all_hazards.append(log_hazard.cpu())
            all_events.append(event.cpu())
            all_times.append(time.cpu())

    all_hazards = torch.cat(all_hazards)
    all_events = torch.cat(all_events)
    all_times = torch.cat(all_times)

    # C-index
    cindex_metric = ConcordanceIndex()
    c_index = cindex_metric(all_hazards, all_events, all_times).item()

    return total_loss / len(loader), c_index

# ============================================================
# Load Data
# ============================================================

print("\n[1] Loading data...")

matching_table = pd.read_csv(os.path.join(DATA_DIR, 'full_matching_table.csv'))

# RNA-seq + Survival 모두 있는 환자만 선택
valid_patients = matching_table[
    (matching_table['has_rnaseq'] == True) &
    (matching_table['has_survival'] == True)
]

print(f"  ✓ Total patients with RNA-seq + Survival: {len(valid_patients)}")

# RNA-seq 데이터 로드
rnaseq_file = os.path.join(DATA_DIR, 'rnaseq_normalized_mapped.csv')
if not os.path.exists(rnaseq_file):
    rnaseq_file = os.path.join(DATA_DIR, 'rnaseq_normalized.csv')

rnaseq_df = pd.read_csv(rnaseq_file, index_col=0)

# Survival data는 matching_table에 포함되어 있음
# Create survival_df from matching_table
survival_df = valid_patients[['patient_id', 'survival_time', 'survival_status']].set_index('patient_id')

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
    train_dataset = RNASeqDataset(train_ids, rnaseq_df, survival_df)
    val_dataset = RNASeqDataset(val_ids, rnaseq_df, survival_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = RNASeqSurvivalModel(input_dim=rnaseq_df.shape[1]).to(DEVICE)
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
    'model': 'RNASeq-Only',
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
