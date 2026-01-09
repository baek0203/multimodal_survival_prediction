# TCGA-OV Multimodal Survival Prediction

Deep learning-based survival prediction for ovarian cancer using multimodal data (CT imaging, RNA-seq, clinical).

## Project Overview

This project implements a multimodal deep learning framework for survival prediction in TCGA-OV ovarian cancer patients using:
- **CT Imaging**: 3D medical images from TCIA
- **RNA-seq**: Gene expression data (5,005 genes)
- **Clinical Data**: Patient demographics and clinical variables

### Key Features
- Partial modality support with gating mechanism
- Late fusion architecture
- Cox proportional hazards loss with torchsurv
- MONAI DenseNet121 for CT encoding
- K-fold cross-validation

## Dataset Statistics

- **Total patients**: 608
- **With CT imaging**: 142 patients
- **With RNA-seq**: 427 patients
- **Complete multimodal**: 109 patients
- **With survival labels**: 348 patients

## Repository Structure

```
.
├── scripts/
│   ├── download/              # Phase 1: Data Download
│   │   ├── sample_patients.py
│   │   ├── download_tcga.py
│   │   ├── download_tcia.py
│   │   └── validate_data.py
│   │
│   ├── preprocessing/         # Phase 2: Data Preprocessing
│   │   ├── convert_dicom_to_nifti.py
│   │   ├── preprocess_genomic.py
│   │   ├── create_multimodal_dataset.py
│   │   ├── create_full_matching_table.py
│   │   └── map_rnaseq_via_gdc_api.py
│   │
│   ├── training/              # Phase 3: Model Training
│   │   ├── final_multimodal.py
│   │   ├── partial_modality_training.py
│   │   ├── train_rnaseq_only.py
│   │   ├── simple_fusion.py
│   │   ├── final_comparison.py
│   │   ├── flexible_multimodal.py
│   │   └── comprehensive_analysis.py
│   │
│   └── analysis/              # Phase 4: Results Analysis
│       ├── analyze_all_results.py
│       ├── create_report_figures.py
│       ├── generate_km_curves.py
│       └── evaluate_model.py
│
├── data/                      # Data directory (gitignored)
│   ├── clinical/              # Clinical data (CSV)
│   ├── imaging/               # CT imaging (DICOM & NIfTI)
│   ├── genomic/               # RNA-seq data
│   └── processed/             # Preprocessed datasets
│       ├── full_matching_table.csv
│       ├── rnaseq_normalized_mapped.csv
│       └── multimodal_dataset.py
│
├── models/                    # Trained models (gitignored)
│   ├── final/
│   ├── partial_modality/
│   └── simple_fusion/
│
├── results/                   # Training results (gitignored)
│   ├── final/
│   │   └── cv_results.json
│   ├── partial_modality/
│   │   └── cv_results.json
│   └── simple_fusion/
│       ├── cv_results.json
│       └── training_curves.png
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd tcga-ov-multimodal
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `torch` (PyTorch for deep learning)
- `torchsurv` (Survival analysis)
- `monai` (Medical imaging)
- `pandas`, `numpy` (Data processing)
- `SimpleITK`, `nibabel` (Medical image I/O)
- `scikit-learn` (Machine learning utilities)
- `tqdm` (Progress bars)

## Quick Start

### Phase 1: Download Data

```bash
cd scripts/download

# 1. Sample patients from TCGA-OV
python sample_patients.py

# 2. Download TCGA genomic data
python download_tcga.py

# 3. Download TCIA imaging data
python download_tcia.py

# 4. Validate downloaded data
python validate_data.py
```

### Phase 2: Preprocess Data

```bash
cd scripts/preprocessing

# 1. Convert DICOM to NIfTI
python convert_dicom_to_nifti.py

# 2. Preprocess RNA-seq data
python preprocess_genomic.py

# 3. Map RNA-seq UUIDs to patient IDs (requires GDC API)
python map_rnaseq_via_gdc_api.py

# 4. Create full matching table
python create_full_matching_table.py

# 5. Create multimodal dataset
python create_multimodal_dataset.py
```

### Phase 3: Train Models

```bash
cd scripts/training

# Option 1: Simple Late Fusion (Recommended for Quick Start)
python simple_fusion.py

# Option 2: Partial Modality Learning (Uses All 608 Patients)
python partial_modality_training.py

# Option 3: Complete Multimodal (88 Complete Patients)
python final_multimodal.py

# Option 4: Compare All Models
python final_comparison.py
```

### Phase 4: Analyze Results

```bash
cd scripts/analysis

# Generate comprehensive analysis
python analyze_all_results.py

# Create figures for publication
python create_report_figures.py

# Generate Kaplan-Meier survival curves
python generate_km_curves.py
```

## Model Architecture

### Late Fusion with Gating Mechanism

```
Input Modalities:
├── CT Image (64×64×32)    → 3D CNN / MONAI DenseNet121 → 128-dim
├── RNA-seq (5,005 genes)  → MLP (512 → 128)             → 128-dim
└── Clinical (age)         → MLP                         → 32-dim

Gating Mechanism:
├── Concat features + modality masks → Gate Network → 3 weights
└── Weighted fusion of modalities

Fusion & Prediction:
├── Weighted features → Fusion MLP (256 → 128)
└── Cox head (128 → 1) → Log hazard
```

### Key Components

1. **Partial Modality Support**
   - Missing modalities are masked with zeros
   - Gating network learns to weight available modalities
   - Patients without survival labels used for encoder training

2. **Loss Functions**
   - Cox proportional hazards loss (torchsurv)
   - Gate entropy regularization (prevents collapse)

3. **Evaluation Metrics**
   - Concordance Index (C-index)
   - Kaplan-Meier survival curves

## Results Structure

Training results are saved in JSON format:

```json
{
  "model": "MultiModalSurvivalNet (Late Fusion)",
  "c_index_mean": 0.5458,
  "c_index_std": 0.0503,
  "fold_results": [
    {"fold": 1, "best_c_index": 0.6275},
    {"fold": 2, "best_c_index": 0.5033},
    ...
  ],
  "hyperparameters": {
    "batch_size": 4,
    "learning_rate": 0.0001,
    "epochs": 50,
    "n_folds": 5
  }
}
```

### Expected Performance

- **Baseline (Image only)**: C-index 0.55 ± 0.07
- **Complete Multimodal**: C-index 0.55 ± 0.05
- **Partial Modality (608 patients)**: Target C-index 0.65 - 0.70

## Data Processing Pipeline

### RNA-seq Processing
1. Raw counts (TSV) → Normalized expression matrix
2. File UUID → Patient ID mapping via GDC API
3. Filtering to 5,005 genes
4. Z-score normalization

### Imaging Processing
1. DICOM → NIfTI conversion
2. Resampling to 64×64×32
3. Intensity normalization [0, 1]
4. 3D augmentation (optional)

### Survival Labels
- **Time**: Days to death or last follow-up
- **Event**: 1 (death) or 0 (censored)

## Configuration

### Hyperparameters

Edit hyperparameters directly in training scripts:

```python
# training/simple_fusion.py
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
N_FOLDS = 3
PATIENCE = 15
IMAGE_SIZE = (64, 64, 32)
```

### Data Paths

Data paths are configured in each script:

```python
DATA_DIR = 'data/processed'
IMAGING_DIR = 'data/imaging/nifti'
RNASEQ_FILE = 'data/processed/rnaseq_normalized_mapped.csv'
MATCHING_TABLE = 'data/processed/full_matching_table.csv'
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE`
   - Reduce `IMAGE_SIZE` to (32, 32, 16)
   - Use gradient checkpointing

2. **RNA-seq UUID Mapping Fails**
   - Re-run `preprocessing/map_rnaseq_via_gdc_api.py`
   - Check GDC API connectivity
   - Verify internet connection

3. **Missing Modality Data**
   - Use partial modality models (`training/partial_modality_training.py`)
   - Check `full_matching_table.csv` for data availability
   - Ensure Phase 1 & 2 completed successfully

4. **torchsurv Boolean Error**
   - Events must be boolean type
   - Already fixed in latest scripts
   - Convert: `event = event.bool()`

## Citation

If you use this code, please cite:

```bibtex
@article{tcga_ov_multimodal,
  title={Multimodal Deep Learning for Ovarian Cancer Survival Prediction},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Data Sources

- **TCGA-OV**: The Cancer Genome Atlas - Ovarian Cancer
  - Genomic data: [GDC Portal](https://portal.gdc.cancer.gov/)
  - Clinical data: GDC API

- **TCIA**: The Cancer Imaging Archive
  - CT imaging: [TCIA-OV Collection](https://www.cancerimagingarchive.net/)

## License

[Specify license here - e.g., MIT, Apache 2.0]

## Acknowledgments

- **torchsurv**: PyTorch survival analysis library
- **MONAI**: Medical Open Network for AI
- **TCGA Research Network**: For providing the TCGA-OV dataset
- **TCIA**: For providing medical imaging data

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

## Pretrained Models

Trained model checkpoints are **not included** in this repository due to file size constraints.

### Download Models

You can download pretrained models from:
- **Hugging Face Hub**: [your-username/tcga-ov-multimodal](https://huggingface.co/your-username/tcga-ov-multimodal) _(recommended)_
- **Google Drive**: [Download Link](https://drive.google.com/...)
- **Zenodo**: [DOI Link](https://zenodo.org/...)

### Model Structure

```
models/
├── simple_fusion/
│   ├── best_model_fold1.pth (67 MB)
│   ├── best_model_fold2.pth (67 MB)
│   └── best_model_fold3.pth (67 MB)
├── flexible_multimodal/
│   └── best_model_fold*.pth
└── rnaseq_only/
    └── best_model_fold*.pth (23 MB each)
```

### Loading Pretrained Models

```python
import torch
from scripts.training.simple_fusion import LateFusionModel

# Load model
model = LateFusionModel()
model.load_state_dict(torch.load('models/simple_fusion/best_model_fold1.pth'))
model.eval()

# Use for inference
with torch.no_grad():
    log_hazard = model(image, rnaseq)
```
