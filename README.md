# Data-centric strategies for imbalanced skin lesion classification (HAM10000)

Code for the manuscript "Which data-centric strategies help imbalanced skin lesion classification? A leakage-controlled,
statistically tested benchmark with external validation". It reproduces every number, table and figure from the raw data.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                           # install the CUDA build of torch for your GPU first
```

Data (put each in its own folder; any sub-folder layout works):

| Dataset | Where to get it | Folder |
|---|---|---|
| HAM10000 | https://doi.org/10.7910/DVN/DBW86T (or Kaggle "skin-cancer-mnist-ham10000") | `data/HAM10000/` with `HAM10000_metadata.csv` + images |
| ISIC 2019 (external) | https://challenge.isic-archive.com/data/ → 2019 training images + ground truth | `data/ISIC2019/` with `ISIC_2019_Training_GroundTruth.csv` + `ISIC_2019_Training_Input/` |
| PH2 (external) | https://www.fc.up.pt/addi/ph2%20database.html | `data/PH2/` with `PH2_dataset.txt` + `PH2 Dataset images/` |

## 2. Run

```bash
# 1) main experiments (resumable: re-running skips finished runs)
python run_experiments.py --data-dir data/HAM10000 --isic2019-dir data/ISIC2019 --ph2-dir data/PH2 --out-dir results/main

# 2) sensitivity analysis on ResNet50 (Supplementary Table S5)
bash run_sensitivity.sh data/HAM10000

# 3) tables, statistics, figures (300 dpi), values.json and Supplementary_Material.docx
python analyze.py --results results/main --cache-dir cache --sensitivity results/sens

# 4) fill the manuscript (and cover letter) — writes *_FILLED.docx next to them
python fill_documents.py --values results/main/analysis/values.json \
    --docs Manuscript.docx Cover_Letter.docx --set CODE_URL=https://github.com/<user>/<repo> SUBMISSION_DATE="<date>"
```

**Several GPUs:** run one process per fold (`CUDA_VISIBLE_DEVICES=k python run_experiments.py --folds k ...`), all with
the same `--out-dir`, after the image cache exists (run fold 0 with `--models cnn --strategies baseline` once first).

**Compute (rough, one A100/RTX 4090-class GPU):** full-label conditions 6 models x 6 conditions x 5 folds, about 1.5-2 days;
label-budget conditions 3 models x 10 conditions x 5 folds, about 3-4 days (active learning and FixMatch dominate);
diffusion model about 4-6 h per fold. Total about one week on one GPU, or 1-2 days on five GPUs (one fold each).
To save time, run the full-label regime first (`--strategies baseline cwl focal oversample dcgan diffusion`).

**Kaggle (free GPU T4 x2):** `python kaggle_run.py --stage test`, then `--stage A` and `--stage B` (see the docstring).

**Smoke test** (CPU, synthetic images, numbers meaningless): see `tests/make_synthetic_data.py` and the command in its header.

## 3. Protocol

| Item | Setting |
|---|---|
| Splits | 5-fold CV, stratified and grouped by `lesion_id`; ~15 % lesion-grouped validation per training fold (early stopping only); every image tested once; splits saved in `results/main/splits/` |
| Models | baseline CNN, ResNet50, InceptionV3, MobileNetV2, EfficientNet-B0, ViT-B/16 (ImageNet weights, full fine-tuning) |
| Training | Adam, lr 1e-4, batch 16, max 50 epochs, early stopping on validation macro-F1 (patience 8); same initial weights for every condition of a model/fold |
| Full-label conditions | baseline, class-weighted loss, focal loss (γ=2), oversampling, DCGAN per minority class, class-conditional diffusion (DDPM + DDIM 50 steps + classifier-free guidance 2.0); synthetic top-up to 1500 images/class |
| Label-budget conditions (ResNet50, EfficientNet-B0, ViT-B/16) | supervised 50 %, self-training (θ=0.90, ≤5 iterations), self-training + CWL, FixMatch (τ=0.95, μ=2); random sampling and AL (least confidence, entropy, BALD with 10 MC-dropout samples, core-set k-centre, least confidence + CWL), 20 % → 50 % in 5 % steps |
| External tests | ISIC 2019 non-HAM images (7 HAM classes, SCC/UNK excluded); PH2 melanoma vs nevus |
| Metrics | accuracy, balanced accuracy, macro-F1, MCC, macro AUC, per-class F1, melanoma AUC / sensitivity at 80 % specificity / specificity at 90 % sensitivity, ECE, Brier |
| Statistics | pooled out-of-fold, image-level: exact McNemar; lesion-clustered bootstrap (2000) CIs and p; Holm per family (full-label; label-budget) |

## 4. Files

`config.py` settings · `data.py` data, folds, augmentation · `external.py` ISIC 2019 / PH2 · `models.py` architectures ·
`train.py` training, MC dropout, features · `gan.py` DCGAN · `diffusion.py` diffusion model · `quality.py` FID/KID/SSIM ·
`fixmatch.py` FixMatch · `strategies.py` all conditions · `run_experiments.py` runner · `run_sensitivity.sh` sensitivity ·
`analyze.py` tables/stats/figures · `supplement.py` supplementary file · `fill_documents.py` fills the Word files.
