"""Experiment configuration (every value is the setting reported in the manuscript).

Change values only through the command line of run_experiments.py (--set key=value) so that the
settings actually used are written to results/<run>/config.json.
"""
from dataclasses import dataclass, field, asdict
from typing import List

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # HAM10000 'dx' codes, fixed order
MODELS = ["cnn", "resnet50", "inceptionv3", "mobilenetv2", "efficientnet_b0", "vit_b16"]

# full-label regime: every model; label-budget regime: the budget models only (compute)
FULL_STRATEGIES = ["baseline", "cwl", "focal", "oversample", "dcgan", "diffusion"]
BUDGET_STRATEGIES = ["sup50", "selftrain", "st_cwl", "fixmatch",
                     "random", "al_lc", "al_entropy", "al_bald", "al_coreset", "al_cwl"]
STRATEGIES = FULL_STRATEGIES + BUDGET_STRATEGIES
AL_STRATEGIES = ["random", "al_lc", "al_entropy", "al_bald", "al_coreset", "al_cwl"]


@dataclass
class Config:
    # ---------------- data ----------------
    data_dir: str = "data/HAM10000"          # HAM10000_metadata.csv + image folders
    isic2019_dir: str = ""                   # external test 1: ISIC 2019 training set (non-HAM images are used)
    ph2_dir: str = ""                        # external test 2: PH2
    cache_dir: str = "cache"
    image_size: int = 224
    n_folds: int = 5                         # lesion-grouped, stratified K-fold cross-validation
    val_fraction: float = 0.15               # of each training fold, lesion-grouped (early stopping only)
    seed: int = 42

    # ---------------- training ----------------
    lr: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 50                     # identical cap for all architectures
    patience: int = 8                        # early stopping on validation macro-F1
    num_workers: int = 4
    amp: bool = True
    focal_gamma: float = 2.0

    # ---------------- synthetic augmentation ----------------
    synth_target_per_class: int = 1500       # minority classes topped up to this many training images
    gan_epochs: int = 200                    # DCGAN (one per minority class)
    gan_min_steps: int = 5000
    gan_lr: float = 2e-4
    gan_beta1: float = 0.5
    gan_beta2: float = 0.999
    gan_nz: int = 100
    gan_batch_size: int = 32
    diff_resolution: int = 128               # class-conditional DDPM trained on all training classes jointly
    diff_channels: str = ""                  # U-Net widths, e.g. "64,128,128,256" ("" = default for the resolution)
    diff_train_steps: int = 40000
    diff_batch_size: int = 32
    diff_lr: float = 1e-4
    diff_sample_steps: int = 50              # DDIM steps
    diff_guidance: float = 2.0               # classifier-free guidance scale
    diff_cond_dropout: float = 0.1
    quality_samples: int = 200               # generated images per class for FID / KID / SSIM

    # ---------------- label-budget experiments ----------------
    label_budget: float = 0.50
    al_initial_fraction: float = 0.20
    al_query_fraction: float = 0.05
    al_epochs_per_round: int = 10
    bald_mc_samples: int = 10
    st_threshold: float = 0.90
    st_iterations: int = 5
    st_epochs_per_round: int = 10
    fixmatch_threshold: float = 0.95
    fixmatch_mu: int = 2                     # unlabelled : labelled images per batch
    fixmatch_lambda: float = 1.0
    fixmatch_epochs: int = 50

    # ---------------- what to run ----------------
    models: List[str] = field(default_factory=lambda: list(MODELS))
    budget_models: List[str] = field(default_factory=lambda: ["resnet50", "efficientnet_b0", "vit_b16"])
    strategies: List[str] = field(default_factory=lambda: list(STRATEGIES))
    folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    out_dir: str = "results/main"
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)
