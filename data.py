"""HAM10000 loading, resized-image cache, lesion-grouped folds and PyTorch datasets (Sections 3.1-3.2)."""
import json
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from config import CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- metadata + cache
def load_metadata(data_dir):
    """Read HAM10000_metadata.csv and attach the path of every image (searches all sub-folders)."""
    meta_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found. See README.md for the expected folder layout.")
    meta = pd.read_csv(meta_path)
    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)}
    missing = [i for i in meta.image_id if i not in paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images listed in the metadata were not found, e.g. {missing[:3]}")
    meta["path"] = meta.image_id.map(paths)
    meta["label"] = meta.dx.map({c: i for i, c in enumerate(CLASSES)})
    if meta.label.isna().any():
        raise ValueError(f"Unknown dx codes: {sorted(set(meta.dx) - set(CLASSES))}")
    meta = meta.sort_values("image_id").reset_index(drop=True)   # deterministic order
    meta["label"] = meta.label.astype(int)
    return meta


def build_cache(meta, cache_dir, size):
    """Resize every image once to size x size RGB uint8 and store as a memory-mapped array."""
    os.makedirs(cache_dir, exist_ok=True)
    arr_path = os.path.join(cache_dir, f"images_{size}.npy")
    ids_path = os.path.join(cache_dir, f"images_{size}_ids.json")
    if os.path.exists(arr_path) and os.path.exists(ids_path):
        if json.load(open(ids_path)) == list(meta.image_id):
            return np.load(arr_path, mmap_mode="r")
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8, shape=(len(meta), size, size, 3))
    for i, p in enumerate(meta.path):
        arr[i] = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
        if (i + 1) % 1000 == 0:
            print(f"  cached {i + 1}/{len(meta)} images", flush=True)
    arr.flush()
    json.dump(list(meta.image_id), open(ids_path, "w"))
    return np.load(arr_path, mmap_mode="r")


# ---------------------------------------------------------------- splits
def make_folds(meta, n_folds, seed):
    """Stratified K-fold grouped by lesion_id: all images of one lesion fall in the same fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(meta), -1, dtype=int)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(meta)), meta.label, groups=meta.lesion_id)):
        fold[test_idx] = k
    assert (fold >= 0).all()
    return fold


def grouped_holdout(idx, meta, fraction, seed):
    """Split idx into (keep, holdout) with ~fraction of images in holdout, grouped by lesion, stratified."""
    n_splits = max(2, int(round(1.0 / fraction)))
    sub = meta.iloc[idx]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep, hold = next(sgkf.split(np.zeros(len(idx)), sub.label, groups=sub.lesion_id))
    return np.asarray(idx)[keep], np.asarray(idx)[hold]


def sample_lesions(idx, meta, fraction, seed):
    """Choose whole lesions, class by class, until ~fraction of the images in idx are selected."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    sub = meta.iloc[idx]
    chosen = []
    for c in sorted(sub.label.unique()):
        cls_rows = sub[sub.label == c]
        lesions = np.array(sorted(cls_rows.lesion_id.unique()), dtype=object)
        rng.shuffle(lesions)
        target = max(1, int(round(fraction * len(cls_rows))))
        sizes = cls_rows.groupby("lesion_id").size()
        n = 0
        for les in lesions:
            if n >= target:
                break
            chosen.append(les)
            n += sizes[les]
    mask = sub.lesion_id.isin(set(chosen)).values
    return idx[mask], idx[~mask]


def fold_splits(meta, folds, k, cfg):
    """All index sets for fold k: train / val / test plus the 50 % labelled subset and the AL seed pool."""
    seed = cfg.seed + k
    test = np.where(folds == k)[0]
    rest = np.where(folds != k)[0]
    train, val = grouped_holdout(rest, meta, cfg.val_fraction, seed)
    labelled50, unlabelled50 = sample_lesions(train, meta, cfg.label_budget, seed)
    al_init, al_pool = sample_lesions(train, meta, cfg.al_initial_fraction, seed + 1000)
    s = dict(train=train, val=val, test=test, sup50_labelled=labelled50, sup50_unlabelled=unlabelled50,
             al_initial=al_init, al_pool=al_pool)
    # leakage checks: no lesion may appear in two of train / val / test
    les = {n: set(meta.lesion_id.iloc[s[n]]) for n in ("train", "val", "test")}
    assert not (les["train"] & les["val"]) and not (les["train"] & les["test"]) and not (les["val"] & les["test"])
    return s


def save_splits(meta, splits, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({n: list(meta.image_id.iloc[v]) for n, v in splits.items()}, open(path, "w"))


# ---------------------------------------------------------------- transforms + datasets
def train_transform():
    """Section 3.2: rotation, horizontal/vertical flips, zoom/scale, translation, brightness only."""
    return T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImageSet(Dataset):
    """Images taken from the cached array (by index) and/or extra in-memory images (e.g. GAN samples)."""

    def __init__(self, images, idx, labels, transform, extra_images=None, extra_labels=None):
        self.images, self.idx = images, np.asarray(idx, dtype=int)
        self.labels = np.asarray(labels, dtype=int)
        assert len(self.labels) == len(self.idx)
        self.extra_images = extra_images if extra_images is not None else np.zeros((0, 1, 1, 3), np.uint8)
        self.extra_labels = np.asarray(extra_labels if extra_labels is not None else [], dtype=int)
        self.transform = transform

    def __len__(self):
        return len(self.idx) + len(self.extra_labels)

    def __getitem__(self, i):
        if i < len(self.idx):
            img, y = self.images[self.idx[i]], self.labels[i]
        else:
            j = i - len(self.idx)
            img, y = self.extra_images[j], self.extra_labels[j]
        x = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1)   # uint8 CHW
        return self.transform(x), int(y)

    def all_labels(self):
        return np.concatenate([self.labels, self.extra_labels])
