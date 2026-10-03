"""External test sets (never used for training, model selection or tuning).

ISIC 2019 (training set with ground truth): only images that are NOT in HAM10000 are used (BCN20000 and
MSK portions). Labels mapped to the HAM10000 classes: MEL->mel, NV->nv, BCC->bcc, AK->akiec, BKL->bkl,
DF->df, VASC->vasc. SCC (absent from HAM10000) and UNK are excluded.
    <isic2019_dir>/ISIC_2019_Training_GroundTruth.csv
    <isic2019_dir>/ISIC_2019_Training_Input/*.jpg          (any sub-folder layout works)

PH2: 200 dermoscopic images; melanoma vs (common + atypical) nevus, evaluated as melanoma vs nevus.
    <ph2_dir>/PH2_dataset.txt  (official file)  or  <ph2_dir>/ph2_labels.csv with columns image_id,diagnosis
    <ph2_dir>/**/IMDxxx.bmp
"""
import os
import re
from glob import glob

import numpy as np
import pandas as pd

from config import CLASSES
from data import build_cache

ISIC_MAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc"}


def _find_images(root, exts):
    out = {}
    for e in exts:
        for p in glob(os.path.join(root, "**", f"*.{e}"), recursive=True):
            out.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return out


def load_isic2019(root, ham_ids):
    gt = pd.read_csv(os.path.join(root, "ISIC_2019_Training_GroundTruth.csv"))
    cols = list(ISIC_MAP)
    gt = gt[gt[cols].sum(1) == 1].copy()                      # drops SCC and UNK
    gt["dx"] = gt[cols].idxmax(1).map(ISIC_MAP)
    gt["image_id"] = gt["image"].astype(str)
    base = gt.image_id.str.replace("_downsampled", "", regex=False)
    gt = gt[~base.isin(set(ham_ids))]                          # remove HAM10000 images
    paths = _find_images(root, ["jpg", "jpeg", "png"])
    gt = gt[gt.image_id.isin(paths)].copy()
    gt["path"] = gt.image_id.map(paths)
    gt["label"] = gt.dx.map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return gt.sort_values("image_id").reset_index(drop=True)[["image_id", "dx", "label", "path"]]


def load_ph2(root):
    paths = _find_images(root, ["bmp", "jpg", "png"])
    paths = {k: v for k, v in paths.items() if re.fullmatch(r"IMD\d+", k)}
    csv = os.path.join(root, "ph2_labels.csv")
    txt = glob(os.path.join(root, "**", "PH2_dataset.txt"), recursive=True)
    xls = glob(os.path.join(root, "**", "PH2_dataset.xls*"), recursive=True)
    if not os.path.exists(csv) and not txt and xls:
        raw = pd.read_excel(xls[0], header=None)
        hr = next(i for i in range(len(raw)) if raw.iloc[i].astype(str).str.contains("Image Name").any())
        df = raw.iloc[hr + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[hr]]
        name_col = next(c for c in df.columns if "Image Name" in c)
        diag_col = next(c for c in df.columns if "Clinical Diagnosis" in c)
        df = df[df[name_col].astype(str).str.match(r"IMD\d+")]
        df = pd.DataFrame(dict(image_id=df[name_col].astype(str).str.strip(),
                               mel=pd.to_numeric(df[diag_col], errors="coerce") == 2))
    elif os.path.exists(csv):
        df = pd.read_csv(csv)
        df["mel"] = df.diagnosis.astype(str).str.lower().str.contains("mel") | (df.diagnosis.astype(str) == "2")
    else:
        rows, header = [], None
        for line in open(txt[0] if txt else os.path.join(root, "PH2_dataset.txt"), encoding="latin-1"):
            parts = [p.strip() for p in line.split("||")]
            if len(parts) < 4:
                continue
            if header is None and any("Clinical" in p for p in parts):
                header = parts
                continue
            if header and re.fullmatch(r"IMD\d+", parts[1] or ""):
                rows.append(dict(image_id=parts[1], clinical=parts[header.index(next(p for p in header if "Clinical" in p))]))
        df = pd.DataFrame(rows)
        df["mel"] = df.clinical.astype(str).str.strip() == "2"
    df = df[df.image_id.isin(paths)].copy()
    df["path"] = df.image_id.map(paths)
    df["label"] = np.where(df.mel, CLASSES.index("mel"), CLASSES.index("nv"))
    df["dx"] = np.where(df.mel, "mel", "nv")
    return df.sort_values("image_id").reset_index(drop=True)[["image_id", "dx", "label", "path"]]


def load_external(cfg, ham_ids):
    """Returns {name: (meta, images)} for the configured external sets."""
    out = {}
    if cfg.isic2019_dir:
        m = load_isic2019(cfg.isic2019_dir, ham_ids)
        out["isic2019"] = (m, build_cache(m, os.path.join(cfg.cache_dir, "isic2019"), cfg.image_size))
        print(f"external ISIC 2019 (non-HAM): {len(m)} images; {m.dx.value_counts().to_dict()}")
    if cfg.ph2_dir:
        m = load_ph2(cfg.ph2_dir)
        out["ph2"] = (m, build_cache(m, os.path.join(cfg.cache_dir, "ph2"), cfg.image_size))
        print(f"external PH2: {len(m)} images; {m.dx.value_counts().to_dict()}")
    return out
