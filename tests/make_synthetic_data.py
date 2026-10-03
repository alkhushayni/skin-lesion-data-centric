"""Smoke-test data. Usage:
    python tests/make_synthetic_data.py data/SYNTH
    python run_experiments.py --data-dir data/SYNTH --isic2019-dir data/SYNTH_ISIC2019 --ph2-dir data/SYNTH_PH2 \
        --cache-dir cache_s --out-dir results/smoke --models cnn --budget-models cnn --num-workers 0 --no-pretrained \
        --set max_epochs=1 patience=1 al_epochs_per_round=1 st_epochs_per_round=1 st_iterations=2 gan_epochs=1 \
        gan_min_steps=2 synth_target_per_class=40 diff_train_steps=3 diff_resolution=32 diff_sample_steps=2 \
        fixmatch_epochs=1 bald_mc_samples=2 quality_samples=8
"""
"""Create a tiny fake dataset with the HAM10000 layout, for smoke-testing the pipeline only.

Each class gets a different colour/shape so the models can learn something; lesions have 1-3 images.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

out = sys.argv[1] if len(sys.argv) > 1 else "data/SYNTH"
per_class_lesions = {"akiec": 12, "bcc": 14, "bkl": 18, "df": 10, "mel": 18, "nv": 40, "vasc": 10}
colors = {"akiec": (200, 80, 60), "bcc": (90, 160, 90), "bkl": (120, 90, 50), "df": (160, 120, 200),
          "mel": (40, 30, 30), "nv": (150, 100, 80), "vasc": (200, 40, 120)}
rng = np.random.default_rng(0)
os.makedirs(os.path.join(out, "HAM10000_images_part_1"), exist_ok=True)
rows, n = [], 0
for ci, (dx, nl) in enumerate(per_class_lesions.items()):
    for li in range(nl):
        lesion = f"HAM_{ci:02d}{li:04d}"
        for _ in range(int(rng.integers(1, 4))):
            img_id = f"ISIC_{n:07d}"
            n += 1
            im = Image.new("RGB", (60, 45), tuple(int(v) for v in rng.integers(180, 230, 3)))
            d = ImageDraw.Draw(im)
            r = int(rng.integers(8, 16)) + ci
            cx, cy = int(rng.integers(20, 40)), int(rng.integers(15, 30))
            col = tuple(int(np.clip(c + rng.integers(-25, 25), 0, 255)) for c in colors[dx])
            (d.ellipse if ci % 2 else d.rectangle)([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=col)
            im.save(os.path.join(out, "HAM10000_images_part_1", img_id + ".jpg"))
            rows.append(dict(lesion_id=lesion, image_id=img_id, dx=dx, dx_type="histo", age=50, sex="male",
                             localization="back"))
pd.DataFrame(rows).to_csv(os.path.join(out, "HAM10000_metadata.csv"), index=False)
print(f"{n} synthetic images in {out}")

# ---- fake external sets (layout of the real ones) -------------------------------------------
isic = os.path.join(os.path.dirname(out), "SYNTH_ISIC2019")
os.makedirs(os.path.join(isic, "ISIC_2019_Training_Input"), exist_ok=True)
cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
dx2col = {"mel": "MEL", "nv": "NV", "bcc": "BCC", "akiec": "AK", "bkl": "BKL", "df": "DF", "vasc": "VASC"}
gt = []
for j in range(70):
    dx = list(dx2col)[j % 7] if j < 63 else None
    img_id = f"ISIC_9{j:06d}"
    im = Image.new("RGB", (60, 45), (200, 190, 180))
    ci = list(per_class_lesions).index(dx) if dx else 0
    ImageDraw.Draw(im).ellipse([15, 10, 45, 35], fill=colors.get(dx, (0, 0, 0)))
    im.save(os.path.join(isic, "ISIC_2019_Training_Input", img_id + ".jpg"))
    row = {c: 0.0 for c in cols}
    row["SCC" if dx is None else dx2col[dx]] = 1.0
    gt.append(dict(image=img_id, **row))
gt.append(dict(image=rows[0]["image_id"], **{c: (1.0 if c == "NV" else 0.0) for c in cols}))   # a HAM image: must be excluded
Image.new("RGB", (60, 45)).save(os.path.join(isic, "ISIC_2019_Training_Input", rows[0]["image_id"] + ".jpg"))
pd.DataFrame(gt).to_csv(os.path.join(isic, "ISIC_2019_Training_GroundTruth.csv"), index=False)

ph2 = os.path.join(os.path.dirname(out), "SYNTH_PH2")
lines = ["|| Name || Histological Diagnosis || Clinical Diagnosis || Asymmetry ||"]
for j in range(30):
    name = f"IMD{j:03d}"
    d = os.path.join(ph2, "PH2 Dataset images", name, f"{name}_Dermoscopic_Image")
    os.makedirs(d, exist_ok=True)
    diag = 2 if j % 3 == 0 else j % 2
    im = Image.new("RGB", (76, 57), (205, 185, 175))
    ImageDraw.Draw(im).ellipse([20, 12, 56, 44], fill=colors["mel" if diag == 2 else "nv"])
    im.save(os.path.join(d, name + ".bmp"))
    lines.append(f"|| {name} ||  || {diag} || 0 ||")
open(os.path.join(ph2, "PH2_dataset.txt"), "w").write("\n".join(lines))
print("synthetic ISIC 2019 and PH2 written")
