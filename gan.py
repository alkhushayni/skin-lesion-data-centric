"""DCGAN augmentation: one DCGAN per minority class (class-conditional by construction).

One generator is trained per class, on that class's training-fold images only, so every synthetic
image carries the label of the class it was trained on. Validation and test images are never used.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from config import CLASSES
from quality import image_quality


class Generator(nn.Module):
    """100-d noise -> 7x7x512 projection -> five transposed convolutions -> 224 x 224 RGB."""

    def __init__(self, nz=100):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(nz, 512 * 7 * 7, bias=False), nn.BatchNorm1d(512 * 7 * 7),
                                     nn.ReLU(True))
        def up(cin, cout, last=False):
            layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=last)]
            layers += [nn.Tanh()] if last else [nn.BatchNorm2d(cout), nn.ReLU(True)]
            return layers
        self.net = nn.Sequential(*up(512, 256), *up(256, 128), *up(128, 64), *up(64, 32), *up(32, 3, last=True))

    def forward(self, z):
        return self.net(self.project(z).view(-1, 512, 7, 7))


class Discriminator(nn.Module):
    """Five strided convolutions with LeakyReLU and dropout -> real/fake logit."""

    def __init__(self):
        super().__init__()
        def down(cin, cout, bn=True):
            layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            return layers + [nn.LeakyReLU(0.2, True), nn.Dropout(0.3)]
        self.net = nn.Sequential(*down(3, 32, bn=False), *down(32, 64), *down(64, 128), *down(128, 256),
                                 *down(256, 512), nn.Flatten(), nn.Linear(512 * 7 * 7, 1))

    def forward(self, x):
        return self.net(x)


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def train_class_gan(images, cfg, seed, log=""):
    """images: N x 224 x 224 x 3 uint8 of ONE class. Returns the trained generator and the steps run."""
    dev = cfg.device
    torch.manual_seed(seed)
    G, D = Generator(cfg.gan_nz).to(dev), Discriminator().to(dev)
    G.apply(_weights_init)
    D.apply(_weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=cfg.gan_lr, betas=(cfg.gan_beta1, cfg.gan_beta2))
    optD = torch.optim.Adam(D.parameters(), lr=cfg.gan_lr, betas=(cfg.gan_beta1, cfg.gan_beta2))
    bce = nn.BCEWithLogitsLoss()
    data = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    n, bs = len(data), min(cfg.gan_batch_size, len(images))
    steps_per_epoch = max(1, n // bs)
    total = max(cfg.gan_epochs * steps_per_epoch, cfg.gan_min_steps)
    g = torch.Generator().manual_seed(seed)
    for step in range(total):
        idx = torch.randint(0, n, (bs,), generator=g)
        real = data[idx].to(dev)
        flip = torch.rand(bs, generator=g) < 0.5                       # cheap augmentation for tiny classes
        real[flip.to(dev)] = real[flip.to(dev)].flip(3)
        z = torch.randn(bs, cfg.gan_nz, device=dev)
        fake = G(z)
        # discriminator (one-sided label smoothing 0.9)
        optD.zero_grad(set_to_none=True)
        lossD = bce(D(real), torch.full((bs, 1), 0.9, device=dev)) + bce(D(fake.detach()), torch.zeros(bs, 1, device=dev))
        lossD.backward()
        optD.step()
        # generator
        optG.zero_grad(set_to_none=True)
        lossG = bce(D(fake), torch.ones(bs, 1, device=dev))
        lossG.backward()
        optG.step()
        if (step + 1) % 500 == 0:
            print(f"    GAN {log} step {step + 1}/{total}  lossD {lossD.item():.3f}  lossG {lossG.item():.3f}", flush=True)
    return G, total


@torch.no_grad()
def generate(G, n, cfg, seed):
    G.eval()
    torch.manual_seed(seed)
    out = []
    for s in range(0, n, 64):
        z = torch.randn(min(64, n - s), cfg.gan_nz, device=cfg.device)
        x = G(z).add(1).mul(127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        out.append(x)
    return np.concatenate(out) if out else np.zeros((0, 224, 224, 3), np.uint8)


def gan_augmentation_for_fold(images, meta, train_idx, cfg, fold, cache_dir, compute_quality=True):
    """Train / load per-class generators for fold `fold`; return synthetic images, labels and a report."""
    assert cfg.image_size == 224, "the DCGAN generator produces 224 x 224 images"
    fold_dir = os.path.join(cache_dir, f"dcgan_fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    report_path = os.path.join(fold_dir, "report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    labels = meta.label.values[train_idx]
    xs, ys = [], []
    for c, name in enumerate(CLASSES):
        cls_idx = train_idx[labels == c]
        need = cfg.synth_target_per_class - len(cls_idx)
        if need <= 0:
            continue
        path = os.path.join(fold_dir, f"{name}.npy")
        if not os.path.exists(path):
            t0 = time.time()
            real = np.asarray(images[np.sort(cls_idx)])
            G, steps = train_class_gan(real, cfg, seed=cfg.seed + 100 * fold + c, log=f"fold{fold}/{name}")
            fake = generate(G, need, cfg, seed=cfg.seed + 100 * fold + c)
            np.save(path, fake)
            entry = dict(n_real=int(len(cls_idx)), n_generated=int(need), steps=int(steps),
                         train_seconds=time.time() - t0)
            if compute_quality:
                qfake = generate(G, cfg.quality_samples, cfg, seed=999 + c)
                entry.update(image_quality(real, qfake, cfg.device))
            if fold == 0:   # sample grid for the paper figure
                np.save(os.path.join(fold_dir, f"{name}_examples.npy"), fake[:8])
            report[name] = entry
            json.dump(report, open(report_path, "w"), indent=1)
            del G
        fake = np.load(path)
        xs.append(fake)
        ys.append(np.full(len(fake), c, dtype=int))
    if not xs:
        return np.zeros((0, cfg.image_size, cfg.image_size, 3), np.uint8), np.zeros(0, int), report
    return np.concatenate(xs), np.concatenate(ys), report
