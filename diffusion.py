"""Class-conditional diffusion augmentation (DDPM training, DDIM sampling, classifier-free guidance).

One model per fold is trained on ALL training-fold images (all seven classes jointly, conditioned on the
class label), so the minority classes share visual structure learned from the larger classes; this is
the main advantage over one GAN per tiny class. Validation and test images are never used.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import CLASSES
from quality import image_quality

C = len(CLASSES)


def _unet(cfg):
    from diffusers import UNet2DModel
    r = cfg.diff_resolution
    ch = (64, 128, 256, 256, 512) if r >= 128 else (64, 128, 256, 256)
    blocks_down = ("DownBlock2D",) * (len(ch) - 2) + ("AttnDownBlock2D", "DownBlock2D")
    blocks_up = ("UpBlock2D", "AttnUpBlock2D") + ("UpBlock2D",) * (len(ch) - 2)
    return UNet2DModel(sample_size=r, in_channels=3, out_channels=3, layers_per_block=2,
                       block_out_channels=ch, down_block_types=blocks_down, up_block_types=blocks_up,
                       num_class_embeds=C + 1)              # index C = "no class" (classifier-free guidance)


def train_diffusion(images, labels, cfg, seed, log=""):
    from diffusers import DDPMScheduler
    dev = cfg.device
    torch.manual_seed(seed)
    model = _unet(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    sched = DDPMScheduler(num_train_timesteps=1000)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.diff_lr)
    r = cfg.diff_resolution
    g = torch.Generator().manual_seed(seed)
    n = len(images)
    use_amp = cfg.amp and dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for step in range(cfg.diff_train_steps):
        idx = torch.randint(0, n, (min(cfg.diff_batch_size, n),), generator=g).numpy()
        x = torch.from_numpy(np.ascontiguousarray(images[np.sort(idx)])).permute(0, 3, 1, 2).float()
        y = torch.from_numpy(labels[np.sort(idx)]).long()
        x = F.interpolate(x, size=(r, r), mode="bilinear", antialias=True, align_corners=False) / 127.5 - 1
        flip = torch.rand(len(x), generator=g) < 0.5
        x[flip] = x[flip].flip(3)
        drop = torch.rand(len(y), generator=g) < cfg.diff_cond_dropout
        y[drop] = C
        x, y = x.to(dev), y.to(dev)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (len(x),), device=dev)
        with torch.autocast(device_type=dev.split(":")[0], enabled=use_amp, dtype=torch.float16):
            pred = model(sched.add_noise(x, noise, t), t, class_labels=y).sample
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
        if (step + 1) % 1000 == 0:
            print(f"    diffusion {log} step {step + 1}/{cfg.diff_train_steps}  loss {loss.item():.4f}", flush=True)
    return ema


@torch.no_grad()
def sample(model, cls, n, cfg, seed, out_size=224):
    from diffusers import DDIMScheduler
    dev, r = cfg.device, cfg.diff_resolution
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(cfg.diff_sample_steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for s in range(0, n, 32):
        b = min(32, n - s)
        x = torch.randn(b, 3, r, r, generator=gen).to(dev)
        yc = torch.full((b,), cls, device=dev, dtype=torch.long)
        yu = torch.full((b,), C, device=dev, dtype=torch.long)
        for t in sched.timesteps:
            tt = torch.full((b,), int(t), device=dev, dtype=torch.long)
            eps = model(torch.cat([x, x]), torch.cat([tt, tt]), class_labels=torch.cat([yc, yu])).sample
            ec, eu = eps.chunk(2)
            x = sched.step(eu + cfg.diff_guidance * (ec - eu), t, x).prev_sample
        x = F.interpolate(x.clamp(-1, 1), size=(out_size, out_size), mode="bicubic", align_corners=False)
        out.append(((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, out_size, out_size, 3), np.uint8)


def diffusion_augmentation_for_fold(images, meta, train_idx, cfg, fold, cache_dir, compute_quality=True):
    """The trained model is kept in the cache, so other synthetic amounts (sensitivity analysis) reuse it."""
    fold_dir = os.path.join(cache_dir, f"diffusion_fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    rp = os.path.join(fold_dir, "report.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {}
    labels = meta.label.values[train_idx]
    needs = {c: cfg.synth_target_per_class - int((labels == c).sum()) for c in range(C)}
    fname = lambda c: os.path.join(fold_dir, f"{CLASSES[c]}_n{needs[c]}.npy")
    todo = [c for c in range(C) if needs[c] > 0 and not os.path.exists(fname(c))]
    lock = os.path.join(fold_dir, "busy.lock")
    if todo:                                   # another process may be producing this fold's images: wait for it
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.time() - os.path.getmtime(lock) > 6 * 3600:   # stale lock from a killed process
                    os.remove(lock)
                    continue
                time.sleep(30)
        report = json.load(open(rp)) if os.path.exists(rp) else {}
        todo = [c for c in range(C) if needs[c] > 0 and not os.path.exists(fname(c))]
    try:
        _produce(todo, images, meta, train_idx, labels, needs, fname, cfg, fold, fold_dir, rp, report, compute_quality)
    finally:
        if os.path.exists(lock):
            os.remove(lock)
    xs, ys = [], []
    for c in range(C):
        if needs[c] > 0 and os.path.exists(fname(c)):
            a = np.load(fname(c))
            xs.append(a)
            ys.append(np.full(len(a), c, dtype=int))
    report = json.load(open(rp)) if os.path.exists(rp) else report
    if not xs:
        return np.zeros((0, cfg.image_size, cfg.image_size, 3), np.uint8), np.zeros(0, int), report
    return np.concatenate(xs), np.concatenate(ys), report


def _produce(todo, images, meta, train_idx, labels, needs, fname, cfg, fold, fold_dir, rp, report, compute_quality):
    if todo:
        wpath = os.path.join(fold_dir, "unet_ema.pt")
        t0 = time.time()
        if os.path.exists(wpath):
            model = _unet(cfg).to(cfg.device)
            model.load_state_dict(torch.load(wpath, map_location="cpu"))
            model.eval()
            train_sec = 0.0
        else:
            order = np.argsort(train_idx)
            imgs = images[np.asarray(train_idx)[order]]
            model = train_diffusion(np.asarray(imgs), labels[order], cfg, seed=cfg.seed + 1000 + fold, log=f"fold{fold}")
            torch.save(model.state_dict(), wpath)
            train_sec = time.time() - t0
        for c in todo:
            t1 = time.time()
            fake = sample(model, c, needs[c], cfg, seed=cfg.seed + 2000 + 10 * fold + c)
            np.save(fname(c), fake)
            name = CLASSES[c]
            entry = report.get(name, {})
            entry.update(n_real=int((labels == c).sum()), n_generated=int(needs[c]),
                         train_seconds=entry.get("train_seconds", 0.0) + train_sec / len(todo),
                         sample_seconds=time.time() - t1)
            if compute_quality and "fid" not in entry:
                real = np.asarray(images[np.sort(np.asarray(train_idx)[labels == c])])
                entry.update(image_quality(real, sample(model, c, cfg.quality_samples, cfg, seed=7 + c), cfg.device))
            if fold == 0 and not os.path.exists(os.path.join(fold_dir, f"{name}_examples.npy")):
                np.save(os.path.join(fold_dir, f"{name}_examples.npy"), fake[:8])
            report[name] = entry
            json.dump(report, open(rp, "w"), indent=1)
        del model
