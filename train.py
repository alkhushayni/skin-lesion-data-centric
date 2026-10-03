"""Training (unified configuration, early stopping on validation macro-F1), prediction, MC-dropout
prediction and feature extraction."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from metrics import compute_metrics
from models import head


def make_loader(ds, cfg, shuffle, sampler=None, batch_size=None):
    return DataLoader(ds, batch_size=batch_size or cfg.batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"),
                      drop_last=False, persistent_workers=cfg.num_workers > 0)


def _autocast(cfg):
    dev = cfg.device.split(":")[0]
    return torch.autocast(device_type=dev, enabled=cfg.amp and cfg.device.startswith("cuda"))


@torch.no_grad()
def predict(model, ds, cfg):
    model.eval()
    out = []
    for x, _ in make_loader(ds, cfg, shuffle=False):
        with _autocast(cfg):
            logits = model(x.to(cfg.device, non_blocking=True))
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 7), np.float32)


@torch.no_grad()
def predict_mc_dropout(model, ds, cfg, T):
    """T stochastic forward passes with dropout active (BatchNorm stays in inference mode)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    out = []
    for x, _ in make_loader(ds, cfg, shuffle=False):
        x = x.to(cfg.device, non_blocking=True)
        ps = []
        for _ in range(T):
            with _autocast(cfg):
                ps.append(torch.softmax(model(x).float(), 1).cpu().numpy())
        out.append(np.stack(ps, 0))                                   # T x B x C
    model.eval()
    return np.concatenate(out, 1)


@torch.no_grad()
def extract_features(model, ds, cfg):
    """Penultimate features = input of the final linear layer."""
    model.eval()
    feats = []
    h = head(model).register_forward_hook(lambda m, i, o: feats.append(i[0].detach().float().cpu()))
    for x, _ in make_loader(ds, cfg, shuffle=False):
        with _autocast(cfg):
            model(x.to(cfg.device, non_blocking=True))
    h.remove()
    return torch.cat(feats).numpy()


def focal_loss(logits, y, gamma, weight=None):
    logp = F.log_softmax(logits.float(), 1)
    logpt = logp.gather(1, y[:, None]).squeeze(1)
    loss = -((1 - logpt.exp()) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[y]
    return loss.mean()


def balanced_sampler(labels, seed):
    """Random oversampling: every class is drawn with equal probability (with replacement)."""
    counts = np.bincount(labels)
    w = 1.0 / counts[labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True,
                                 generator=g)


def train_model(model, train_ds, val_ds, cfg, class_weights=None, max_epochs=None, patience=None, tag="",
                loss="ce", sampler=None):
    """Train; keep the epoch with the best validation macro-F1. Returns (model, history, seconds)."""
    dev = cfg.device
    max_epochs = max_epochs or cfg.max_epochs
    patience = patience or cfg.patience
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=dev)
    if loss == "focal":
        loss_fn = lambda lg, y: focal_loss(lg, y, cfg.focal_gamma, w)
    else:
        ce = nn.CrossEntropyLoss(weight=w)
        loss_fn = lambda lg, y: ce(lg.float(), y)
    use_amp = cfg.amp and dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = make_loader(train_ds, cfg, shuffle=True, sampler=sampler)
    val_y = val_ds.all_labels()

    hist = dict(epoch=[], train_loss=[], train_acc=[], val_loss=[], val_acc=[], val_macro_f1=[])
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast(cfg):
                logits = model(x)
            l = loss_fn(logits, y)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            tot += l.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        vp = predict(model, val_ds, cfg)
        vm = compute_metrics(val_y, vp)
        vloss = float(F.nll_loss(torch.log(torch.tensor(vp).clamp_min(1e-12)), torch.tensor(val_y)).item())
        for k, v in zip(hist, (ep, tot / n, correct / n, vloss, vm["accuracy"], vm["macro_f1"])):
            hist[k].append(float(v))
        print(f"    {tag} epoch {ep:3d}  loss {tot / n:.4f}  acc {correct / n:.4f}  "
              f"val_loss {vloss:.4f}  val_acc {vm['accuracy']:.4f}  val_macroF1 {vm['macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best + 1e-6:
            best, bad = vm["macro_f1"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    hist["best_val_macro_f1"] = best
    hist["best_epoch"] = int(hist["epoch"][int(np.argmax(hist["val_macro_f1"]))])
    return model, hist, time.time() - t0


def class_weights_from(labels, n_classes):
    """w_c = N / (C * n_c), computed from the training labels only."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return len(labels) / (n_classes * counts)
