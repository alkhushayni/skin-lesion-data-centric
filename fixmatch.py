"""FixMatch (Sohn et al., 2020): confidence-thresholded pseudo-labels from weakly augmented views,
consistency on strongly augmented views of the same unlabelled images."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from data import train_transform, IMAGENET_MEAN, IMAGENET_STD
from metrics import compute_metrics
from train import predict, make_loader, _autocast


def strong_transform():
    """Weak geometry + RandAugment + cutout (random erasing)."""
    return T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandAugment(num_ops=2, magnitude=10),
        T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=1.0, scale=(0.02, 0.15), value=0),
    ])


class UnlabelledPair(Dataset):
    def __init__(self, images, idx):
        self.images, self.idx = images, np.asarray(idx)
        self.weak, self.strong = train_transform(), strong_transform()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.images[self.idx[i]], copy=True)).permute(2, 0, 1)
        return self.weak(x), self.strong(x)


def train_fixmatch(model, labelled_ds, images, unlabelled_idx, val_ds, cfg, tag=""):
    dev = cfg.device
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    use_amp = cfg.amp and dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lab = make_loader(labelled_ds, cfg, shuffle=True)
    unl = DataLoader(UnlabelledPair(images, unlabelled_idx), batch_size=cfg.batch_size * cfg.fixmatch_mu, shuffle=True,
                     num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers > 0)
    unl_it = iter(unl)
    val_y = val_ds.all_labels()
    hist = dict(epoch=[], loss_l=[], loss_u=[], mask_rate=[], val_macro_f1=[])
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(1, cfg.fixmatch_epochs + 1):
        model.train()
        sl = su = sm = 0.0
        nb = 0
        for x, y in lab:
            try:
                uw, us = next(unl_it)
            except StopIteration:
                unl_it = iter(unl)
                uw, us = next(unl_it)
            x, y, uw, us = (t.to(dev, non_blocking=True) for t in (x, y, uw, us))
            with _autocast(cfg):
                logits = model(torch.cat([x, uw, us]))
            lx, lw, ls = logits.float().split([len(x), len(uw), len(us)])
            loss_l = F.cross_entropy(lx, y)
            probs = torch.softmax(lw.detach(), 1)
            conf, pl = probs.max(1)
            mask = (conf >= cfg.fixmatch_threshold).float()
            loss_u = (F.cross_entropy(ls, pl, reduction="none") * mask).mean()
            loss = loss_l + cfg.fixmatch_lambda * loss_u
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sl, su, sm, nb = sl + loss_l.item(), su + loss_u.item(), sm + mask.mean().item(), nb + 1
        vm = compute_metrics(val_y, predict(model, val_ds, cfg))
        for k, v in zip(hist, (ep, sl / nb, su / nb, sm / nb, vm["macro_f1"])):
            hist[k].append(float(v))
        print(f"    {tag} epoch {ep:3d}  loss_l {sl / nb:.4f}  loss_u {su / nb:.4f}  mask {sm / nb:.3f}  "
              f"val_macroF1 {vm['macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best + 1e-6:
            best, bad = vm["macro_f1"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    hist["best_val_macro_f1"] = best
    return model, hist, time.time() - t0
