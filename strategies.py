"""All experimental conditions.

Full-label regime (all labelled training images):
  baseline    standard cross-entropy
  cwl         class-weighted cross-entropy, w_c = N / (C n_c)
  focal       focal loss (gamma = 2)
  oversample  random oversampling (class-balanced sampling with replacement)
  dcgan       + synthetic minority images from one DCGAN per minority class
  diffusion   + synthetic minority images from one class-conditional diffusion model per fold
Label-budget regime (50 % of the training fold labelled):
  sup50       supervised on the 50 % labelled subset                     (reference for selftrain, st_cwl, fixmatch)
  selftrain   iterative pseudo-labelling of the other 50 %
  st_cwl      selftrain with class-weighted loss                          (combination)
  fixmatch    FixMatch on the same labelled / unlabelled split
  random      random queries from 20 % to 50 % labelled                    (reference for all active learning)
  al_lc / al_entropy / al_bald / al_coreset   active learning with least-confidence, entropy, BALD, core-set
  al_cwl      al_lc with class-weighted loss                               (combination)
"""
import os

import numpy as np
import torch

from config import CLASSES, MODELS
from data import ImageSet, train_transform, eval_transform
from metrics import compute_metrics, binary_melanoma_metrics
from models import build_model
from train import (train_model, predict, predict_mc_dropout, extract_features, class_weights_from,
                   balanced_sampler)

C = len(CLASSES)


class FoldContext:
    def __init__(self, images, meta, splits, cfg, fold, model_name, pretrained, work_dir, external=None):
        self.images, self.meta, self.s, self.cfg = images, meta, splits, cfg
        self.fold, self.model_name, self.pretrained, self.work_dir = fold, model_name, pretrained, work_dir
        self.y = meta.label.values
        self.val_ds = ImageSet(images, splits["val"], self.y[splits["val"]], eval_transform())
        self.test_ds = ImageSet(images, splits["test"], self.y[splits["test"]], eval_transform())
        self.external = {n: (m, ImageSet(imgs, np.arange(len(m)), m.label.values, eval_transform()))
                         for n, (m, imgs) in (external or {}).items()}
        self.model_seed = cfg.seed + 10 * fold + MODELS.index(model_name)

    def new_model(self):
        torch.manual_seed(self.model_seed)            # identical initial weights for every condition
        return build_model(self.model_name, C, self.pretrained)

    def train_set(self, idx, labels=None, extra_x=None, extra_y=None):
        labels = self.y[idx] if labels is None else labels
        return ImageSet(self.images, idx, labels, train_transform(), extra_x, extra_y)

    def pool_set(self, idx):
        return ImageSet(self.images, idx, self.y[idx], eval_transform())

    def evaluate(self, model):
        p = predict(model, self.test_ds, self.cfg)
        return p, compute_metrics(self.y[self.s["test"]], p)

    def state_path(self, name):
        return os.path.join(self.work_dir, f"_{name}_state.pt")


def _result(ctx, model, hist, seconds, extras=None):
    probs, test_m = ctx.evaluate(model)
    vp = predict(model, ctx.val_ds, ctx.cfg)
    ext_probs, ext_metrics = {}, {}
    for name, (m, ds) in ctx.external.items():
        p = predict(model, ds, ctx.cfg)
        ext_probs[name] = p
        ext_metrics[name] = binary_melanoma_metrics(m.label.values, p) if name == "ph2" \
            else compute_metrics(m.label.values, p)
    return dict(probs=probs, test_metrics=test_m, val_metrics=compute_metrics(ctx.y[ctx.s["val"]], vp),
                external_probs=ext_probs, external_metrics=ext_metrics,
                history=hist, train_seconds=seconds, extras=extras or {})


# ======================================================================= full-label regime
def run_full(ctx, tag, kind, synth=None):
    train_idx = ctx.s["train"]
    y = ctx.y[train_idx]
    extras, kw = {}, {}
    if kind == "cwl":
        w = class_weights_from(y, C)
        kw["class_weights"] = w
        extras["class_weights"] = dict(zip(CLASSES, map(float, w)))
    elif kind == "focal":
        kw["loss"] = "focal"
        extras["gamma"] = ctx.cfg.focal_gamma
    elif kind == "oversample":
        kw["sampler"] = balanced_sampler(y, ctx.model_seed)
    ex_x = ex_y = None
    if kind in ("dcgan", "diffusion"):
        ex_x, ex_y, report = synth
        extras["synthetic_per_class"] = {CLASSES[c]: int((ex_y == c).sum()) for c in range(C)}
        extras["generator_seconds_fold"] = sum(v.get("train_seconds", 0) + v.get("sample_seconds", 0)
                                               for v in report.values())
    model, hist, sec = train_model(ctx.new_model(), ctx.train_set(train_idx, extra_x=ex_x, extra_y=ex_y),
                                   ctx.val_ds, ctx.cfg, tag=tag, **kw)
    return _result(ctx, model, hist, sec, extras)


# ======================================================================= semi-supervised
def run_sup50(ctx, tag):
    L = ctx.s["sup50_labelled"]
    model, hist, sec = train_model(ctx.new_model(), ctx.train_set(L), ctx.val_ds, ctx.cfg, tag=tag)
    torch.save(model.state_dict(), ctx.state_path("sup50"))
    return _result(ctx, model, hist, sec, extras=dict(n_labelled=int(len(L))))


def _load_sup50(ctx, tag):
    model = ctx.new_model()
    if os.path.exists(ctx.state_path("sup50")):
        model.load_state_dict(torch.load(ctx.state_path("sup50"), map_location="cpu"))
        return model.to(ctx.cfg.device), 0.0
    model, _, sec = train_model(model, ctx.train_set(ctx.s["sup50_labelled"]), ctx.val_ds, ctx.cfg, tag=tag + " init")
    return model, sec


def run_selftrain(ctx, tag, cwl=False):
    cfg = ctx.cfg
    L, U = ctx.s["sup50_labelled"], ctx.s["sup50_unlabelled"]
    model, total_sec = _load_sup50(ctx, tag)
    best_val = compute_metrics(ctx.y[ctx.s["val"]], predict(model, ctx.val_ds, cfg))["macro_f1"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _, m0 = ctx.evaluate(model)
    rounds = [dict(iteration=0, n_pseudo=0, pseudo_acc=None, val_macro_f1=best_val,
                   test_accuracy=m0["accuracy"], test_macro_f1=m0["macro_f1"])]
    best_iter = 0
    for it in range(1, cfg.st_iterations + 1):
        pu = predict(model, ctx.pool_set(U), cfg)                 # pseudo-labels recomputed for all of U
        conf, pl = pu.max(1), pu.argmax(1)
        keep = conf >= cfg.st_threshold
        if not keep.any():
            rounds.append(dict(iteration=it, n_pseudo=0, pseudo_acc=None, stopped="no pseudo-labels above threshold"))
            break
        P = U[keep]
        idx = np.concatenate([L, P])
        labels = np.concatenate([ctx.y[L], pl[keep]])           # pseudo-labels, never the hidden truth
        kw = dict(class_weights=class_weights_from(labels, C)) if cwl else {}
        model, hist, sec = train_model(model, ctx.train_set(idx, labels), ctx.val_ds, cfg,
                                       max_epochs=cfg.st_epochs_per_round,
                                       patience=min(cfg.patience, cfg.st_epochs_per_round), tag=f"{tag} it{it}", **kw)
        total_sec += sec
        _, mt = ctx.evaluate(model)
        v = hist["best_val_macro_f1"]
        rounds.append(dict(iteration=it, n_pseudo=int(keep.sum()), pseudo_acc=float((pl[keep] == ctx.y[P]).mean()),
                           pseudo_per_class={CLASSES[c]: int((pl[keep] == c).sum()) for c in range(C)},
                           pseudo_f1_per_class={CLASSES[c]: float(_f1(ctx.y[P], pl[keep], c)) for c in range(C)},
                           val_macro_f1=v, test_accuracy=mt["accuracy"], test_macro_f1=mt["macro_f1"]))
        if v > best_val + 1e-6:
            best_val, best_iter = v, it
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)                             # iteration chosen on validation only
    return _result(ctx, model, {}, total_sec,
                   extras=dict(rounds=rounds, selected_iteration=best_iter, threshold=cfg.st_threshold, cwl=cwl))


def _f1(y, p, c):
    tp = ((y == c) & (p == c)).sum()
    d = (y == c).sum() + (p == c).sum()
    return 2 * tp / d if d else 0.0


def run_fixmatch(ctx, tag):
    from fixmatch import train_fixmatch
    L, U = ctx.s["sup50_labelled"], ctx.s["sup50_unlabelled"]
    model, hist, sec = train_fixmatch(ctx.new_model(), ctx.train_set(L), ctx.images, U, ctx.val_ds, ctx.cfg, tag=tag)
    return _result(ctx, model, hist, sec, extras=dict(threshold=ctx.cfg.fixmatch_threshold, mu=ctx.cfg.fixmatch_mu))


# ======================================================================= active learning
def _al_initial(ctx, tag):
    """Model trained on the 20 % seed pool; shared by every query strategy so all start identically."""
    path = ctx.state_path("al_init")
    model = ctx.new_model()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.to(ctx.cfg.device), 0.0
    model, _, sec = train_model(model, ctx.train_set(ctx.s["al_initial"]), ctx.val_ds, ctx.cfg, tag=tag + " init")
    torch.save(model.state_dict(), path)
    return model, sec


def _kcenter_greedy(f_lab, f_pool, k, device):
    """Core-set selection (Sener & Savarese, 2018): repeatedly take the pool point farthest from all centres."""
    P = torch.tensor(f_pool, device=device)
    mind = torch.full((len(P),), float("inf"), device=device)
    for s in range(0, len(f_lab), 1024):
        mind = torch.minimum(mind, torch.cdist(P, torch.tensor(f_lab[s:s + 1024], device=device)).min(1).values)
    chosen = []
    for _ in range(k):
        i = int(torch.argmax(mind))
        chosen.append(i)
        mind = torch.minimum(mind, torch.cdist(P, P[i:i + 1]).squeeze(1))
        mind[i] = -1
    return np.array(chosen)


def acquire(ctx, model, L, pool, k, method, rng):
    cfg = ctx.cfg
    if method == "random":
        return rng.choice(len(pool), size=k, replace=False)
    if method in ("lc", "entropy"):
        p = predict(model, ctx.pool_set(pool), cfg)
        score = 1 - p.max(1) if method == "lc" else -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    elif method == "bald":
        ps = predict_mc_dropout(model, ctx.pool_set(pool), cfg, cfg.bald_mc_samples)   # T x N x C
        H = lambda q: -(q * np.log(np.clip(q, 1e-12, 1))).sum(-1)
        score = H(ps.mean(0)) - H(ps).mean(0)
    elif method == "coreset":
        fl = extract_features(model, ctx.pool_set(np.array(L)), cfg)
        fp = extract_features(model, ctx.pool_set(pool), cfg)
        return _kcenter_greedy(fl, fp, k, cfg.device)
    else:
        raise ValueError(method)
    return np.argsort(-score, kind="stable")[:k]


def run_query_loop(ctx, tag, method, cwl=False):
    """method: random | lc | entropy | bald | coreset. Oracle = HAM10000 ground truth (simulated)."""
    cfg = ctx.cfg
    rng = np.random.default_rng(cfg.seed + ctx.fold)
    n_train = len(ctx.s["train"])
    target = int(round(cfg.label_budget * n_train))
    q = max(1, int(round(cfg.al_query_fraction * n_train)))
    L, pool = list(ctx.s["al_initial"]), np.array(ctx.s["al_pool"])
    model, sec = _al_initial(ctx, tag)
    total_sec = sec
    _, m = ctx.evaluate(model)
    curve = [dict(round=0, n_labelled=len(L), frac_labelled=len(L) / n_train, test_accuracy=m["accuracy"],
                  test_macro_f1=m["macro_f1"], queried_per_class=None)]
    r = 0
    while len(L) < target and len(pool):
        r += 1
        k = min(q, target - len(L), len(pool))
        pick = acquire(ctx, model, L, pool, k, method, rng)
        chosen = pool[pick]
        L.extend(chosen.tolist())
        pool = np.delete(pool, pick)
        Ly = ctx.y[np.array(L)]
        kw = dict(class_weights=class_weights_from(Ly, C)) if cwl else {}
        model, hist, s = train_model(model, ctx.train_set(np.array(L)), ctx.val_ds, cfg,
                                     max_epochs=cfg.al_epochs_per_round,
                                     patience=min(cfg.patience, cfg.al_epochs_per_round), tag=f"{tag} r{r}", **kw)
        total_sec += s
        _, m = ctx.evaluate(model)
        curve.append(dict(round=r, n_labelled=len(L), frac_labelled=len(L) / n_train,
                          val_macro_f1=hist["best_val_macro_f1"], test_accuracy=m["accuracy"],
                          test_macro_f1=m["macro_f1"],
                          queried_per_class={CLASSES[c]: int((ctx.y[chosen] == c).sum()) for c in range(C)}))
    return _result(ctx, model, {}, total_sec,
                   extras=dict(curve=curve, n_labelled=len(L), query_size=q, rounds=r, method=method, cwl=cwl,
                               initial_labelled=len(ctx.s["al_initial"]), init_seconds=sec))


QUERY = {"random": ("random", False), "al_lc": ("lc", False), "al_entropy": ("entropy", False),
         "al_bald": ("bald", False), "al_coreset": ("coreset", False), "al_cwl": ("lc", True)}


def run(ctx, strategy, tag, synth=None):
    if strategy in ("baseline", "cwl", "focal", "oversample", "dcgan", "diffusion"):
        return run_full(ctx, tag, strategy, synth)
    if strategy == "sup50":
        return run_sup50(ctx, tag)
    if strategy == "selftrain":
        return run_selftrain(ctx, tag)
    if strategy == "st_cwl":
        return run_selftrain(ctx, tag, cwl=True)
    if strategy == "fixmatch":
        return run_fixmatch(ctx, tag)
    if strategy in QUERY:
        return run_query_loop(ctx, tag, *QUERY[strategy])
    raise ValueError(strategy)
