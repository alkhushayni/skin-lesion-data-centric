"""Run every (fold, model, condition). Resumable: finished runs are skipped.

    python run_experiments.py --data-dir data/HAM10000 --isic2019-dir data/ISIC2019 --ph2-dir data/PH2 \
        --out-dir results/main

Full-label conditions run for every model in --models; label-budget conditions only for --budget-models.
Per run: <out-dir>/fold<k>/<model>/<condition>.npz (test + external probabilities) and .json (metrics,
history, timing, details).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from config import Config, MODELS, STRATEGIES, FULL_STRATEGIES, BUDGET_STRATEGIES, AL_STRATEGIES
from data import load_metadata, build_cache, make_folds, fold_splits, save_splits
import strategies as S


def parse():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=cfg.data_dir)
    ap.add_argument("--isic2019-dir", default=cfg.isic2019_dir)
    ap.add_argument("--ph2-dir", default=cfg.ph2_dir)
    ap.add_argument("--cache-dir", default=cfg.cache_dir)
    ap.add_argument("--out-dir", default=cfg.out_dir)
    ap.add_argument("--models", nargs="+", default=cfg.models, choices=MODELS)
    ap.add_argument("--budget-models", nargs="+", default=cfg.budget_models, choices=MODELS)
    ap.add_argument("--strategies", nargs="+", default=cfg.strategies, choices=STRATEGIES)
    ap.add_argument("--folds", nargs="+", type=int, default=cfg.folds)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=cfg.num_workers)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--no-quality", action="store_true", help="skip FID/KID/SSIM (smoke tests only)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Config field")
    ap.add_argument("--deadline-hours", type=float, default=0, help="do not START new runs after this many hours (0 = none)")
    ap.add_argument("--skip-cache-build", action="store_true", help="fail instead of building the image cache")
    a = ap.parse_args()
    cfg.data_dir, cfg.isic2019_dir, cfg.ph2_dir = a.data_dir, a.isic2019_dir, a.ph2_dir
    cfg.cache_dir, cfg.out_dir = a.cache_dir, a.out_dir
    cfg.models, cfg.budget_models, cfg.strategies, cfg.folds = a.models, a.budget_models, a.strategies, a.folds
    cfg.device, cfg.num_workers = a.device, a.num_workers
    for kv in a.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        setattr(cfg, k, v.lower() in ("1", "true", "yes") if isinstance(old, bool) else type(old)(v))
    if cfg.device == "cpu":
        cfg.amp = False
    return cfg, a


T_START = time.time()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    cfg, a = parse()
    os.makedirs(cfg.out_dir, exist_ok=True)
    json.dump(dict(cfg.to_dict(), pretrained=not a.no_pretrained, torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
              open(os.path.join(cfg.out_dir, "config.json"), "w"), indent=1)
    seed_everything(cfg.seed)
    meta = load_metadata(cfg.data_dir)
    print(f"{len(meta)} images, {meta.lesion_id.nunique()} lesions; {meta.dx.value_counts().to_dict()}")
    images = build_cache(meta, cfg.cache_dir, cfg.image_size)
    from external import load_external
    external = load_external(cfg, set(meta.image_id))
    for name, (m, _) in external.items():
        m[["image_id", "dx"]].to_csv(os.path.join(cfg.out_dir, f"external_{name}.csv"), index=False)
    folds = make_folds(meta, cfg.n_folds, cfg.seed)
    meta.assign(fold=folds)[["image_id", "lesion_id", "dx", "fold"]].to_csv(os.path.join(cfg.out_dir, "folds.csv"),
                                                                           index=False)
    order = [s for s in STRATEGIES if s in cfg.strategies]
    for k in cfg.folds:
        splits = fold_splits(meta, folds, k, cfg)
        save_splits(meta, splits, os.path.join(cfg.out_dir, "splits", f"fold{k}.json"))
        print(f"\n=== fold {k}: train {len(splits['train'])}  val {len(splits['val'])}  test {len(splits['test'])}")
        synth = {}
        for m in cfg.models:
            work = os.path.join(cfg.out_dir, f"fold{k}", m)
            os.makedirs(work, exist_ok=True)
            todo = [s for s in order if s in FULL_STRATEGIES or m in cfg.budget_models]
            for st in todo:
                out = os.path.join(work, st)
                if a.deadline_hours and (time.time() - T_START) / 3600 > a.deadline_hours:
                    print(f"  deadline reached: not starting fold{k}/{m}/{st}", flush=True)
                    continue
                if os.path.exists(out + ".json"):
                    print(f"  skip fold{k}/{m}/{st} (done)")
                    continue
                seed_everything(cfg.seed + 10 * k)
                ctx = S.FoldContext(images, meta, splits, cfg, k, m, not a.no_pretrained, work, external)
                tag = f"[f{k} {m} {st}]"
                print(f"  run {tag}", flush=True)
                t0 = time.time()
                if st in ("dcgan", "diffusion") and st not in synth:
                    if st == "dcgan":
                        from gan import gan_augmentation_for_fold as gen
                    else:
                        from diffusion import diffusion_augmentation_for_fold as gen
                    synth[st] = gen(images, meta, splits["train"], cfg, k, cfg.cache_dir, compute_quality=not a.no_quality)
                    json.dump(synth[st][2], open(os.path.join(cfg.out_dir, f"{st}_report_fold{k}.json"), "w"), indent=1)
                res = S.run(ctx, st, tag, synth.get(st))
                test_idx = splits["test"]
                arrays = dict(probs=res["probs"].astype(np.float32), y=meta.label.values[test_idx],
                              image_id=meta.image_id.values[test_idx], lesion_id=meta.lesion_id.values[test_idx])
                for name, p in res["external_probs"].items():
                    arrays[f"ext_{name}"] = p.astype(np.float32)
                np.savez_compressed(out + ".npz", **arrays)
                info = {kk: v for kk, v in res.items() if kk not in ("probs", "external_probs")}
                info.update(fold=k, model=m, strategy=st, wall_seconds=time.time() - t0)
                json.dump(jsonable(info), open(out + ".json", "w"), indent=1)
                tm = res["test_metrics"]
                print(f"  done {tag}  acc {tm['accuracy']:.4f}  macroF1 {tm['macro_f1']:.4f}  mcc {tm['mcc']:.4f}", flush=True)
            # temporary checkpoints are removed once everything that needs them has finished
            needs = {"_sup50_state.pt": ["selftrain", "st_cwl"], "_al_init_state.pt": AL_STRATEGIES}
            for f, users in needs.items():
                p = os.path.join(work, f)
                if os.path.exists(p) and all(os.path.exists(os.path.join(work, s + ".json"))
                                             for s in users if s in todo):
                    os.remove(p)
    print("\nAll runs finished. Next: python analyze.py --results", cfg.out_dir)


if __name__ == "__main__":
    main()
