"""Kaggle launcher (GPU T4 x2): downloads the datasets, builds the image caches once, then runs one
run_experiments.py process per GPU in parallel and finally runs analyze.py.

    python kaggle_run.py --stage test     # ~15 min: data check + speed benchmark
    python kaggle_run.py --stage A        # main run (fits one 12 h Kaggle session)
    python kaggle_run.py --stage B        # second run (needs stage A's output attached as input)

Everything is written to /kaggle/working (kept by Kaggle as the notebook output).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

W = "/kaggle/working"
OUT = os.path.join(W, "results", "main")
CACHE = os.path.join(W, "cache")

COMMON_SET = ["batch_size=32", "max_epochs=30", "patience=5", "num_workers=2", "diff_resolution=64",
              "diff_channels=64,128,128,256", "diff_train_steps=5000", "diff_batch_size=32", "diff_sample_steps=25",
              "st_iterations=3", "st_epochs_per_round=8", "al_epochs_per_round=6", "fixmatch_epochs=25",
              "bald_mc_samples=8", "quality_samples=150"]
BUDGET = ["sup50", "selftrain", "st_cwl", "fixmatch", "random", "al_entropy", "al_bald", "al_coreset"]
FULL = ["baseline", "cwl", "focal", "oversample"]

STAGES = {
    # one entry per GPU: (models, budget models, strategies in run order, extra --set, out sub-dir)
    # GPU1 (faster EfficientNet) reaches the diffusion condition first and trains the shared diffusion model;
    # GPU0 runs diffusion last in each fold, by which time the samples exist.
    "A": [(["resnet50"], ["resnet50"], FULL + BUDGET + ["diffusion"], [], None),
          (["efficientnet_b0"], ["efficientnet_b0"], FULL + ["diffusion"] + BUDGET, [], None)],
    "B": [(["vit_b16"], [], FULL + ["diffusion"], [], None),
          (["cnn", "mobilenetv2"], [], FULL + ["diffusion"], [], None)],
}
SENS = [("st_threshold", ["0.80", "0.95"], ["sup50", "selftrain"]),
        ("lr", ["3e-05", "0.0003"], ["baseline"]),
        ("focal_gamma", ["1.0", "3.0"], ["focal"])]


def download():
    import kagglehub
    roots = {}
    for key, slug in (("ham", "kmader/skin-cancer-mnist-ham10000"), ("isic", "andrewmvd/isic-2019"),
                      ("ph2", "kliuiev/ph2databaseaddi")):
        t = time.time()
        roots[key] = kagglehub.dataset_download(slug)
        print(f"{slug} -> {roots[key]}  ({time.time() - t:.0f}s)", flush=True)
    # HAM10000 metadata may sit in a sub-folder
    meta = glob.glob(os.path.join(roots["ham"], "**", "HAM10000_metadata.csv"), recursive=True)[0]
    roots["ham"] = os.path.dirname(meta)
    isic_gt = glob.glob(os.path.join(roots["isic"], "**", "ISIC_2019_Training_GroundTruth.csv"), recursive=True)
    if isic_gt:
        roots["isic"] = os.path.dirname(isic_gt[0])
    json.dump(roots, open(os.path.join(W, "data_roots.json"), "w"), indent=1)
    return roots


def build_caches(roots):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from data import load_metadata, build_cache
    from external import load_external
    cfg = Config()
    cfg.cache_dir, cfg.isic2019_dir, cfg.ph2_dir = CACHE, roots["isic"], roots["ph2"]
    t = time.time()
    meta = load_metadata(roots["ham"])
    build_cache(meta, CACHE, cfg.image_size)
    print(f"HAM10000 cache: {len(meta)} images, {meta.lesion_id.nunique()} lesions ({time.time() - t:.0f}s)", flush=True)
    ext = load_external(cfg, set(meta.image_id))
    return meta, ext


def cmd(roots, models, budget, strategies, deadline, extra_set=(), out=OUT):
    return [sys.executable, "run_experiments.py", "--data-dir", roots["ham"], "--isic2019-dir", roots["isic"],
            "--ph2-dir", roots["ph2"], "--cache-dir", CACHE, "--out-dir", out, "--models", *models,
            "--budget-models", *(budget or models), "--strategies", *strategies, "--deadline-hours", str(deadline),
            "--set", *COMMON_SET, *extra_set]


def benchmark(roots):
    import torch
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    sys.path.insert(0, ".")
    from config import Config
    from data import load_metadata, ImageSet, train_transform
    from models import build_model
    from train import make_loader
    import numpy as np
    cfg = Config(); cfg.batch_size, cfg.num_workers, cfg.device = 32, 2, "cuda"
    meta = load_metadata(roots["ham"])
    imgs = np.load(os.path.join(CACHE, "images_224.npy"), mmap_mode="r")
    ds = ImageSet(imgs, np.arange(1600), meta.label.values[:1600], train_transform())
    for name in ("resnet50", "efficientnet_b0", "vit_b16", "cnn"):
        m = build_model(name, 7, True).cuda()
        opt = torch.optim.Adam(m.parameters(), 1e-4)
        scaler = torch.amp.GradScaler("cuda")
        t, n = None, 0
        for i, (x, y) in enumerate(make_loader(ds, cfg, True)):
            if i == 5:
                torch.cuda.synchronize(); t = time.time(); n = 0
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(m(x).float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            n += len(y)
        torch.cuda.synchronize()
        print(f"  {name}: {n / (time.time() - t):.0f} img/s training -> ~{5600 / (n / (time.time() - t)):.0f} s per epoch", flush=True)
        del m
    from diffusion import train_diffusion
    cfg.diff_resolution, cfg.diff_batch_size, cfg.diff_train_steps = 64, 64, 60
    t = time.time()
    train_diffusion(np.asarray(imgs[:512]), meta.label.values[:512], cfg, seed=0)
    print(f"  diffusion 64px: {(time.time() - t) / 60:.2f} s per step -> 12k steps ~{(time.time() - t) / 60 * 12000 / 3600:.1f} h per fold", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["test", "A", "B"])
    ap.add_argument("--hours", type=float, default=10.5, help="no new runs start after this (Kaggle limit is 12 h)")
    ap.add_argument("--prev", default="", help="stage B: <user>/<notebook-slug> whose output holds stage A")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    roots = download()
    meta, ext = build_caches(roots)
    if a.stage == "test":
        benchmark(roots)
        return
    if a.stage == "B":   # restore stage A's results and diffusion samples from the earlier notebook version
        import kagglehub
        prev = kagglehub.notebook_output_download(a.prev) if a.prev else None
        src = glob.glob(os.path.join(prev, "**", "results", "main"), recursive=True) if prev else []
        if src and not os.path.exists(os.path.join(OUT, "config.json")):
            shutil.copytree(src[0], OUT, dirs_exist_ok=True)
            for d in glob.glob(os.path.join(os.path.dirname(os.path.dirname(src[0])), "cache", "diffusion_fold*")):
                shutil.copytree(d, os.path.join(CACHE, os.path.basename(d)), dirs_exist_ok=True)
            print("restored stage A:", len(glob.glob(os.path.join(OUT, "fold*", "*", "*.json"))), "runs", flush=True)
    procs = []
    for gpu, (models, budget, strategies, extra, _) in enumerate(STAGES[a.stage]):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        if a.stage == "B" and gpu == 1:      # GPU1 also runs the sensitivity analysis afterwards (ResNet50)
            script = [cmd(roots, models, budget, strategies, a.hours)]
            for param, vals, strats in SENS:
                for v in vals:
                    script.append(cmd(roots, ["resnet50"], ["resnet50"], strats, a.hours, [f"{param}={v}"],
                                      out=os.path.join(W, "results", "sens", f"{param}={v}")))
            sh = os.path.join(W, "gpu1_B.sh")
            open(sh, "w").write("\n".join(" ".join(f"'{x}'" for x in c) for c in script) + "\n")
            run = ["bash", sh]
        else:
            run = cmd(roots, models, budget, strategies, a.hours)
        log = open(os.path.join(W, f"log_{a.stage}_gpu{gpu}.txt"), "w")
        procs.append(subprocess.Popen(run, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(60)          # let the first process write folds/config before the second starts
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        done = len(glob.glob(os.path.join(OUT, "fold*", "*", "*.json")))
        print(f"[{time.strftime('%H:%M')}] finished runs: {done}", flush=True)
    print("exit codes:", [p.returncode for p in procs], flush=True)
    sens = os.path.join(W, "results", "sens")
    subprocess.run([sys.executable, "analyze.py", "--results", OUT, "--cache-dir", CACHE]
                   + (["--sensitivity", sens] if os.path.isdir(sens) else []))
    for d in glob.glob(os.path.join(CACHE, "diffusion_fold*")):   # drop the large weights, keep samples
        for f in glob.glob(os.path.join(d, "unet_ema.pt")):
            os.remove(f)
    for f in glob.glob(os.path.join(CACHE, "**", "images_224.npy"), recursive=True):   # image caches are rebuilt
        os.remove(f)


if __name__ == "__main__":
    main()
