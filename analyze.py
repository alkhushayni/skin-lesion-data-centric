"""Aggregate all runs into the manuscript's tables, statistics, figures and generated sentences.

    python analyze.py --results results/main --cache-dir cache [--sensitivity results/sens]

Writes to <results>/analysis/: per_run_metrics.csv, summary.csv, comparisons.csv, per_class_f1.csv,
external_metrics.csv, values.json (all numbers/sentences used by fill_documents.py), fig_*.png (300 dpi)
and Supplementary_Material.docx.
"""
import argparse
import glob
import json
import os
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_curve, auc as sk_auc

from config import CLASSES, MODELS, STRATEGIES, FULL_STRATEGIES, BUDGET_STRATEGIES
from metrics import compute_metrics, binary_melanoma_metrics, confusion
from models import DISPLAY_NAMES

C = len(CLASSES)
MKEY = {"cnn": "CNN", "resnet50": "RES", "inceptionv3": "INC", "mobilenetv2": "MOB", "efficientnet_b0": "EFF",
        "vit_b16": "VIT"}
SKEY = {"baseline": "BASE", "cwl": "CWL", "focal": "FOC", "oversample": "OVS", "dcgan": "DCG", "diffusion": "DIF",
        "sup50": "SUP", "selftrain": "ST", "st_cwl": "STC", "fixmatch": "FXM", "random": "RND", "al_lc": "ALC",
        "al_entropy": "ALE", "al_bald": "ALB", "al_coreset": "ALS", "al_cwl": "ALW"}
SNAME = {"baseline": "Baseline", "cwl": "Class-weighted loss", "focal": "Focal loss", "oversample": "Oversampling",
         "dcgan": "DCGAN augmentation", "diffusion": "Diffusion augmentation", "sup50": "Supervised (50%)",
         "selftrain": "Self-training", "st_cwl": "Self-training + CWL", "fixmatch": "FixMatch",
         "random": "Random sampling", "al_lc": "AL: least confidence", "al_entropy": "AL: entropy",
         "al_bald": "AL: BALD", "al_coreset": "AL: core-set", "al_cwl": "AL: least confidence + CWL"}
METRICS = {"accuracy": "ACC", "balanced_accuracy": "BACC", "macro_f1": "MF1", "mcc": "MCC", "auc": "AUC",
           "mel_auc": "MAUC", "mel_sens_at_spec80": "MSENS", "mel_spec_at_sens90": "MSPEC", "ece": "ECE",
           "brier": "BRIER"}
PCT = {"accuracy", "balanced_accuracy", "mel_sens_at_spec80", "mel_spec_at_sens90"}
FULL_FAMILY = [(s, "baseline") for s in ["cwl", "focal", "oversample", "dcgan", "diffusion"]]
BUDGET_FAMILY = [("selftrain", "sup50"), ("st_cwl", "sup50"), ("fixmatch", "sup50"), ("al_lc", "random"),
                 ("al_entropy", "random"), ("al_bald", "random"), ("al_coreset", "random"), ("al_cwl", "al_lc"),
                 ("st_cwl", "selftrain")]
MINORITY = [c for c in CLASSES if c != "nv"]


# ------------------------------------------------------------------ loading
def load_runs(res_dir):
    runs = {}
    for js in glob.glob(os.path.join(res_dir, "fold*", "*", "*.json")):
        info = json.load(open(js))
        z = np.load(js[:-5] + ".npz", allow_pickle=True)
        runs[(info["fold"], info["model"], info["strategy"])] = dict(
            info=info, probs=z["probs"], y=z["y"], image_id=z["image_id"], lesion_id=z["lesion_id"],
            ext={k[4:]: z[k] for k in z.files if k.startswith("ext_")})
    if not runs:
        raise SystemExit(f"no finished runs under {res_dir}")
    return runs


def pooled(runs, m, s):
    parts = [runs[k] for k in sorted(runs) if k[1] == m and k[2] == s]
    if not parts:
        return None
    return dict(probs=np.concatenate([p["probs"] for p in parts]), y=np.concatenate([p["y"] for p in parts]),
                image_id=np.concatenate([p["image_id"] for p in parts]),
                lesion_id=np.concatenate([p["lesion_id"] for p in parts]))


# ------------------------------------------------------------------ statistics
def macro_f1_w(y, pred, w):
    cm = np.bincount(y * C + pred, weights=w, minlength=C * C).reshape(C, C)
    tp, d = np.diag(cm), cm.sum(0) + cm.sum(1)
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0).mean()


def paired_test(a, b, B, seed):
    ib = {x: j for j, x in enumerate(b["image_id"])}
    pairs = [(i, ib[x]) for i, x in enumerate(a["image_id"]) if x in ib]
    ia, jb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    y = a["y"][ia]
    pa, pb = a["probs"][ia].argmax(1), b["probs"][jb].argmax(1)
    ca, cb = pa == y, pb == y
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p_mn = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    uniq, inv = np.unique(a["lesion_id"][ia], return_inverse=True)
    rng = np.random.default_rng(seed)
    d_f1, d_acc = np.empty(B), np.empty(B)
    for r in range(B):
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[inv].astype(float)
        d_f1[r] = macro_f1_w(y, pa, w) - macro_f1_w(y, pb, w)
        d_acc[r] = ((w * ca).sum() - (w * cb).sum()) / w.sum()
    one = np.ones(len(y))
    return dict(n=len(y), d_macro_f1=macro_f1_w(y, pa, one) - macro_f1_w(y, pb, one),
                f1_lo=np.percentile(d_f1, 2.5), f1_hi=np.percentile(d_f1, 97.5), d_acc=ca.mean() - cb.mean(),
                acc_lo=np.percentile(d_acc, 2.5), acc_hi=np.percentile(d_acc, 97.5), n01=n01, n10=n10,
                p_mcnemar=p_mn, p_boot=max(min(1.0, 2 * min((d_f1 <= 0).mean(), (d_f1 >= 0).mean())), 1 / B))


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ formatting
fmt_ms = lambda m, s, pct: f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"
f_p = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
f_d = lambda d, lo, hi: f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
sig = lambda r: r.p_boot_holm < 0.05 and (r.f1_lo > 0 or r.f1_hi < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--sensitivity", default=None, help="folder with one sub-folder per setting (run_sensitivity.sh)")
    a = ap.parse_args()
    out = os.path.join(a.results, "analysis")
    os.makedirs(out, exist_ok=True)
    runs = load_runs(a.results)
    V = {}

    # ---------------- per-run metrics (internal + external)
    rows, erows = [], []
    for (k, m, s), r in sorted(runs.items()):
        info = r["info"]
        gen_min = info["extras"].get("generator_seconds_fold", 0) / 60
        rows.append(dict(fold=k, model=m, strategy=s, **compute_metrics(r["y"], r["probs"]),
                         train_minutes=info["train_seconds"] / 60, generator_minutes=gen_min))
        for name, met in info.get("external_metrics", {}).items():
            erows.append(dict(fold=k, model=m, strategy=s, dataset=name, **met))
    df, edf = pd.DataFrame(rows), pd.DataFrame(erows)
    df.to_csv(os.path.join(out, "per_run_metrics.csv"), index=False)
    edf.to_csv(os.path.join(out, "external_metrics.csv"), index=False)
    models = [m for m in MODELS if m in set(df.model)]
    strategies = [s for s in STRATEGIES if s in set(df.strategy)]
    budget_models = [m for m in models if set(df[df.model == m].strategy) & set(BUDGET_STRATEGIES)]
    df.groupby(["model", "strategy"]).agg(["mean", "std"]).to_csv(os.path.join(out, "summary.csv"))
    V["N_FOLDS"] = str(df.fold.nunique())
    V["N_MODELS"] = str(len(models))

    sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0
    for m in models:
        for s in strategies:
            sub = df[(df.model == m) & (df.strategy == s)]
            if sub.empty:
                continue
            for met, key in METRICS.items():
                V[f"{SKEY[s]}_{MKEY[m]}_{key}"] = fmt_ms(sub[met].mean(), sd(sub[met]), met in PCT)
    for s in strategies:   # mean over architectures (fold means first)
        sub = df[df.strategy == s].groupby("model").mean(numeric_only=True)
        for met, key in METRICS.items():
            V[f"{SKEY[s]}_MEAN_{key}"] = fmt_ms(sub[met].mean(), sd(sub[met]), met in PCT)
        t = df[df.strategy == s]
        V[f"TIME_{SKEY[s]}"] = f"{t.train_minutes.mean() + t.generator_minutes.mean() / max(1, len(models)):.1f}"

    # ---------------- external validation
    if not edf.empty:
        for name in edf.dataset.unique():
            e = edf[edf.dataset == name]
            for s in strategies:
                sub = e[e.strategy == s].groupby("model").mean(numeric_only=True)
                if sub.empty:
                    continue
                cols = ["macro_f1", "balanced_accuracy", "accuracy", "mel_auc", "mel_sens_at_spec80"] if name != "ph2" \
                    else ["mel_auc", "mel_sens_at_spec80", "mel_spec_at_sens90", "mel_vs_nv_accuracy"]
                for met in cols:
                    V[f"EXT_{name.upper()}_{SKEY[s]}_{METRICS.get(met, met.upper())}"] = fmt_ms(
                        sub[met].mean(), sd(sub[met]), met in PCT or met == "mel_vs_nv_accuracy")
            for m in models:
                sub = e[(e.strategy == "baseline") & (e.model == m)]
                if not sub.empty:
                    met = "macro_f1" if name != "ph2" else "mel_auc"
                    V[f"EXT_{name.upper()}_BASE_{MKEY[m]}_{METRICS[met]}"] = fmt_ms(sub[met].mean(), sd(sub[met]), False)
        if "isic2019" in set(edf.dataset):
            i_mf1 = df[df.strategy == "baseline"].macro_f1.mean()
            e_mf1 = edf[(edf.dataset == "isic2019") & (edf.strategy == "baseline")].macro_f1.mean()
            V["DROP_ISIC_MF1"] = f"{i_mf1:.3f} to {e_mf1:.3f}"
        ip = os.path.join(a.results, "external_isic2019.csv")
        if os.path.exists(ip):
            vc = pd.read_csv(ip).dx.value_counts()
            for c in CLASSES:
                V[f"EXTN_{c.upper()}"] = str(int(vc.get(c, 0)))
        for name in edf.dataset.unique():
            V[f"N_EXT_{name.upper()}"] = str(len(pd.read_csv(os.path.join(a.results, f"external_{name}.csv")))) \
                if os.path.exists(os.path.join(a.results, f"external_{name}.csv")) else ""

    # ---------------- per-class F1 (pooled out-of-fold), mean over the architectures that ran the condition
    pcf = []
    for m in models:
        for s in strategies:
            p = pooled(runs, m, s)
            if p is not None:
                tm = compute_metrics(p["y"], p["probs"])
                pcf.append(dict(model=m, strategy=s, **{c: tm[f"f1_{c}"] for c in CLASSES}))
    pcf = pd.DataFrame(pcf)
    pcf.to_csv(os.path.join(out, "per_class_f1.csv"), index=False)
    for s in strategies:
        sub = pcf[pcf.strategy == s]
        for c in CLASSES:
            V[f"PCF1_{SKEY[s]}_{c.upper()}"] = f"{sub[c].mean():.3f}"

    # ---------------- paired comparisons (two Holm families)
    comps = []
    for fam, pairs, ms in (("full", FULL_FAMILY, models), ("budget", BUDGET_FAMILY, budget_models)):
        for m in ms:
            for s, ref in pairs:
                A, B = pooled(runs, m, s), pooled(runs, m, ref)
                if A is not None and B is not None:
                    comps.append(dict(family=fam, model=m, strategy=s, reference=ref,
                                      **paired_test(A, B, a.bootstrap, zlib.crc32(f"{m}|{s}|{ref}".encode()))))
    cdf = pd.DataFrame(comps)
    if not cdf.empty:
        for fam in cdf.family.unique():
            i = cdf.family == fam
            cdf.loc[i, "p_mcnemar_holm"] = holm(cdf.loc[i, "p_mcnemar"])
            cdf.loc[i, "p_boot_holm"] = holm(cdf.loc[i, "p_boot"])
        cdf.to_csv(os.path.join(out, "comparisons.csv"), index=False)
        for _, r in cdf.iterrows():
            key = f"CMP_{SKEY[r.strategy]}vs{SKEY[r.reference]}_{MKEY[r.model]}"
            V[key] = f"{r.d_macro_f1:+.3f}{'*' if sig(r) else ''}\n[{r.f1_lo:+.3f}, {r.f1_hi:+.3f}]"
            V[key + "_DMF1"] = f_d(r.d_macro_f1, r.f1_lo, r.f1_hi)
            V[key + "_PBS"] = f_p(r.p_boot_holm)
            V[key + "_PMN"] = f_p(r.p_mcnemar_holm)
        for fam in ("full", "budget"):
            V[f"N_CMP_{fam.upper()}"] = str(int((cdf.family == fam).sum()))

    # ---------------- ranking within regimes
    for regime in (FULL_STRATEGIES, BUDGET_STRATEGIES):
        present = [s for s in regime if s in strategies]
        ms = models if regime is FULL_STRATEGIES else budget_models
        ranks = {s: [] for s in present}
        for m in ms:
            vals = {s: df[(df.model == m) & (df.strategy == s)].macro_f1.mean() for s in present}
            vals = {s: v for s, v in vals.items() if not np.isnan(v)}
            for i, s in enumerate(sorted(vals, key=lambda s: -vals[s])):
                ranks[s].append(i + 1)
        for s in present:
            if ranks[s]:
                V[f"RANK_{SKEY[s]}"] = f"{np.mean(ranks[s]):.1f}"

    # ---------------- split sizes
    sizes = [{k: len(v) for k, v in json.load(open(js)).items()} for js in glob.glob(os.path.join(a.results, "splits", "*.json"))]
    if sizes:
        sz = pd.DataFrame(sizes).mean()
        for k in sz.index:
            V[f"N_{k.upper()}"] = f"{sz[k]:.0f}"

    # ---------------- active learning mechanism and self-training behaviour
    al = {s: [r["info"]["extras"] for kk, r in runs.items() if kk[2] == s] for s in strategies if s.startswith("al_") or s == "random"}
    def minority_share(infos):
        tot = mino = 0
        for e in infos:
            for pt in e["curve"][1:]:
                q = pt["queried_per_class"]
                tot += sum(q.values())
                mino += sum(q[c] for c in MINORITY)
        return 100 * mino / max(tot, 1)
    for s, infos in al.items():
        if infos:
            V[f"MINSHARE_{SKEY[s]}"] = f"{minority_share(infos):.1f}"
            V["AL_ROUNDS"], V["AL_QUERY"] = str(infos[0]["rounds"]), f"{np.mean([e['query_size'] for e in infos]):.0f}"
            V["AL_INIT_N"] = f"{np.mean([e['initial_labelled'] for e in infos]):.0f}"
            V["AL_FINAL_N"] = f"{np.mean([e['n_labelled'] for e in infos]):.0f}"
    st = [r["info"]["extras"] for kk, r in runs.items() if kk[2] == "selftrain"]
    if st:
        for it in range(1, 6):
            pts = [e["rounds"][it] for e in st if len(e["rounds"]) > it and e["rounds"][it].get("n_pseudo", 0) > 0]
            V[f"ST_NPSEUDO_IT{it}"] = f"{np.mean([p['n_pseudo'] for p in pts]):.0f}" if pts else "–"
            V[f"ST_PACC_IT{it}"] = f"{100 * np.mean([p['pseudo_acc'] for p in pts]):.1f}" if pts else "–"
            V[f"ST_PF1MIN_IT{it}"] = f"{np.mean([np.mean([p['pseudo_f1_per_class'][c] for c in MINORITY]) for p in pts]):.3f}" if pts else "–"
        sel = [e["selected_iteration"] for e in st]
        V["ST_SEL_ITER_MEDIAN"], V["ST_SEL_ITER_RANGE"] = f"{np.median(sel):.0f}", f"{min(sel)}–{max(sel)}"

    # ---------------- synthetic image quality
    for gen in ("dcgan", "diffusion"):
        reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.results, f"{gen}_report_fold*.json")))]
        if not reps:
            continue
        g = SKEY[gen]
        for c in MINORITY:
            ent = [r[c] for r in reps if c in r]
            V[f"Q_{g}_{c.upper()}_NREAL"] = f"{np.mean([e['n_real'] for e in ent]):.0f}" if ent else "–"
            V[f"Q_{g}_{c.upper()}_NGEN"] = f"{np.mean([e['n_generated'] for e in ent]):.0f}" if ent else "–"
            for q, f in (("fid", "{:.1f}"), ("kid", "{:.1f}"), ("nn_ssim", "{:.3f}")):
                vals = [e[q] * (1000 if q == "kid" else 1) for e in ent if q in e]
                V[f"Q_{g}_{c.upper()}_{q.upper()}"] = f.format(np.mean(vals)) if vals else "–"
        for q, f in (("fid", "{:.1f}"), ("kid", "{:.1f}"), ("nn_ssim", "{:.3f}")):
            vals = [e[q] * (1000 if q == "kid" else 1) for r in reps for e in r.values() if q in e]
            V[f"Q_{g}_MEAN_{q.upper()}"] = f.format(np.mean(vals)) if vals else "–"
        V[f"Q_{g}_MINUTES"] = f"{np.mean([sum(e.get('train_seconds', 0) + e.get('sample_seconds', 0) for e in r.values()) for r in reps]) / 60:.1f}"

    cfgp = os.path.join(a.results, "config.json")
    if os.path.exists(cfgp):
        gpu = json.load(open(cfgp)).get("gpu")
        V["GPU"] = f"one NVIDIA {gpu.replace('NVIDIA ', '')} GPU" if gpu else "a CPU"

    if a.sensitivity:
        sensitivity(a.sensitivity, df, V, out)
    sentences(df, edf, cdf, models, budget_models, V)
    figures(runs, df, edf, pcf, cdf, models, budget_models, strategies, out, a.cache_dir, a.results, V)
    json.dump(V, open(os.path.join(out, "values.json"), "w"), indent=1, ensure_ascii=False)
    try:
        from supplement import write_supplement
        write_supplement(out, df, edf, pcf, cdf, a.results)
    except Exception as e:  # supplement is a convenience; never block the main outputs
        print("supplement not written:", e)
    print(f"{len(V)} values written to {os.path.join(out, 'values.json')}")


# ------------------------------------------------------------------ sensitivity analysis
def sensitivity(root, df, V, out):
    """Each sub-folder is named <param>=<value> and contains a run_experiments output for one model."""
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*=*"))):
        param, val = os.path.basename(d).split("=", 1)
        for js in glob.glob(os.path.join(d, "fold*", "*", "*.json")):
            info = json.load(open(js))
            z = np.load(js[:-5] + ".npz", allow_pickle=True)
            rows.append(dict(param=param, value=val, model=info["model"], strategy=info["strategy"], fold=info["fold"],
                             **compute_metrics(z["y"], z["probs"])))
    if not rows:
        return
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out, "sensitivity.csv"), index=False)
    for (param, val, strat), g in s.groupby(["param", "value", "strategy"]):
        V[f"SENS_{param.upper()}_{val}_{SKEY[strat]}_MF1"] = f"{g.macro_f1.mean():.3f} ± {g.macro_f1.std(ddof=1) if len(g) > 1 else 0:.3f}"


# ------------------------------------------------------------------ generated sentences
def sentences(df, edf, cdf, models, budget_models, V):
    base = {m: df[(df.model == m) & (df.strategy == "baseline")].macro_f1.mean() for m in models}
    base = {m: v for m, v in base.items() if not np.isnan(v)}
    if base:
        best = max(base, key=base.get)
        V["TXT_BEST_MODEL"] = DISPLAY_NAMES[best]
        V["BEST_BASE_MF1"] = V.get(f"BASE_{MKEY[best]}_MF1", "")
        V["BEST_BASE_BACC"] = V.get(f"BASE_{MKEY[best]}_BACC", "")
        V["TXT_MODEL_ORDER"] = ", ".join(DISPLAY_NAMES[m] for m in sorted(base, key=lambda m: -base[m]))
    if cdf is None or cdf.empty:
        return
    def effect(s, ref):
        sub = cdf[(cdf.strategy == s) & (cdf.reference == ref)]
        if sub.empty:
            return None
        return dict(mean=sub.d_macro_f1.mean(), up=[DISPLAY_NAMES[r.model] for r in sub.itertuples() if sig(r) and r.d_macro_f1 > 0],
                    down=[DISPLAY_NAMES[r.model] for r in sub.itertuples() if sig(r) and r.d_macro_f1 < 0], n=len(sub),
                    lo=sub.d_macro_f1.min(), hi=sub.d_macro_f1.max())
    def sentence(name, refname, e):
        rng = f"{e['lo']:+.3f} to {e['hi']:+.3f}" if e["n"] > 1 else f"{e['lo']:+.3f}"
        if e["up"] and not e["down"]:
            return f"{name} improved macro-F1 significantly over {refname} for {len(e['up'])} of {e['n']} architectures ({', '.join(e['up'])}; change {rng})."
        if e["down"] and not e["up"]:
            return f"{name} reduced macro-F1 significantly relative to {refname} for {len(e['down'])} of {e['n']} architectures ({', '.join(e['down'])}; change {rng})."
        if e["up"] and e["down"]:
            return f"The effect of {name.lower()} depended on the architecture: significant gains for {', '.join(e['up'])} and losses for {', '.join(e['down'])} (change {rng})."
        return f"{name} did not differ significantly from {refname} for any architecture after Holm correction (change {rng})."
    names = {"cwl": "class-weighted loss", "focal": "focal loss", "oversample": "oversampling", "dcgan": "DCGAN augmentation",
             "diffusion": "diffusion augmentation", "selftrain": "self-training", "st_cwl": "self-training with CWL",
             "fixmatch": "FixMatch", "al_lc": "least-confidence AL", "al_entropy": "entropy AL", "al_bald": "BALD AL",
             "al_coreset": "core-set AL", "al_cwl": "least-confidence AL with CWL"}
    refs = {"baseline": "standard training", "sup50": "supervised training on the same labels",
            "random": "random sampling at the same budget", "al_lc": "least-confidence AL without CWL",
            "selftrain": "self-training without CWL"}
    effs = {}
    for s, ref in FULL_FAMILY + BUDGET_FAMILY:
        e = effect(s, ref)
        if e:
            effs[(s, ref)] = e
            V[f"TXT_{SKEY[s]}vs{SKEY[ref]}"] = sentence(names[s][0].upper() + names[s][1:], refs[ref], e)
            V[f"TXT_{SKEY[s]}vs{SKEY[ref]}_LC"] = V[f"TXT_{SKEY[s]}vs{SKEY[ref]}"][0].lower() + V[f"TXT_{SKEY[s]}vs{SKEY[ref]}"][1:]
    def family(items, refname, noun):
        up, down, mixed, ns = [], [], [], []
        for n, e in items:
            (mixed if e["up"] and e["down"] else up if e["up"] else down if e["down"] else ns).append((n, e))
        parts = []
        for n, e in up:
            parts.append(f"{n} improved macro-F1 significantly over {refname} for {len(e['up'])} of {e['n']} architectures "
                         f"({', '.join(e['up'])}; mean change {e['mean']:+.3f})")
        for n, e in down:
            parts.append(f"{n} reduced macro-F1 significantly for {', '.join(e['down'])} (mean change {e['mean']:+.3f})")
        for n, e in mixed:
            parts.append(f"{n} had architecture-dependent effects, with significant gains for {', '.join(e['up'])} and "
                         f"losses for {', '.join(e['down'])}")
        if ns:
            lst = ", ".join(f"{n} {e['mean']:+.3f}" for n, e in ns)
            if len(ns) == len(items):
                W = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
                parts.append(f"none of the {W.get(len(items), len(items))} {noun} differed significantly from {refname} for any architecture "
                             f"after Holm correction (mean changes: {lst})")
            else:
                names_ = [n for n, _ in ns]
                who = names_[0] if len(names_) == 1 else ", ".join(names_[:-1]) + " and " + names_[-1]
                parts.append(f"the differences for {who} were not significant for any architecture after Holm correction "
                             f"(mean changes: {', '.join(f'{e[chr(109)+chr(101)+chr(97)+chr(110)]:+.3f}' for _, e in ns)})")
        txt = "; ".join(parts)
        return txt[0].upper() + txt[1:] + "." if txt else ""
    fams = {"FULL": ([s for s, _ in FULL_FAMILY], "baseline", "standard training", "full-label strategies"),
            "AL": (["al_lc", "al_entropy", "al_bald", "al_coreset"], "random", "random sampling at the same budget",
                   "active-learning criteria"),
            "SEMI": (["selftrain", "st_cwl", "fixmatch"], "sup50", "supervised training on the same labels",
                     "semi-supervised methods")}
    for key, (ss, ref, refname, noun) in fams.items():
        items = [(names[s], effs[(s, ref)]) for s in ss if (s, ref) in effs]
        if items:
            V[f"TXT_FAM_{key}"] = family(items, refname, noun)
    full = {s: effs[(s, r)] for s, r in FULL_FAMILY if (s, r) in effs}
    budget = {s: effs[(s, r)] for s, r in BUDGET_FAMILY[:-2] if (s, r) in effs}
    def sig_txt(e):
        a, b = len(e["up"]), len(e["down"])
        if a and b: return f"significant gain for {a}/{e['n']}, loss for {b}/{e['n']}"
        if a: return f"significant gain for {a}/{e['n']}"
        if b: return f"significant loss for {b}/{e['n']}"
        return "not significant for any architecture"
    for label, group, regime, refname in (("FULL", full, "With full labels", "standard training"),
                                          ("BUDGET", budget, "With 50% of the labels", "the equal-budget references")):
        if not group:
            continue
        order = sorted(group, key=lambda s: -group[s]["mean"])
        top, worst = order[0], order[-1]
        V[f"TXT_ORDER_{label}"] = "; ".join(f"{names[s]} ({group[s]['mean']:+.3f}; {sig_txt(group[s])})" for s in order)
        V[f"TXT_TOP_{label}"] = names[top]
        V[f"TXT_TOP_{label}_D"] = f"{group[top]['mean']:+.3f}"
        V[f"TXT_TOP_{label}_SIG"] = f"{len(group[top]['up'])} of {group[top]['n']}"
        n_up = sum(len(e["up"]) for e in group.values()); n_dn = sum(len(e["down"]) for e in group.values())
        n_tot = sum(e["n"] for e in group.values())
        losers = [names[s] for s in order if group[s]["down"]]
        if group[top]["mean"] > 0:
            head = (f"{regime}, the largest mean change in macro-F1 relative to {refname} came from {names[top]} "
                    f"({group[top]['mean']:+.3f}; {sig_txt(group[top])})")
        else:
            head = (f"{regime}, no strategy improved mean macro-F1 over {refname}; mean changes ranged from "
                    f"{group[top]['mean']:+.3f} ({names[top]}) to {group[worst]['mean']:+.3f} ({names[worst]})")
        tail = f"; {n_up} of {n_tot} strategy–architecture comparisons showed a significant gain and {n_dn} a significant loss"
        if losers:
            tail += f" ({', '.join(losers)})"
        V[f"TXT_ABS_{label}"] = head + tail + "."
        V[f"TXT_ABS_{label}_LC"] = V[f"TXT_ABS_{label}"][0].lower() + V[f"TXT_ABS_{label}"][1:]
        rng = f"{group[top]['mean']:+.3f} ({names[top]}) to {group[worst]['mean']:+.3f} ({names[worst]})"
        gains = f"significant gains in {n_up} and losses in {n_dn} of {n_tot} comparisons"
        V[f"TXT_ABS_{label}_SHORT"] = (f"{regime}, mean macro-F1 changes relative to {refname} ranged from {rng}, with {gains}.")
    if full:
        n_any = sum(1 for e in full.values() if e["up"])
        V["TXT_FULL_SUMMARY"] = (f"{n_any} of {len(full)} full-label strategies improved macro-F1 significantly for at least one architecture"
                                 if n_any else "no full-label strategy improved macro-F1 significantly for any architecture")
    if not edf.empty and "isic2019" in set(edf.dataset) and full:
        e = edf[edf.dataset == "isic2019"]
        base = e[e.strategy == "baseline"].groupby("model").macro_f1.mean()
        parts = []
        for s in sorted(full, key=lambda s: -full[s]["mean"]):
            d = (e[e.strategy == s].groupby("model").macro_f1.mean() - base).dropna()
            if len(d):
                V[f"EXT_D_{SKEY[s]}"] = f"{d.mean():+.3f}"
                parts.append(f"{names[s]} {d.mean():+.3f} (internal {full[s]['mean']:+.3f})")
        top = max(full, key=lambda s: full[s]["mean"])
        if f"EXT_D_{SKEY[top]}" in V:
            V["TXT_TOP_FULL_EXT_D"] = V[f"EXT_D_{SKEY[top]}"]
        if parts:
            V["TXT_EXT_TRANSFER"] = ("Relative to the baseline, the mean change in external ISIC 2019 macro-F1 was: "
                                     + "; ".join(parts) + ".")


# ------------------------------------------------------------------ figures (300 dpi)
def save(fig, out, name):
    fig.savefig(os.path.join(out, f"fig_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(runs, df, edf, pcf, cdf, models, budget_models, strategies, out, cache_dir, res_dir, V):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    # ROC (baseline, macro OvR, pooled) + melanoma ROC
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    grid = np.linspace(0, 1, 501)
    for m in models:
        p = pooled(runs, m, "baseline")
        if p is None:
            continue
        tprs = []
        for c in np.unique(p["y"]):
            fpr, tpr, _ = roc_curve(p["y"] == c, p["probs"][:, c])
            tprs.append(np.interp(grid, fpr, tpr))
        mt = np.mean(tprs, 0); mt[0] = 0
        ax[0].plot(grid, mt, lw=1.2, label=f"{DISPLAY_NAMES[m]} ({sk_auc(grid, mt):.3f})")
        fpr, tpr, _ = roc_curve(p["y"] == CLASSES.index("mel"), p["probs"][:, CLASSES.index("mel")])
        ax[1].plot(fpr, tpr, lw=1.2, label=f"{DISPLAY_NAMES[m]} ({sk_auc(fpr, tpr):.3f})")
    for x, t in zip(ax, ("Macro-average ROC, 7 classes", "Melanoma vs rest")):
        x.plot([0, 1], [0, 1], "k--", lw=0.6); x.set_title(t); x.set_xlabel("False positive rate"); x.set_ylabel("True positive rate")
        x.legend(fontsize=6, loc="lower right", title="AUC (pooled)", title_fontsize=6)
    ax[1].axvline(0.2, color="grey", ls=":", lw=0.8)
    save(fig, out, "roc")

    # effect heat map: full-label strategies x architectures
    if cdf is not None and not cdf.empty:
        for fam, pairs, ms, name in (("full", FULL_FAMILY, models, "effects_full"), ("budget", BUDGET_FAMILY, budget_models, "effects_budget")):
            sub = cdf[cdf.family == fam]
            if sub.empty:
                continue
            rows_ = [(s, r) for s, r in pairs if ((sub.strategy == s) & (sub.reference == r)).any()]
            M = np.full((len(rows_), len(ms)), np.nan)
            S = np.zeros_like(M, dtype=bool)
            for i, (s, r) in enumerate(rows_):
                for j, m in enumerate(ms):
                    x = sub[(sub.strategy == s) & (sub.reference == r) & (sub.model == m)]
                    if len(x):
                        M[i, j], S[i, j] = x.d_macro_f1.iloc[0], sig(x.iloc[0])
            lim = max(0.01, np.nanmax(np.abs(M)))
            fig, ax = plt.subplots(figsize=(1.1 * len(ms) + 2.6, 0.42 * len(rows_) + 1.2))
            im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:+.3f}{'*' if S[i, j] else ''}", ha="center", va="center", fontsize=6.5)
            ax.set_xticks(range(len(ms)), [DISPLAY_NAMES[m] for m in ms], rotation=30, ha="right")
            ax.set_yticks(range(len(rows_)), [f"{SNAME[s]} vs {SNAME[r].lower()}" for s, r in rows_])
            fig.colorbar(im, fraction=0.03, label="Δ macro-F1")
            ax.set_title("Change in macro-F1 (* Holm-adjusted p < 0.05)")
            save(fig, out, name)

    # active learning curves
    al_like = [s for s in ["random", "al_lc", "al_entropy", "al_bald", "al_coreset", "al_cwl"] if s in strategies]
    if al_like:
        fig, ax = plt.subplots(1, len(budget_models), figsize=(3.0 * len(budget_models), 2.8), squeeze=False)
        for j, m in enumerate(budget_models):
            for s in al_like:
                curves = [r["info"]["extras"]["curve"] for (k, mm, ss), r in runs.items() if mm == m and ss == s]
                if not curves:
                    continue
                L = min(len(c) for c in curves)
                x = 100 * np.mean([[p["frac_labelled"] for p in c[:L]] for c in curves], 0)
                yv = np.array([[p["test_macro_f1"] for p in c[:L]] for c in curves])
                ax[0, j].plot(x, yv.mean(0), "-o" if s != "random" else "--o", ms=2.5, lw=1, label=SNAME[s])
            b = df[(df.model == m) & (df.strategy == "baseline")].macro_f1.mean()
            if not np.isnan(b):
                ax[0, j].axhline(b, color="grey", ls=":", lw=0.8, label="Baseline, 100% labels")
            ax[0, j].set_title(DISPLAY_NAMES[m]); ax[0, j].set_xlabel("% of training fold labelled"); ax[0, j].set_ylabel("Test macro-F1")
        ax[0, -1].legend(fontsize=6)
        save(fig, out, "active_learning")

    # self-training iterations
    st = [(kk[1], r["info"]["extras"]["rounds"]) for kk, r in runs.items() if kk[2] == "selftrain"]
    if st:
        L = max(len(x) for _, x in st)
        its = list(range(1, L))
        g = lambda key: [np.mean([x[i][key] for _, x in st if len(x) > i and x[i].get(key) is not None] or [np.nan]) for i in its]
        fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
        ax[0].bar(its, g("n_pseudo"), color="C0"); ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("Pseudo-labelled images")
        t = ax[0].twinx(); t.plot(its, 100 * np.array(g("pseudo_acc")), "o-", color="C3"); t.set_ylabel("Pseudo-label accuracy (%)")
        f1 = [np.mean([x[i]["test_macro_f1"] for _, x in st if len(x) > i and "test_macro_f1" in x[i]] or [np.nan]) for i in range(L)]
        ax[1].plot(range(L), f1, "o-"); ax[1].set_xlabel("Iteration (0 = supervised, 50% labels)"); ax[1].set_ylabel("Test macro-F1")
        save(fig, out, "self_training")

    # per-class F1 heat map
    if not pcf.empty:
        tab = pcf.groupby("strategy")[CLASSES].mean().reindex([s for s in STRATEGIES if s in set(pcf.strategy)])
        fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(tab) + 1.0))
        im = ax.imshow(tab.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                ax.text(j, i, f"{tab.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if tab.values[i, j] < 0.6 else "black")
        ax.set_xticks(range(C), CLASSES); ax.set_yticks(range(len(tab)), [SNAME[s] for s in tab.index])
        fig.colorbar(im, fraction=0.03, label="F1"); ax.set_title("Per-class F1 (pooled out-of-fold, mean over architectures)")
        save(fig, out, "per_class_f1")

    # internal vs external
    if not edf.empty and "isic2019" in set(edf.dataset):
        e = edf[edf.dataset == "isic2019"].groupby("strategy").macro_f1.mean()
        i_ = df.groupby("strategy").macro_f1.mean()
        ss = [s for s in STRATEGIES if s in e.index and s in i_.index]
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        x = np.arange(len(ss))
        ax.bar(x - 0.2, [i_[s] for s in ss], 0.4, label="HAM10000 (internal, cross-validated)")
        ax.bar(x + 0.2, [e[s] for s in ss], 0.4, label="ISIC 2019 non-HAM (external)")
        ax.set_xticks(x, [SNAME[s] for s in ss], rotation=40, ha="right"); ax.set_ylabel("Macro-F1 (mean over architectures)")
        ax.legend(fontsize=7)
        save(fig, out, "external")

    # synthetic examples: DCGAN vs diffusion (fold 1)
    ex = {}
    for gen in ("dcgan", "diffusion"):
        for c in MINORITY:
            p = os.path.join(cache_dir, f"{gen}_fold0", f"{c}_examples.npy")
            if os.path.exists(p):
                ex[(gen, c)] = np.load(p)
    if ex:
        gens = [g for g in ("dcgan", "diffusion") if any(k[0] == g for k in ex)]
        per = 6 // len(gens)                                   # 6 columns: 3 per generator, or 6 for one generator
        cls = [c for c in MINORITY if any((g, c) in ex for g in gens)]
        fig, ax = plt.subplots(len(cls), 6, figsize=(6.0, 1.05 * len(cls)), squeeze=False)
        for i, c in enumerate(cls):
            for j in range(6):
                gen, jj = gens[j // per], j % per
                a_ = ax[i, j]; a_.axis("off")
                if (gen, c) in ex and len(ex[(gen, c)]) > jj:
                    a_.imshow(ex[(gen, c)][jj])
                if i == 0 and len(gens) > 1 and jj == 1:
                    a_.set_title("DCGAN" if gen == "dcgan" else "Diffusion", fontsize=8)
            ax[i, 0].text(-0.15, 0.5, c, transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=8)
        save(fig, out, "synthetic_examples")

    # supplementary: learning curves, confusion, reliability
    k0 = min((k for k, _, s in runs if s == "baseline"), default=None)
    if k0 is not None:
        fig, ax = plt.subplots(2, len(models), figsize=(2.3 * len(models), 4.2), squeeze=False)
        for j, m in enumerate(models):
            r = runs.get((k0, m, "baseline"))
            if r is None:
                continue
            h = r["info"]["history"]
            ax[0, j].plot(h["epoch"], h["train_loss"], label="training"); ax[0, j].plot(h["epoch"], h["val_loss"], label="validation")
            ax[1, j].plot(h["epoch"], h["train_acc"], label="training acc."); ax[1, j].plot(h["epoch"], h["val_acc"], label="validation acc.")
            ax[1, j].plot(h["epoch"], h["val_macro_f1"], "--", label="validation macro-F1")
            ax[1, j].axvline(h["best_epoch"], color="grey", ls=":", lw=0.8); ax[0, j].set_title(DISPLAY_NAMES[m])
            ax[1, j].set_xlabel("Epoch")
        ax[0, 0].set_ylabel("Loss"); ax[1, 0].set_ylabel("Accuracy / macro-F1"); ax[0, 0].legend(fontsize=6); ax[1, 0].legend(fontsize=6)
        save(fig, out, "learning_curves")
    if "TXT_BEST_MODEL" in V:
        best = next(m for m in models if DISPLAY_NAMES[m] == V["TXT_BEST_MODEL"])
        p = pooled(runs, best, "baseline")
        cm = confusion(p["y"], p["probs"]).astype(float)
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(C):
            for j in range(C):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=5.5,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(C), CLASSES); ax.set_yticks(range(C), CLASSES); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{DISPLAY_NAMES[best]}, baseline (pooled folds)"); fig.colorbar(im, fraction=0.046)
        save(fig, out, "confusion")
        V["CM_MODEL"] = DISPLAY_NAMES[best]
        conf, pred = p["probs"].max(1), p["probs"].argmax(1)
        edges = np.linspace(0, 1, 11)
        mids, accs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (conf > lo) & (conf <= hi)
            if mm.any():
                mids.append(conf[mm].mean()); accs.append((pred[mm] == p["y"][mm]).mean())
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        ax.plot([0, 1], [0, 1], "k--", lw=0.6); ax.plot(mids, accs, "o-")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title(f"Reliability, {DISPLAY_NAMES[best]} baseline")
        save(fig, out, "reliability")


if __name__ == "__main__":
    main()
