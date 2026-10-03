"""Evaluation metrics: overall, per-class, melanoma operating points and calibration."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score, roc_curve, confusion_matrix)

from config import CLASSES

MEL = CLASSES.index("mel")


def melanoma_metrics(y_true, p_mel):
    """Melanoma vs all: AUC, sensitivity at 80 % specificity, specificity at 90 % sensitivity."""
    pos = np.asarray(y_true) == MEL
    if pos.all() or not pos.any():
        return dict(mel_auc=np.nan, mel_sens_at_spec80=np.nan, mel_spec_at_sens90=np.nan)
    fpr, tpr, _ = roc_curve(pos, p_mel)
    sens80 = tpr[fpr <= 0.20].max() if (fpr <= 0.20).any() else 0.0
    spec90 = 1 - fpr[tpr >= 0.90].min() if (tpr >= 0.90).any() else 0.0
    return dict(mel_auc=roc_auc_score(pos, p_mel), mel_sens_at_spec80=sens80, mel_spec_at_sens90=spec90)


def calibration(y_true, probs, bins=15):
    """Expected calibration error (top label, equal-width bins) and multi-class Brier score."""
    y_true = np.asarray(y_true)
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    onehot = np.eye(probs.shape[1])[y_true]
    return dict(ece=ece, brier=float(((probs - onehot) ** 2).sum(1).mean()))


def compute_metrics(y_true, probs):
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    labels = list(range(len(CLASSES)))
    out = dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
    )
    present = np.unique(y_true)
    try:
        p = probs[:, present] / np.clip(probs[:, present].sum(1, keepdims=True), 1e-12, None)
        out["auc"] = roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=present) \
            if len(present) > 2 else roc_auc_score(y_true == present[1], probs[:, present[1]])
    except ValueError:
        out["auc"] = float("nan")
    for c, v in zip(CLASSES, f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)):
        out[f"f1_{c}"] = v
    out.update(melanoma_metrics(y_true, probs[:, MEL]))
    out.update(calibration(y_true, probs))
    return out


def binary_melanoma_metrics(y_true, probs):
    """For PH2 (melanoma vs nevus only): melanoma AUC and operating points from p(mel)."""
    out = melanoma_metrics(y_true, probs[:, MEL])
    pred_mel = probs[:, MEL] > probs[:, CLASSES.index("nv")]
    pos = np.asarray(y_true) == MEL
    out["mel_vs_nv_accuracy"] = float((pred_mel == pos).mean())
    return out


def confusion(y_true, probs):
    return confusion_matrix(y_true, probs.argmax(1), labels=list(range(len(CLASSES))))
