"""Scan-level v3+LCC separation of batch2 positives vs tumor-free negatives.

Measurement only — no training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from config import PROJECT_ROOT

SPLIT_DIR = PROJECT_ROOT / "splits" / "batch2_pos_neg"
OUT_DIR = PROJECT_ROOT / "results" / "pos_neg_separation"
BATCH2_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients_batch2"
NEG_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients"

SCORE_COLS = (
    "WT_lcc_ml",
    "TC_lcc_ml",
    "ET_lcc_ml",
    "WT_raw_n_cc",
    "WT_max_prob_on_lcc",
)


@dataclass
class AucResult:
    score: str
    n_pos: int
    n_neg: int
    auc: float
    ci_lo: float
    ci_hi: float
    best_threshold: float
    sensitivity: float
    specificity: float
    subset: str


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(np.unique(y_true)) < 2 or len(y_true) < 3:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(float(roc_auc_score(yt, ys)))
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def best_threshold_sens_spec(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float, float]:
    """Youden's J: maximize sensitivity + specificity - 1."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    k = int(np.argmax(j))
    # sklearn thr length = len(tpr); last thr may be nan-ish handling
    t = float(thr[k]) if k < len(thr) else float("nan")
    sens = float(tpr[k])
    spec = float(1.0 - fpr[k])
    return t, sens, spec


def evaluate_score(
    df: pd.DataFrame,
    score: str,
    *,
    subset: str,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
) -> AucResult:
    y = np.concatenate(
        [
            np.ones(int(pos_mask.sum()), dtype=int),
            np.zeros(int(neg_mask.sum()), dtype=int),
        ]
    )
    s = np.concatenate(
        [
            df.loc[pos_mask, score].to_numpy(dtype=float),
            df.loc[neg_mask, score].to_numpy(dtype=float),
        ]
    )
    auc, lo, hi = bootstrap_auc_ci(y, s)
    thr, sens, spec = best_threshold_sens_spec(y, s)
    return AucResult(
        score=score,
        n_pos=int(pos_mask.sum()),
        n_neg=int(neg_mask.sum()),
        auc=auc,
        ci_lo=lo,
        ci_hi=hi,
        best_threshold=thr,
        sensitivity=sens,
        specificity=spec,
        subset=subset,
    )


def plot_roc_and_distributions(
    df: pd.DataFrame,
    *,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    scores: Sequence[str],
    title_prefix: str,
    out_path: Path,
) -> None:
    n = len(scores)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, score in enumerate(scores):
        y = np.concatenate(
            [np.ones(int(pos_mask.sum())), np.zeros(int(neg_mask.sum()))]
        )
        s = np.concatenate(
            [
                df.loc[pos_mask, score].to_numpy(dtype=float),
                df.loc[neg_mask, score].to_numpy(dtype=float),
            ]
        )
        m = np.isfinite(s)
        y, s = y[m], s[m]
        ax = axes[0, i]
        if len(np.unique(y)) >= 2:
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.legend(fontsize=8)
        ax.set_title(score, fontsize=9)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax2 = axes[1, i]
        pos_s = df.loc[pos_mask, score].to_numpy(dtype=float)
        neg_s = df.loc[neg_mask, score].to_numpy(dtype=float)
        ax2.hist(neg_s[np.isfinite(neg_s)], bins=20, alpha=0.55, label="neg", color="C0")
        ax2.hist(pos_s[np.isfinite(pos_s)], bins=20, alpha=0.55, label="pos", color="C3")
        ax2.legend(fontsize=8)
        ax2.set_xlabel(score)
        ax2.set_ylabel("count")
    fig.suptitle(title_prefix, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def shortcut_auc_patient_cv(
    df: pd.DataFrame,
    *,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    feature_cols: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Logistic regression AUC on non-tumor features; GroupKFold by patient_id."""
    use = df.loc[pos_mask | neg_mask].copy()
    y = (use["cohort"] == "positive").astype(int).to_numpy()
    # Encode categoricals
    X_parts = []
    names = []
    for col in feature_cols:
        if col not in use.columns:
            continue
        if use[col].dtype == object or str(use[col].dtype) == "bool":
            dummies = pd.get_dummies(use[col].astype(str), prefix=col)
            X_parts.append(dummies.to_numpy(dtype=float))
            names.extend(list(dummies.columns))
        else:
            vals = pd.to_numeric(use[col], errors="coerce").to_numpy(dtype=float)
            med = float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else 0.0
            if not np.isfinite(med):
                med = 0.0
            vals = np.nan_to_num(vals, nan=med, posinf=med, neginf=med)
            X_parts.append(vals.reshape(-1, 1))
            names.append(col)
    if not X_parts:
        return {"auc_mean": float("nan"), "auc_folds": [], "warning": False, "n": 0}
    X = np.concatenate(X_parts, axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    groups = use["patient_id"].astype(str).fillna("unknown").to_numpy()
    n_groups = len(np.unique(groups))
    k = min(n_folds, n_groups)
    if k < 2 or len(np.unique(y)) < 2:
        return {"auc_mean": float("nan"), "auc_folds": [], "warning": False, "n": len(y)}
    gkf = GroupKFold(n_splits=k)
    fold_aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
            continue
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(Xtr, y[train_idx])
        proba = clf.predict_proba(Xte)[:, 1]
        fold_aucs.append(float(roc_auc_score(y[test_idx], proba)))
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    warning = bool(mean_auc == mean_auc and mean_auc >= 0.75)
    return {
        "auc_mean": mean_auc,
        "auc_folds": fold_aucs,
        "n_folds": k,
        "n": int(len(y)),
        "features": names,
        "warning": warning,
        "warning_threshold": 0.75,
    }


def verdict_from_wt_auc(auc: float) -> str:
    if auc != auc:
        return "no"
    if auc >= 0.85:
        return "yes"
    if auc >= 0.70:
        return "partly"
    return "no"


def write_summary(
    *,
    results: list[AucResult],
    shortcut: dict[str, Any],
    confirmed_ran: bool,
    n_confirmed: int,
    out_path: Path,
) -> None:
    lines = [
        "# Positive vs negative separation (v3+LCC)",
        "",
        "Measurement only — no training. Checkpoint `brats_scale_full` read-only.",
        "",
        "## Limits",
        "",
        "- **19** batch2 positives only.",
        "- Negatives: tumor-free status from `manifest.csv` "
        "(radiologist_report / radiologist_verbal / claimed_unverified).",
        f"- Radiologist-confirmed negatives available for analysis: **{n_confirmed}**"
        + (" (ran confirmed subset)." if confirmed_ran else " — **confirmed subset skipped; (b) all claimed only**."),
        "- Scanner mix differs (batch2 mostly Philips Ingenia; negatives mostly Achieva).",
        "",
        "## ROC AUC tables",
        "",
        "| subset | score | n_pos | n_neg | AUC | 95% CI | best thr | sens | spec |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in results:
        ci = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if r.ci_lo == r.ci_lo else "—"
        lines.append(
            f"| {r.subset} | {r.score} | {r.n_pos} | {r.n_neg} | {r.auc:.3f} | {ci} | "
            f"{r.best_threshold:.4g} | {r.sensitivity:.3f} | {r.specificity:.3f} |"
        )

    # Verdict from primary WT on all claimed
    primary = next(
        (
            r
            for r in results
            if r.subset == "all_claimed_negatives" and r.score == "WT_lcc_ml"
        ),
        None,
    )
    verdict = verdict_from_wt_auc(primary.auc) if primary else "no"
    lines += [
        "",
        "## Shortcut check (non-tumor features)",
        "",
        "Features: in-plane spacing, brain mask fraction, scanner model, T1c native/stand-in, "
        "mean brain intensity. Logistic regression, patient-grouped 5-fold CV.",
        "",
        f"- Mean CV AUC: **{shortcut.get('auc_mean', float('nan')):.3f}** "
        f"(folds={shortcut.get('auc_folds')})",
        "",
    ]
    if shortcut.get("warning"):
        lines += [
            "> **WARNING:** Non-tumor features separate positives from negatives well "
            f"(CV AUC ≥ {shortcut.get('warning_threshold', 0.75)}). "
            "Any classifier trained on these groups may be reading the scanner / "
            "acquisition protocol — not the tumor.",
            "",
        ]
    else:
        lines += [
            "Non-tumor shortcut AUC is below the warning threshold (0.75), "
            "but residual acquisition differences may still leak.",
            "",
        ]

    lines += [
        "## Verdict",
        "",
        f"**Does v3 WT volume separate the groups: {verdict}**",
        "",
        (
            f"Primary WT AUC vs all claimed negatives = "
            f"{primary.auc:.3f} [{primary.ci_lo:.3f}, {primary.ci_hi:.3f}]."
            if primary
            else "Primary WT AUC unavailable."
        ),
        "",
        "Plots: `roc_distributions_*.png` in this folder.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
