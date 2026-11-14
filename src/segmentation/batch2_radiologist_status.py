"""Apply radiologist tumor-free reads to batch2 pos/neg manifests (metadata only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

SPLIT_DIR = PROJECT_ROOT / "splits" / "batch2_pos_neg"
REPORT_DIR = PROJECT_ROOT / "outputs" / "batch2_pos_neg"
RAD_CSV = PROJECT_ROOT / "data" / "processed" / "real_patients" / "radiologist_reads.csv"

STATUS_REPORT = "radiologist_report"
STATUS_VERBAL = "radiologist_verbal"
STATUS_UNVERIFIED = "claimed_unverified"
ALLOWED = {STATUS_REPORT, STATUS_VERBAL, STATUS_UNVERIFIED}

STATUS_PHRASE = {
    STATUS_REPORT: "tumor-free per radiologist report",
    STATUS_VERBAL: "tumor-free per radiologist verbal confirmation",
    STATUS_UNVERIFIED: "tumor-free claimed by user, unverified in data",
}


def load_radiologist_reads(path: Path | None = None) -> pd.DataFrame:
    p = path or RAD_CSV
    if not p.is_file():
        return pd.DataFrame(
            columns=["study_id", "radiologist_read", "other_findings", "source", "date"]
        )
    df = pd.read_csv(p, dtype=str).fillna("")
    for col in ("study_id", "radiologist_read", "other_findings", "source", "date"):
        if col not in df.columns:
            df[col] = ""
    return df


def status_from_row(row: pd.Series | dict[str, Any]) -> str:
    """Map a radiologist_reads row to tumor_free_status. Empty → claimed_unverified."""
    read = str(row.get("radiologist_read", "") or "").strip()
    source = str(row.get("source", "") or "").strip().lower()
    if not read and not source:
        return STATUS_UNVERIFIED
    if read in ALLOWED:
        return read
    if "verbal" in source:
        return STATUS_VERBAL
    if "report" in source or read:
        # Non-empty read without explicit verbal → treat as report-backed only if
        # source says report, otherwise still radiologist_report when source empty
        # but read present (they filled a confirmation). Do not invent content.
        if "report" in source or source == "" or "radiologist" in source:
            return STATUS_REPORT
        if "verbal" in source:
            return STATUS_VERBAL
        return STATUS_REPORT
    return STATUS_UNVERIFIED


def apply_radiologist_status_to_manifest(
    manifest_path: Path | None = None,
    rad_path: Path | None = None,
) -> pd.DataFrame:
    """Update manifest.csv with tumor_free_status / other_findings / read_date."""
    mpath = manifest_path or (SPLIT_DIR / "manifest.csv")
    df = pd.read_csv(mpath)
    rad = load_radiologist_reads(rad_path)
    by_id: dict[str, dict[str, Any]] = {}
    for _, r in rad.iterrows():
        by_id[str(r["study_id"])] = {
            "radiologist_read": r.get("radiologist_read", ""),
            "other_findings": r.get("other_findings", ""),
            "source": r.get("source", ""),
            "date": r.get("date", ""),
        }

    statuses: list[str] = []
    findings: list[str] = []
    dates: list[str] = []
    phrases: list[str] = []
    for _, row in df.iterrows():
        sid = str(row["study_id"])
        if row.get("cohort") != "negative":
            # Positives are not tumor-free claims
            statuses.append("")
            findings.append("")
            dates.append("")
            phrases.append("")
            continue
        if sid in by_id:
            r = by_id[sid]
            st = status_from_row(r)
            statuses.append(st)
            findings.append(str(r.get("other_findings", "") or ""))
            dates.append(str(r.get("date", "") or ""))
            phrases.append(STATUS_PHRASE[st])
        else:
            statuses.append(STATUS_UNVERIFIED)
            findings.append("")
            dates.append("")
            phrases.append(STATUS_PHRASE[STATUS_UNVERIFIED])

    df["tumor_free_status"] = statuses
    df["other_findings"] = findings
    df["read_date"] = dates
    # Replace free-text claim with per-scan status phrase for negatives
    if "tumor_free_claim" in df.columns:
        df["tumor_free_claim"] = [
            ph if cohort == "negative" else ""
            for ph, cohort in zip(phrases, df["cohort"])
        ]
    df.to_csv(mpath, index=False)
    return df


def status_counts(df: pd.DataFrame) -> dict[str, int]:
    neg = df[df["cohort"] == "negative"]
    return neg["tumor_free_status"].value_counts(dropna=False).to_dict()


def list_unverified_selected(df: pd.DataFrame) -> list[str]:
    sel = df[(df["cohort"] == "negative") & (df["selected"] == True)]  # noqa: E712
    return [
        str(r.study_id)
        for r in sel.itertuples()
        if str(getattr(r, "tumor_free_status", "")) == STATUS_UNVERIFIED
    ]


def suggest_confirmed_replacements(
    df: pd.DataFrame,
    unverified_selected: list[str],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Suggest unused radiologist-confirmed negatives matched on prior criteria.

    Does **not** change selection. Matching: all4, thickness within 1 mm of
    selected-unverified medians where available, same t1c_status, prefer same
    manufacturer / field strength.
    """
    confirmed = df[
        (df["cohort"] == "negative")
        & (df["unused_negative"] == True)  # noqa: E712
        & (df["tumor_free_status"].isin([STATUS_REPORT, STATUS_VERBAL]))
        & (df["has_all_four_registered"] == True)  # noqa: E712
    ].copy()
    if confirmed.empty or not unverified_selected:
        return []

    unver = df[df["study_id"].isin(unverified_selected)]
    suggestions: list[dict[str, Any]] = []
    used: set[str] = set()
    rng = np.random.default_rng(seed)

    for sid in unverified_selected:
        row = unver[unver["study_id"] == sid]
        if row.empty:
            continue
        u = row.iloc[0]
        pool = confirmed[~confirmed["study_id"].isin(used)].copy()
        # t1c status
        pool = pool[pool["t1c_status"] == u["t1c_status"]]
        # thickness within 1 mm for T1/T2/FLAIR when both present
        for mod in ("T1", "T2", "FLAIR"):
            col = f"slice_thickness_{mod}_mm"
            if col not in pool.columns:
                continue
            uv = u.get(col)
            if pd.isna(uv):
                continue
            pool = pool[
                pool[col].isna() | ((pool[col] - float(uv)).abs() <= 1.0)
            ]
        # manufacturer / field
        if "manufacturer" in pool.columns and pd.notna(u.get("manufacturer")):
            pref = pool[
                pool["manufacturer"].astype(str).str.lower().str.contains(
                    str(u["manufacturer"]).split()[0].lower(), na=False
                )
            ]
            if not pref.empty:
                pool = pref
        if pool.empty:
            suggestions.append(
                {
                    "unverified_selected": sid,
                    "suggested_replacement": None,
                    "reason": "no radiologist-confirmed unused match",
                }
            )
            continue
        pick = pool.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        used.add(str(pick["study_id"]))
        suggestions.append(
            {
                "unverified_selected": sid,
                "suggested_replacement": str(pick["study_id"]),
                "replacement_status": str(pick["tumor_free_status"]),
                "reason": (
                    f"unused confirmed; t1c={pick['t1c_status']}; "
                    f"manufacturer={pick.get('manufacturer')}"
                ),
            }
        )
    return suggestions


def bootstrap_fraction_ci(
    flags: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for fraction of True, with percentile bootstrap CI."""
    flags = np.asarray(flags, dtype=bool)
    n = len(flags)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(flags.mean())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = flags[rng.integers(0, n, size=n)]
        boots.append(float(sample.mean()))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def write_status_reports(df: pd.DataFrame, suggestions: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts = status_counts(df)
    unver = list_unverified_selected(df)
    lines = [
        "# Radiologist tumor-free status",
        "",
        "Filled only from `data/processed/real_patients/radiologist_reads.csv`. "
        "Empty / missing rows → `claimed_unverified`. No values were invented.",
        "",
        "## Counts (negative cohort)",
        "",
    ]
    for k in (STATUS_REPORT, STATUS_VERBAL, STATUS_UNVERIFIED):
        lines.append(f"- **{k}**: {counts.get(k, 0)}")
    lines += [
        "",
        f"Selected negatives still `{STATUS_UNVERIFIED}`: **{len(unver)}**",
        "",
    ]
    if unver:
        lines.append("### Unverified selected (selection unchanged)")
        lines.append("")
        for s in unver:
            lines.append(f"- `{s}`")
        lines.append("")
    if suggestions:
        lines += [
            "### Suggested replacements (awaiting your confirmation)",
            "",
            "| unverified_selected | suggested_replacement | status | reason |",
            "|---|---|---|---|",
        ]
        for s in suggestions:
            lines.append(
                f"| {s['unverified_selected']} | {s.get('suggested_replacement') or '—'} | "
                f"{s.get('replacement_status', '')} | {s.get('reason', '')} |"
            )
        lines.append("")
        lines.append("Selection was **not** changed.")
        lines.append("")
    elif unver:
        lines.append(
            "No radiologist-confirmed unused replacements available yet "
            "(fill `radiologist_reads.csv` first)."
        )
        lines.append("")

    (REPORT_DIR / "RADIOLOGIST_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "replacement_suggestions.json").write_text(
        json.dumps(
            {
                "unverified_selected": unver,
                "suggestions": suggestions,
                "status_counts": counts,
                "selection_changed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
