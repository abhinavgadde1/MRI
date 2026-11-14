"""Stable raw↔anon study ID mapping (local gitignored file).

The mapping lives at ``data/private/id_map.csv`` (under gitignored ``data/``).
Committed artifacts must only contain ``patient_A##`` style IDs.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_ID_MAP_PATH = PROJECT_ROOT / "data" / "private" / "id_map.csv"

# Folder / roster tokens that must never appear in committed text.
# Pattern assembled so the repo tip does not contain the literal DICOM suffix.
_RAW_STUDY_RE = re.compile(
    r"(?:"
    r"[0-9]{10,}_" + "MR_" + r"Study\d+(?: \d+)?"
    r"|"
    r"[A-Za-z][A-Za-z0-9. ]*?\d{2}" + r"Y [MF]" + r"(?:_" + "MR_" + r"Study\d+)?"
    r")"
)


def load_id_map(path: Path | None = None) -> dict[str, str]:
    """Return ``{raw_id: anon_id}`` from CSV. Empty dict if missing."""
    path = path or DEFAULT_ID_MAP_PATH
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = (row.get("raw_id") or "").strip()
            anon = (row.get("anon_id") or "").strip()
            if raw and anon:
                out[raw] = anon
    return out


def save_id_map(mapping: dict[str, str], path: Path | None = None) -> Path:
    """Write ``raw_id,anon_id`` CSV (parent dirs created)."""
    path = path or DEFAULT_ID_MAP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["raw_id", "anon_id"])
        writer.writeheader()
        for raw in sorted(mapping, key=lambda r: (mapping[r], r)):
            writer.writerow({"raw_id": raw, "anon_id": mapping[raw]})
    return path


def reverse_map(mapping: dict[str, str]) -> dict[str, str]:
    return {anon: raw for raw, anon in mapping.items()}


def anonymize_study_id(
    raw_id: str,
    mapping: dict[str, str] | None = None,
    *,
    assign_missing: bool = False,
) -> str:
    """Map a raw study folder name to ``patient_A##``.

    If ``assign_missing`` and the id is absent, append the next free anon ID
    into ``mapping`` (mutates the dict).
    """
    raw_id = str(raw_id).strip()
    if raw_id.startswith("patient_A") and re.fullmatch(r"patient_A\d{2,}", raw_id):
        return raw_id
    mapping = mapping if mapping is not None else load_id_map()
    if raw_id in mapping:
        return mapping[raw_id]
    if not assign_missing:
        # Already-anonymous or unknown — leave unchanged only if it looks anon.
        if re.fullmatch(r"patient_A\d{2,}", raw_id):
            return raw_id
        raise KeyError(
            f"No anon ID for {raw_id!r}. Add it to {DEFAULT_ID_MAP_PATH} "
            "or pass assign_missing=True when building the map."
        )
    used = {int(m.group(1)) for a in mapping.values() if (m := re.fullmatch(r"patient_A(\d+)", a))}
    n = 1
    while n in used:
        n += 1
    anon = f"patient_A{n:02d}"
    mapping[raw_id] = anon
    return anon


def resolve_raw_ids(
    anon_or_raw: list[str],
    *,
    mapping: dict[str, str] | None = None,
) -> list[str]:
    """Resolve CLI study args: anon→raw via map; raw IDs pass through if on disk map."""
    mapping = mapping if mapping is not None else load_id_map()
    rev = reverse_map(mapping)
    out: list[str] = []
    for item in anon_or_raw:
        item = str(item).strip()
        if item in rev:
            out.append(rev[item])
        elif item in mapping:
            # already raw
            out.append(item)
        else:
            out.append(item)
    return out


def replace_raw_ids_in_text(text: str, mapping: dict[str, str]) -> str:
    """Replace all known raw IDs (longest first) in ``text``."""
    # Also catch unmapped tokens matching the raw pattern by assigning? No —
    # only replace keys present in mapping so the map stays authoritative.
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for raw, anon in items:
        text = text.replace(raw, anon)
    return text


def find_raw_study_tokens(text: str) -> set[str]:
    return {m.group(0) for m in _RAW_STUDY_RE.finditer(text)}
