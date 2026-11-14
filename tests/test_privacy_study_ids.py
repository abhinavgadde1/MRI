"""Tests for anonymized study ID helpers."""

from __future__ import annotations

from pathlib import Path

from privacy.study_ids import (
    anonymize_study_id,
    load_id_map,
    replace_raw_ids_in_text,
    resolve_raw_ids,
    save_id_map,
)


def test_anonymize_and_resolve_roundtrip(tmp_path: Path):
    mapping = {"RAW_Study1": "patient_A01"}
    path = tmp_path / "id_map.csv"
    save_id_map(mapping, path)
    loaded = load_id_map(path)
    assert loaded["RAW_Study1"] == "patient_A01"
    assert anonymize_study_id("RAW_Study1", loaded) == "patient_A01"
    assert resolve_raw_ids(["patient_A01"], mapping=loaded) == ["RAW_Study1"]


def test_replace_raw_ids_in_text():
    raw_a = "RAWFOLDER_001"
    raw_b = "OTHERFOLDER_002"
    mapping = {raw_a: "patient_A01", raw_b: "patient_A02"}
    text = f"see {raw_a} and {raw_b}"
    out = replace_raw_ids_in_text(text, mapping)
    assert raw_a not in out and raw_b not in out
    assert "patient_A01" in out and "patient_A02" in out
