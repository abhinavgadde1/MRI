"""Regression: model 4-channel inputs must follow MODALITY_NAMES order."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from preprocessing.h5_to_nifti import MODALITY_NAMES

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (ROOT / "src", ROOT / "scripts", ROOT / "tests")

WRONG_ASSIGN = re.compile(
    r"""MODALITY_NAMES\s*=\s*\(\s*["']T1c["']\s*,\s*["']T1["']\s*,\s*["']T2["']\s*,\s*["']FLAIR["']\s*\)"""
)


def test_modality_names_canonical():
    assert MODALITY_NAMES == ("FLAIR", "T1", "T1c", "T2")


def test_no_hardcoded_wrong_modality_names_assignment():
    """Fail if any .py redefines MODALITY_NAMES as the stale (T1c,T1,T2,FLAIR) order."""
    offenders: list[str] = []
    for base in SCAN_DIRS:
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part.startswith(".") for part in path.parts):
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if WRONG_ASSIGN.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "Hardcoded wrong BraTS channel order assignment. "
        "Import preprocessing.h5_to_nifti.MODALITY_NAMES instead.\n"
        + "\n".join(offenders)
    )


def _const_str_tuple(node: ast.AST) -> tuple[str, ...] | None:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    names: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names.append(elt.value)
        else:
            return None
    return tuple(names)


def test_model_image_stacks_use_modality_names_order():
    """AST: literals used to build ``image`` modality stacks must match MODALITY_NAMES.

    Catches ``"image": [str(...[m]) for m in ("T1c", "T1", ...)]`` and similar
    comprehension/list constructions. Ignores unrelated 4-name loops (e.g. copy
    metadata for T1/T1c/T2/FLAIR).
    """
    bad: list[str] = []
    for base in (ROOT / "src", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src, filename=str(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                # "image": [ ... for m in (T1c, T1, T2, FLAIR) ]
                if isinstance(node, ast.Dict):
                    for key, val in zip(node.keys, node.values):
                        if not (
                            isinstance(key, ast.Constant) and key.value == "image"
                        ):
                            continue
                        # ListComp / GeneratorExp with iter = wrong tuple
                        comps = []
                        if isinstance(val, ast.ListComp):
                            comps = val.generators
                        elif isinstance(val, ast.List) and val.elts:
                            # maybe nested
                            for elt in val.elts:
                                if isinstance(elt, ast.ListComp):
                                    comps.extend(elt.generators)
                        for gen in comps:
                            names = _const_str_tuple(gen.iter)
                            if names is None:
                                continue
                            if set(names) == set(MODALITY_NAMES) and names != MODALITY_NAMES:
                                bad.append(
                                    f"{path.relative_to(ROOT)}:{node.lineno}: "
                                    f'image stack order {names}'
                                )
                # MODALITY_NAMES = (wrong...)
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "MODALITY_NAMES":
                            names = _const_str_tuple(node.value)
                            if names is not None and names != MODALITY_NAMES:
                                bad.append(
                                    f"{path.relative_to(ROOT)}:{node.lineno}: "
                                    f"MODALITY_NAMES = {names}"
                                )
    assert not bad, (
        "Model input channel order must be "
        f"{MODALITY_NAMES}. Offenders:\n" + "\n".join(bad)
    )
