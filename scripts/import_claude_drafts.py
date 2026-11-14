#!/usr/bin/env python3
"""Import Claude draft masks into batch2 labeling packages (draft_unverified).

Expects ``data/external/claude_drafts/batch2_<id>_claude_draft_label.nii.gz``.
Copies to ``outputs/batch2_labeling/<id>/claude_draft_label.nii.gz`` after
shape/affine match against that study's T1c. Updates ``.itksnap`` /
``open_itksnap.sh`` and marks ``package_meta.json`` status ``draft_unverified``.

Does not train, does not modify checkpoints / BraTS splits / existing metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PROJECT_ROOT  # noqa: E402

DEFAULT_SRC = PROJECT_ROOT / "data" / "external" / "claude_drafts"
DEFAULT_LABELING = PROJECT_ROOT / "outputs" / "batch2_labeling"
DRAFT_NAME = "claude_draft_label.nii.gz"
STATUS = "draft_unverified"
FNAME_RE = re.compile(r"^(batch2_\d+)_claude_draft_label\.nii\.gz$")


def assert_same_grid(draft: nib.Nifti1Image, t1c: nib.Nifti1Image, *, atol: float = 1e-3) -> None:
    if draft.shape != t1c.shape:
        raise ValueError(f"shape mismatch draft={draft.shape} vs T1c={t1c.shape}")
    if not np.allclose(draft.affine, t1c.affine, atol=atol, rtol=0.0):
        raise ValueError("affine mismatch vs T1c")


def update_package_meta(pkg: Path, *, source_name: str) -> dict:
    meta_path = pkg / "package_meta.json"
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["claude_draft"] = {
        "filename": DRAFT_NAME,
        "status": STATUS,
        "source": source_name,
        "label_convention": "BraTS classic 1=NCR, 2=ED, 4=ET",
        "note": "Draft / unverified — not a clinical reference mask.",
    }
    meta["draft_status"] = STATUS
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def update_itksnap_workspace(pkg: Path, study_id: str) -> Path:
    """Ensure v3_draft + claude_draft segmentation layers exist in the workspace."""
    path = pkg / f"{study_id}.itksnap"
    if path.is_file():
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except ET.ParseError:
            root = None
    else:
        root = None

    if root is None:
        root = ET.Element("ITKSnap")
        root.set("version", "2.0")
        project = ET.SubElement(root, "project")
        layers = ET.SubElement(project, "layers")
        main = ET.SubElement(layers, "layer")
        main.set("type", "main")
        ET.SubElement(main, "filename").text = "T1c.nii.gz"
        ET.SubElement(main, "nickname").text = "T1c"
        for nick, fname in (
            ("FLAIR", "FLAIR.nii.gz"),
            ("T2", "T2.nii.gz"),
            ("T1", "T1.nii.gz"),
        ):
            lay = ET.SubElement(layers, "layer")
            lay.set("type", "overlay")
            ET.SubElement(lay, "filename").text = fname
            ET.SubElement(lay, "nickname").text = nick
            ET.SubElement(lay, "opacity").text = "0.35"
    else:
        project = root.find("project")
        if project is None:
            project = ET.SubElement(root, "project")
        layers = project.find("layers")
        if layers is None:
            layers = ET.SubElement(project, "layers")

    # Remove existing claude seg layers; keep/ensure v3
    for lay in list(layers.findall("layer")):
        if lay.get("type") != "segmentation":
            continue
        fn = (lay.findtext("filename") or "").strip()
        nick = (lay.findtext("nickname") or "").strip()
        if fn == DRAFT_NAME or nick in {"claude_draft", "claude"}:
            layers.remove(lay)

    has_v3 = any(
        (lay.findtext("filename") or "").strip() == "v3_draft_label.nii.gz"
        for lay in layers.findall("layer")
        if lay.get("type") == "segmentation"
    )
    if not has_v3 and (pkg / "v3_draft_label.nii.gz").is_file():
        seg = ET.SubElement(layers, "layer")
        seg.set("type", "segmentation")
        ET.SubElement(seg, "filename").text = "v3_draft_label.nii.gz"
        ET.SubElement(seg, "nickname").text = "v3_draft"

    seg_c = ET.SubElement(layers, "layer")
    seg_c.set("type", "segmentation")
    ET.SubElement(seg_c, "filename").text = DRAFT_NAME
    ET.SubElement(seg_c, "nickname").text = "claude_draft"

    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_open_itksnap(pkg: Path) -> Path:
    """Rewrite open_itksnap.sh with v3 as -s and claude draft as an extra overlay."""
    t1c = pkg / "T1c.nii.gz"
    v3 = pkg / "v3_draft_label.nii.gz"
    claude = pkg / DRAFT_NAME
    overlays = [
        pkg / "FLAIR.nii.gz",
        pkg / "T2.nii.gz",
        pkg / "T1.nii.gz",
        claude,
    ]
    overlay_args = " ".join(f'"{p}"' for p in overlays if p.is_file())
    seg = v3 if v3.is_file() else claude
    cmd = f'itksnap -g "{t1c}" -s "{seg}" -o {overlay_args}'
    path = pkg / "open_itksnap.sh"
    path.write_text(
        "#!/bin/zsh\n"
        "# Open this study in ITK-SNAP (requires `itksnap` on PATH).\n"
        "# Prefer the .itksnap workspace (includes v3_draft + claude_draft).\n"
        f"# Claude draft status: {STATUS}\n"
        f'cd "{pkg}"\n'
        f"{cmd}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def import_one(
    src: Path,
    labeling_root: Path,
) -> dict:
    m = FNAME_RE.match(src.name)
    if not m:
        raise ValueError(f"Unexpected filename: {src.name}")
    study_id = m.group(1)
    pkg = labeling_root / study_id
    t1c_path = pkg / "T1c.nii.gz"
    if not t1c_path.is_file():
        raise FileNotFoundError(f"Missing labeling package T1c: {t1c_path}")

    draft_img = nib.load(str(src))
    t1c_img = nib.load(str(t1c_path))
    assert_same_grid(draft_img, t1c_img)

    dest = pkg / DRAFT_NAME
    # Re-save on T1c header to lock grid metadata; do not alter source file.
    data = np.rint(np.asanyarray(draft_img.dataobj)).astype(np.int16)
    nib.save(nib.Nifti1Image(data, t1c_img.affine, t1c_img.header), str(dest))

    update_package_meta(pkg, source_name=src.name)
    update_itksnap_workspace(pkg, study_id)
    update_open_itksnap(pkg)
    return {
        "study_id": study_id,
        "source": str(src),
        "dest": str(dest),
        "status": STATUS,
        "shape": list(data.shape),
    }


def discover_sources(src_dir: Path) -> list[Path]:
    if not src_dir.is_dir():
        return []
    return sorted(p for p in src_dir.glob("batch2_*_claude_draft_label.nii.gz") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--labeling-root", type=Path, default=DEFAULT_LABELING)
    args = parser.parse_args(argv)

    sources = discover_sources(args.src_dir)
    if not sources:
        print(f"No Claude draft files found under {args.src_dir}")
        return 1

    rows = []
    for src in sources:
        try:
            row = import_one(src, args.labeling_root)
            rows.append(row)
            print(f"OK {row['study_id']} → {row['dest']} ({STATUS})")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {src.name}: {exc}")
            rows.append({"source": str(src), "error": str(exc)})

    out = args.src_dir / "import_log.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"Imported {n_ok}/{len(sources)}; log → {out}")
    return 0 if n_ok == len(sources) else 2


if __name__ == "__main__":
    raise SystemExit(main())
