#!/usr/bin/env python3
"""Build an anonymized reader pack for unverified draft masks.

For each study with ``claude_draft_label.nii.gz``: study-id-only folder,
T1c/FLAIR triplanar PNGs at the lesion centroid (± contour), and
``reader_form.csv``. No patient names/dates; NIfTI headers stripped of
descriptive metadata.

Drafts are **unverified**. The form is the only thing to return.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PROJECT_ROOT  # noqa: E402
from segmentation.hdglio_labels import brats_regions_from_classic  # noqa: E402

LABELING_ROOT = PROJECT_ROOT / "outputs" / "batch2_labeling"
OUT_ROOT = PROJECT_ROOT / "outputs" / "reader_pack"
DEFAULT_DRAFT = "claude_draft_label.nii.gz"

FORM_COLUMNS = [
    "study_id",
    "lesion_present",
    "draft_correct",
    "what_is_wrong",
    "reader_initials",
]


def _norm(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(float)
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return np.zeros_like(sl)
    p2, p98 = np.percentile(finite, (2, 98))
    if p98 <= p2:
        return np.zeros_like(sl)
    return np.clip((sl - p2) / (p98 - p2), 0, 1)


def _contour(mask2d: np.ndarray) -> np.ndarray:
    m = np.asarray(mask2d, dtype=bool)
    if not np.any(m):
        return m
    return m & ~ndi.binary_erosion(m, iterations=1)


def lesion_centroid_vox(draft: np.ndarray) -> tuple[int, int, int]:
    regs = brats_regions_from_classic(draft)
    mask = regs["ET"] if np.any(regs["ET"]) else regs["WT"]
    if not np.any(mask):
        return tuple(s // 2 for s in draft.shape)  # type: ignore[return-value]
    c = ndi.center_of_mass(mask)
    return tuple(int(round(c[i])) for i in range(3))  # type: ignore[return-value]


def save_triplanar(
    out_png: Path,
    *,
    volume: np.ndarray,
    contour: np.ndarray | None,
    centroid: tuple[int, int, int],
    title: str,
) -> None:
    x, y, z = centroid
    x = int(np.clip(x, 0, volume.shape[0] - 1))
    y = int(np.clip(y, 0, volume.shape[1] - 1))
    z = int(np.clip(z, 0, volume.shape[2] - 1))
    slices = [
        ("axial", volume[:, :, z], None if contour is None else contour[:, :, z]),
        ("coronal", volume[:, y, :], None if contour is None else contour[:, y, :]),
        ("sagittal", volume[x, :, :], None if contour is None else contour[x, :, :]),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (name, sl, csl) in zip(axs, slices):
        base = _norm(sl)
        rgb = np.stack([base, base, base], axis=-1)
        if csl is not None:
            edge = _contour(csl)
            rgb[edge] = (0.95, 0.2, 0.15)
        ax.imshow(np.rot90(rgb), origin="lower")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def strip_nifti_metadata(src: Path, dest: Path) -> None:
    """Copy array + affine only; blank descriptive header fields."""
    img = nib.load(str(src))
    data = np.asanyarray(img.dataobj)
    out = nib.Nifti1Image(data, img.affine)
    hdr = out.header
    hdr["descrip"] = b""
    hdr["aux_file"] = b""
    # intent / intent_name cleared
    try:
        hdr.set_intent("none")
    except Exception:  # noqa: BLE001
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dest))


def pack_study(pkg: Path, out_root: Path, draft_name: str) -> Path:
    study_id = pkg.name
    dest = out_root / study_id
    dest.mkdir(parents=True, exist_ok=True)

    t1c = np.asanyarray(nib.load(str(pkg / "T1c.nii.gz")).dataobj).astype(np.float32)
    flair = np.asanyarray(nib.load(str(pkg / "FLAIR.nii.gz")).dataobj).astype(np.float32)
    draft = np.rint(np.asanyarray(nib.load(str(pkg / draft_name)).dataobj)).astype(np.int16)
    wt = brats_regions_from_classic(draft)["WT"]
    cent = lesion_centroid_vox(draft)

    save_triplanar(
        dest / "T1c_no_contour.png",
        volume=t1c,
        contour=None,
        centroid=cent,
        title=f"{study_id} T1c (no contour)",
    )
    save_triplanar(
        dest / "T1c_with_draft_contour.png",
        volume=t1c,
        contour=wt,
        centroid=cent,
        title=f"{study_id} T1c + draft contour (unverified)",
    )
    save_triplanar(
        dest / "FLAIR_no_contour.png",
        volume=flair,
        contour=None,
        centroid=cent,
        title=f"{study_id} FLAIR (no contour)",
    )
    save_triplanar(
        dest / "FLAIR_with_draft_contour.png",
        volume=flair,
        contour=wt,
        centroid=cent,
        title=f"{study_id} FLAIR + draft contour (unverified)",
    )

    strip_nifti_metadata(pkg / "T1c.nii.gz", dest / "T1c.nii.gz")
    strip_nifti_metadata(pkg / "FLAIR.nii.gz", dest / "FLAIR.nii.gz")
    strip_nifti_metadata(pkg / draft_name, dest / "draft_label.nii.gz")

    (dest / "STUDY_ID.txt").write_text(
        f"{study_id}\n\nDraft status: draft_unverified\n", encoding="utf-8"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeling-root", type=Path, default=LABELING_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--draft-name", type=str, default=DEFAULT_DRAFT)
    args = parser.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)
    pkgs = sorted(
        p
        for p in args.labeling_root.iterdir()
        if p.is_dir()
        and (p / args.draft_name).is_file()
        and (p / "T1c.nii.gz").is_file()
        and (p / "FLAIR.nii.gz").is_file()
    )
    if not pkgs:
        print(f"No draft studies under {args.labeling_root}")
        return 1

    form_rows = []
    for pkg in pkgs:
        pack_study(pkg, args.out_root, args.draft_name)
        form_rows.append(
            {
                "study_id": pkg.name,
                "lesion_present": "",
                "draft_correct": "",
                "what_is_wrong": "",
                "reader_initials": "",
            }
        )
        print(f"Packed {pkg.name}")

    form_path = args.out_root / "reader_form.csv"
    with form_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FORM_COLUMNS)
        w.writeheader()
        w.writerows(form_rows)

    (args.out_root / "README.md").write_text(
        "# Reader pack (draft / unverified)\n\n"
        "These folders contain **unverified draft** tumor outlines for qualitative "
        "review only. They are **not** clinical reference masks.\n\n"
        "- Each subfolder is identified by **study_id only** (no patient names or dates).\n"
        "- PNGs: axial / coronal / sagittal of T1c and FLAIR at the lesion centroid, "
        "with and without the draft contour.\n"
        "- NIfTI copies have descriptive header fields stripped.\n\n"
        "## What to return\n\n"
        "Fill in and return **only** `reader_form.csv`:\n\n"
        "| column | values |\n"
        "|---|---|\n"
        "| study_id | (pre-filled) |\n"
        "| lesion_present | yes / no / unsure |\n"
        "| draft_correct | correct / partly / wrong |\n"
        "| what_is_wrong | free text |\n"
        "| reader_initials | your initials |\n\n"
        "Do not redistribute the images outside the review.\n",
        encoding="utf-8",
    )
    print(f"Reader pack → {args.out_root} ({len(pkgs)} studies); form → {form_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
