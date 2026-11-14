"""Review montages for HD-GLIO vs v3 second opinion (drafts only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from segmentation.hdglio_labels import DRAFT_DISCLAIMER, brats_regions_from_classic


def _norm(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(float)
    p2, p98 = np.percentile(sl, (2, 98))
    if p98 <= p2:
        return np.zeros_like(sl)
    return np.clip((sl - p2) / (p98 - p2), 0, 1)


def _contour_edges(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    eroded = ndi.binary_erosion(m, iterations=1)
    return m & ~eroded


def _pick_slices(mask: np.ndarray, n: int = 6) -> list[int]:
    m = np.asarray(mask, dtype=bool)
    z = m.shape[2]
    if not np.any(m):
        return [int(round(i * (z - 1) / (n - 1))) for i in range(n)]
    zs = np.where(m.any(axis=(0, 1)))[0]
    if len(zs) <= n:
        # pad with neighbors
        mid = list(zs)
        while len(mid) < n:
            mid.append(int(np.clip(zs[len(zs) // 2], 0, z - 1)))
        return sorted(mid)[:n]
    idx = np.linspace(0, len(zs) - 1, n).astype(int)
    return [int(zs[i]) for i in idx]


def save_review_montage(
    study_id: str,
    t1c_path: Path,
    flair_path: Path,
    v3_path: Path | None,
    hd_path: Path,
    out_png: Path,
    swi_path: Path | None = None,
) -> dict[str, Any]:
    """6 axial slices: T1c+v3, T1c+HD-GLIO, FLAIR+both, optional SWI+HD-GLIO."""
    t1c = np.asanyarray(nib.load(str(t1c_path)).dataobj).astype(np.float32)
    flair = np.asanyarray(nib.load(str(flair_path)).dataobj).astype(np.float32)
    hd = np.rint(np.asanyarray(nib.load(str(hd_path)).dataobj)).astype(np.int16)
    v3 = (
        np.rint(np.asanyarray(nib.load(str(v3_path)).dataobj)).astype(np.int16)
        if v3_path is not None and Path(v3_path).is_file()
        else np.zeros_like(hd)
    )
    if v3.shape != hd.shape:
        v3 = np.zeros_like(hd)
    hd_regs = brats_regions_from_classic(hd)
    v3_regs = brats_regions_from_classic(v3)
    union = hd_regs["WT"] | v3_regs["WT"]
    slices = _pick_slices(union, 6)

    swi = None
    if swi_path is not None and Path(swi_path).is_file():
        swi_img = nib.load(str(swi_path))
        swi = np.asanyarray(swi_img.dataobj).astype(np.float32)
        if swi.shape != t1c.shape:
            swi = None

    n_rows = 4 if swi is not None else 3
    fig, axs = plt.subplots(n_rows, 6, figsize=(14, 2.4 * n_rows))
    if n_rows == 1:
        axs = np.asarray([axs])
    row_titles = [
        "T1c + v3 contour",
        "T1c + HD-GLIO contour",
        "FLAIR + both contours",
    ]
    if swi is not None:
        row_titles.append(
            "SWI + v3/HD-GLIO (dark signal = review flag, not a diagnosis)"
        )

    for col, z in enumerate(slices):
        base_t1c = _norm(t1c[:, :, z])
        base_flair = _norm(flair[:, :, z])
        v3e = _contour_edges(v3_regs["WT"][:, :, z])
        hde = _contour_edges(hd_regs["WT"][:, :, z])

        rgb0 = np.stack([base_t1c, base_t1c, base_t1c], axis=-1)
        rgb0[v3e] = (0.15, 0.85, 0.25)
        axs[0, col].imshow(np.rot90(rgb0), origin="lower")

        rgb1 = np.stack([base_t1c, base_t1c, base_t1c], axis=-1)
        rgb1[hde] = (0.95, 0.25, 0.2)
        axs[1, col].imshow(np.rot90(rgb1), origin="lower")

        rgb2 = np.stack([base_flair, base_flair, base_flair], axis=-1)
        rgb2[v3e] = (0.15, 0.85, 0.25)
        rgb2[hde] = (0.95, 0.25, 0.2)
        axs[2, col].imshow(np.rot90(rgb2), origin="lower")

        if swi is not None:
            base_swi = _norm(swi[:, :, z])
            rgb3 = np.stack([base_swi, base_swi, base_swi], axis=-1)
            rgb3[v3e] = (0.15, 0.85, 0.25)
            rgb3[hde] = (0.95, 0.25, 0.2)
            axs[3, col].imshow(np.rot90(rgb3), origin="lower")

        for r in range(n_rows):
            axs[r, col].axis("off")
            if col == 0:
                axs[r, col].set_ylabel(row_titles[r], fontsize=8)

    fig.suptitle(
        f"{study_id} — HD-GLIO vs v3 (drafts only)\n{DRAFT_DISCLAIMER}",
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(out_png), "slices": slices, "has_swi_panel": swi is not None}
