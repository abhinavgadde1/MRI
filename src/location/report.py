"""Run optional location report: register, warp atlases, metrics, QC, mesh colors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from location.atlases import (
    ensure_mni_and_harvard_oxford,
    jhu_tract_mask_mni,
    license_notes,
)
from location.metrics import region_location_block
from location.register import register_t1_to_mni, warp_label_to_patient

logger = logging.getLogger(__name__)

LOCATION_REPORT_ENABLED_DEFAULT = False

# Largest connected component must hold at least this fraction of WT volume
# or the case is treated as below the location plausibility gate.
LCC_FRACTION_MIN = 0.50

LOBE_COLORS: dict[str, str] = {
    "frontal": "#F4A261",
    "parietal": "#2A9D8F",
    "temporal": "#E9C46A",
    "occipital": "#9B5DE5",
    "insula": "#00BBF9",
    "cingulate": "#F15BB5",
    "other": "#ADB5BD",
}

REGION_KEYS = (
    ("whole_tumor", "WT"),
    ("tumor_core", "TC"),
    ("enhancing_tumor", "ET"),
)


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel spacing (mm) from affine column norms (handles oblique grids)."""
    a = np.asarray(affine, dtype=np.float64)
    return tuple(float(np.linalg.norm(a[:3, i])) for i in range(3))  # type: ignore[return-value]


def _ventricle_mask(subcortical: np.ndarray, labels: Mapping[int, str]) -> np.ndarray:
    ids = [
        i
        for i, name in labels.items()
        if "ventricle" in name.lower()
    ]
    mask = np.zeros(subcortical.shape, dtype=bool)
    for i in ids:
        mask |= subcortical == int(i)
    return mask


def _save_qc_overlay(
    t1: np.ndarray,
    atlas: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    """Axial / coronal / sagittal overlay of atlas edges on T1."""
    t1 = np.asarray(t1, dtype=np.float32)
    atlas = np.asarray(atlas)
    # Pick slices near tumor/atlas mass.
    if np.any(atlas > 0):
        com = np.array(np.where(atlas > 0)).mean(axis=1).astype(int)
    else:
        com = np.array(t1.shape) // 2
    axes = [
        ("axial", 2, int(np.clip(com[2], 0, t1.shape[2] - 1))),
        ("coronal", 1, int(np.clip(com[1], 0, t1.shape[1] - 1))),
        ("sagittal", 0, int(np.clip(com[0], 0, t1.shape[0] - 1))),
    ]

    def _plane(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
        if axis == 0:
            return vol[index]
        if axis == 1:
            return vol[:, index]
        return vol[:, :, index]

    def _norm(sl: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(sl, (2, 98))
        if p98 <= p2:
            return np.zeros_like(sl)
        return np.clip((sl - p2) / (p98 - p2), 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, axis, idx) in zip(axs, axes):
        img = _norm(_plane(t1, axis, idx))
        lab = _plane(atlas, axis, idx)
        ax.imshow(img.T, cmap="gray", origin="lower")
        if np.any(lab > 0):
            ax.contour(lab.T.astype(float), levels=[0.5], colors="#ff6b6b", linewidths=0.6)
        ax.set_title(f"{name} {idx}")
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _lobe_table_html(lobes: dict[str, Any], caption: str) -> str:
    if not lobes:
        return f"<p><i>{caption}: none</i></p>"
    rows = "".join(
        f"<tr><td>{name}</td><td>{v['pct']:.1f}%</td><td>{v['ml']:.2f}</td></tr>"
        for name, v in lobes.items()
    )
    return (
        f"<p><b>{caption}</b></p>"
        f"<table><tr><th>Lobe</th><th>%</th><th>mL</th></tr>{rows}</table>"
    )


def largest_component_fraction(mask: np.ndarray) -> float:
    """Fraction of nonzero mask voxels in the largest connected component."""
    from scipy import ndimage as ndi

    m = np.asarray(mask, dtype=bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    lab, n = ndi.label(m)
    if n == 0:
        return 0.0
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    return float(np.max(sizes) / total)


def assess_location_plausibility(
    wt_mask: np.ndarray | None,
    *,
    spacing_mm: tuple[float, float, float],
    floor_ml: float | None = None,
    lcc_fraction_min: float = LCC_FRACTION_MIN,
) -> dict[str, Any]:
    """Gate location statements when WT is below the volume floor or fragmented.

    Uses ``pipeline.volume_plausibility`` floor (held-out BraTS GT min). Returns
    a dict with ``gated`` bool and a human-readable ``message``.
    """
    from location.qc import gate_message
    from pipeline.volume_plausibility import (
        MIN_PLAUSIBLE_WT_ML,
        assess_wt_plausibility,
    )

    floor = float(MIN_PLAUSIBLE_WT_ML if floor_ml is None else floor_ml)
    if wt_mask is None or not np.any(wt_mask):
        msg = gate_message(0.0, floor)
        return {
            "gated": True,
            "reason": "empty",
            "wt_ml": 0.0,
            "lcc_fraction": 0.0,
            "floor_ml": floor,
            "message": msg,
        }

    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    wt_ml = float(np.asarray(wt_mask, dtype=bool).sum()) * voxel_ml
    lcc_frac = largest_component_fraction(wt_mask)
    plaus = assess_wt_plausibility(wt_ml, floor_ml=floor)

    if plaus.status == "low":
        msg = gate_message(wt_ml, floor)
        return {
            "gated": True,
            "reason": "below_floor",
            "wt_ml": wt_ml,
            "lcc_fraction": lcc_frac,
            "floor_ml": floor,
            "message": msg,
        }
    if lcc_frac < float(lcc_fraction_min):
        msg = (
            gate_message(wt_ml, floor)
            + f"; largest component holds {100.0 * lcc_frac:.0f}% "
            f"(< {100.0 * lcc_fraction_min:.0f}%)"
        )
        return {
            "gated": True,
            "reason": "fragmented",
            "wt_ml": wt_ml,
            "lcc_fraction": lcc_frac,
            "floor_ml": floor,
            "message": msg,
        }
    return {
        "gated": False,
        "reason": None,
        "wt_ml": wt_ml,
        "lcc_fraction": lcc_frac,
        "floor_ml": floor,
        "message": None,
    }


def _location_html_section(payload: dict[str, Any]) -> str:
    if payload.get("plausibility_gated"):
        gate = payload.get("plausibility") or {}
        msg = payload.get("location_statement") or gate.get("message") or (
            "location not reported (small lesion or false positive, unverified)"
        )
        return f"""
<h2>Location</h2>
<div class="warn"><b>Location statement withheld.</b> {msg}</div>
<p>No atlas-based location claims are reported for this case.</p>
"""
    qc = payload.get("registration", {})
    quality = qc.get("quality", "unknown")
    stage = qc.get("stage", "—")
    warn = ""
    if quality in {"caution", "poor", "registration unreliable"}:
        warn = (
            f'<div class="warn"><b>Registration {quality}.</b> '
            f'{qc.get("quality_note", "")} '
            "Do not claim anatomical accuracy without checking "
            "<code>registration/registration_qc.png</code>.</div>"
        )
    if stage == "centered_init":
        warn += (
            '<div class="warn"><b>Coarse registration (translation only).</b> '
            "Transform stage is centered_init; location statements are labeled coarse.</div>"
        )
    blocks = []
    for key, short in REGION_KEYS:
        block = payload.get("regions", {}).get(key) or payload.get("regions", {}).get(short)
        if not block:
            continue
        nearest = block.get("lobes_nearest") or block.get("lobes") or {}
        cortical = block.get("lobes_cortical_only") or {}
        top = block.get("top_structures") or []
        top_rows = "".join(
            f"<tr><td>{t['name']}</td><td>{t['ml']:.2f}</td><td>{t['pct']:.1f}%</td></tr>"
            for t in top
        )
        dist = block.get("distances_mm") or {}
        blocks.append(
            f"""
<h3>{short}</h3>
<p><b>Summary:</b> {block.get('summary', '')}</p>
<p>Hemisphere: {block.get('hemisphere', {}).get('label')}
 (L {block.get('hemisphere', {}).get('left_pct', 0)}% /
  R {block.get('hemisphere', {}).get('right_pct', 0)}%)</p>
{_lobe_table_html(nearest, "Nearest-lobe (headline; ≤15 mm or deep/central)")}
{_lobe_table_html(cortical, "Cortical-only (direct HO overlap, renormalized)")}
<table><tr><th>Top structures</th><th>mL</th><th>%</th></tr>{top_rows}</table>
<p>Distances (mm): ventricles={dist.get('ventricles_mm')},
 CST={dist.get('corticospinal_tract_mm')},
 optic radiation={dist.get('optic_radiation_mm')},
 midline={dist.get('midline_mm')}</p>
"""
        )
    licenses = payload.get("atlas_licenses") or {}
    lic_html = "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in licenses.items())
    jhu = payload.get("jhu", "")
    cov = qc.get("coverage")
    cov_flag = " (partial FOV)" if qc.get("coverage_partial") else ""
    return f"""
<h2>Location</h2>
{warn}
<p>Stage=<b>{stage}</b>; MI before={qc.get('mi_before')}, after={qc.get('mi_after')};
 brain Dice={qc.get('brain_dice')}, ventricle Dice={qc.get('ventricle_dice')},
 surface mean/p95={qc.get('surface_mean_mm')}/{qc.get('surface_p95_mm')} mm,
 coverage={cov}{cov_flag} (quality=<b>{quality}</b>).
 QC overlay:
 <a href="registration/registration_qc.png">registration/registration_qc.png</a>;
 template outline:
 <a href="registration/template_brain_outline_qc.png">template_brain_outline_qc.png</a></p>
<p>{jhu}</p>
{''.join(blocks)}
<h3>Atlas licenses</h3>
<ul>{lic_html}</ul>
"""


def inject_location_into_report(report_html: Path, payload: dict[str, Any]) -> None:
    """Insert a Location section into an existing report.html if present."""
    report_html = Path(report_html)
    if not report_html.is_file():
        return
    text = report_html.read_text(encoding="utf-8")
    section = _location_html_section(payload)
    if "<!-- LOCATION_SECTION -->" in text:
        # Replace previous injection.
        start = text.index("<!-- LOCATION_SECTION -->")
        end_marker = "<!-- /LOCATION_SECTION -->"
        end = text.index(end_marker) + len(end_marker) if end_marker in text else start
        text = text[:start] + f"<!-- LOCATION_SECTION -->\n{section}\n<!-- /LOCATION_SECTION -->" + text[end:]
    elif "</body>" in text:
        text = text.replace(
            "</body>",
            f"<!-- LOCATION_SECTION -->\n{section}\n<!-- /LOCATION_SECTION -->\n</body>",
        )
    else:
        text = text + "\n" + section
    report_html.write_text(text, encoding="utf-8")


def _lobe_masks_from_atlas(
    cortical: np.ndarray,
    cortical_labels: Mapping[int, str],
) -> dict[str, np.ndarray]:
    from location.atlases import cortical_label_to_lobe_map

    lobe_of = cortical_label_to_lobe_map(dict(cortical_labels))
    out: dict[str, np.ndarray] = {}
    for idx, lobe in lobe_of.items():
        out.setdefault(lobe, np.zeros(cortical.shape, dtype=bool))
        out[lobe] |= cortical == int(idx)
    return out


def add_lobes_to_mesh_html(
    out_dir: Path,
    lobe_masks: Mapping[str, np.ndarray],
    tumor_regions: Mapping[str, np.ndarray],
    *,
    case_id: str,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    registration_quality: str = "good",
) -> None:
    """Colour lobes in Plotly mesh viewer; tumor on top; keep cutaway HTML."""
    import plotly.graph_objects as go

    from visualization.plotly_3d import (
        binary_mask_to_trimesh,
        build_labeled_mesh_figure,
        choose_sagittal_keep_low_x,
        mesh3d_from_trimesh,
    )

    if registration_quality in {"poor", "registration unreliable"}:
        logger.info("Skipping lobe mesh colouring due to %s.", registration_quality)
        return

    meshes: dict[str, Any] = {}
    for key in ("whole_tumor", "tumor_core", "enhancing_tumor"):
        m = tumor_regions.get(key)
        if m is not None and np.any(m):
            mesh = binary_mask_to_trimesh(m, spacing_mm)
            if mesh is not None:
                meshes[key] = mesh

    fig = build_labeled_mesh_figure(
        meshes,
        subject_id=case_id,
        title=f"{case_id} · location lobes + tumor",
    )
    lobe_traces = []
    for lobe, mask in sorted(lobe_masks.items()):
        if lobe == "other" or not np.any(mask):
            continue
        mesh = binary_mask_to_trimesh(
            mask, spacing_mm, step_size=2, smooth=True, taubin_iterations=4
        )
        if mesh is None:
            continue
        lobe_traces.append(
            mesh3d_from_trimesh(
                mesh,
                name=f"{lobe.title()} lobe",
                color=LOBE_COLORS.get(lobe, "#ADB5BD"),
                opacity=0.18,
                showlegend=True,
                flatshading=False,
            )
        )
    if lobe_traces:
        new_fig = go.Figure()
        for tr in lobe_traces:
            new_fig.add_trace(tr)
        for tr in fig.data:
            new_fig.add_trace(tr)
        new_fig.update_layout(fig.layout)
        fig = new_fig

    fig.write_html(str(out_dir / "mesh_3d.html"), include_plotlyjs="cdn", full_html=True)

    wt = tumor_regions.get("whole_tumor")
    if wt is None or not np.any(wt):
        wt = next(
            (v for v in tumor_regions.values() if v is not None and np.any(v)),
            None,
        )
    if wt is None:
        return
    mid_idx, keep_low = choose_sagittal_keep_low_x(wt)
    cut_meshes: dict[str, Any] = {}
    for key, mask in tumor_regions.items():
        if mask is None or not np.any(mask):
            continue
        mesh = binary_mask_to_trimesh(
            mask,
            spacing_mm,
            sagittal_keep_low_x=keep_low,
            sagittal_midline=mid_idx,
        )
        if mesh is not None:
            cut_meshes[key] = mesh
    cut_fig = build_labeled_mesh_figure(
        cut_meshes,
        subject_id=case_id,
        view="cutaway",
        cutaway_keep_low_x=keep_low,
        title=f"{case_id} · location cutaway",
    )
    cut_lobe_traces = []
    for lobe, mask in sorted(lobe_masks.items()):
        if lobe == "other" or not np.any(mask):
            continue
        mesh = binary_mask_to_trimesh(
            mask,
            spacing_mm,
            step_size=2,
            sagittal_keep_low_x=keep_low,
            sagittal_midline=mid_idx,
            taubin_iterations=0,
        )
        if mesh is None:
            continue
        cut_lobe_traces.append(
            mesh3d_from_trimesh(
                mesh,
                name=f"{lobe.title()} lobe",
                color=LOBE_COLORS.get(lobe, "#ADB5BD"),
                opacity=1.0,
                showlegend=True,
                flatshading=False,
            )
        )
    if cut_lobe_traces:
        new_cut = go.Figure()
        for tr in cut_lobe_traces:
            new_cut.add_trace(tr)
        for tr in cut_fig.data:
            new_cut.add_trace(tr)
        new_cut.update_layout(cut_fig.layout)
        cut_fig = new_cut
    cut_fig.write_html(
        str(out_dir / "mesh_3d_cutaway.html"), include_plotlyjs="cdn", full_html=True
    )


def run_location_report(
    *,
    out_dir: str | Path,
    t1_path: str | Path,
    regions: Mapping[str, np.ndarray],
    case_id: str,
    affine: np.ndarray | None = None,
    atlas_root: str | Path | None = None,
    skip_bspline: bool = False,
    enabled: bool = True,
    t2_path: str | Path | None = None,
    synthseg_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Opt-in location report. Returns payload or None when disabled.

    Writes ``location.json``, registration transforms / QC under
    ``out_dir/registration/``, injects a Location section into ``report.html``,
    and colours lobes in the Plotly mesh viewer.
    """
    if not enabled:
        return None

    out_dir = Path(out_dir)
    t1_path = Path(t1_path)
    if not t1_path.is_file():
        raise FileNotFoundError(f"T1 not found for location report: {t1_path}")

    # Whole-tumor mask for plausibility gate + cost-function exclusion.
    wt_mask = None
    for key in ("whole_tumor", "WT"):
        if key in regions and regions[key] is not None:
            wt_mask = np.asarray(regions[key], dtype=bool)
            break

    t1_img = nib.load(str(t1_path))
    if affine is None:
        affine = np.asarray(t1_img.affine, dtype=np.float64)
    spacing = _spacing_from_affine(affine)

    gate = assess_location_plausibility(wt_mask, spacing_mm=spacing)
    if gate["gated"]:
        msg = str(gate["message"])
        payload: dict[str, Any] = {
            "case_id": case_id,
            "enabled": True,
            "status": "gated",
            "plausibility_gated": True,
            "plausibility": gate,
            "location_statement": msg,
            "registration": None,
            "regions": {},
            "jhu": "skipped — location gated by volume plausibility",
            "atlas_licenses": {},
            "disclaimer": (
                "No atlas-based location statement: whole-tumor volume is below "
                "the plausibility floor or too fragmented."
            ),
        }
        loc_path = out_dir / "location.json"
        loc_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Location gated for %s: %s", case_id, msg)
        inject_location_into_report(out_dir / "report.html", payload)
        return payload

    bundle = ensure_mni_and_harvard_oxford(
        Path(atlas_root) if atlas_root is not None else None
    )

    reg_dir = out_dir / "registration"
    reg = register_t1_to_mni(
        t1_path,
        bundle.mni_t1,
        reg_dir,
        tumor_mask=wt_mask,
        skip_bspline=skip_bspline,
    )

    # Warp atlases into patient space (nearest neighbour).
    cort_pat = reg_dir / "ho_cortical_patient.nii.gz"
    sub_pat = reg_dir / "ho_subcortical_patient.nii.gz"
    warp_label_to_patient(
        bundle.ho_cortical,
        t1_path,
        reg.transform,
        cort_pat,
        affine_tfm=reg.affine_tfm,
    )
    warp_label_to_patient(
        bundle.ho_subcortical,
        t1_path,
        reg.transform,
        sub_pat,
        affine_tfm=reg.affine_tfm,
    )

    t1_data = np.asanyarray(t1_img.dataobj, dtype=np.float32)
    cortical = np.asanyarray(nib.load(str(cort_pat)).dataobj)
    subcortical = np.asanyarray(nib.load(str(sub_pat)).dataobj)

    # Independent ventricle Dice: HO ventricles in patient space vs patient mask.
    from location.qc import (
        dice_bool,
        estimate_patient_ventricle_mask,
        flag_registration_quality,
    )
    from location.register import estimate_brain_mask

    atlas_vent = _ventricle_mask(subcortical, bundle.subcortical_labels)
    t2_data = None
    if t2_path is not None and Path(t2_path).is_file():
        t2_data = np.asanyarray(nib.load(str(t2_path)).dataobj, dtype=np.float32)
    synth = None
    synth_candidates = []
    if synthseg_path is not None:
        synth_candidates.append(Path(synthseg_path))
    synth_candidates.extend(
        [
            out_dir / "synthseg_anatomy.nii.gz",
            out_dir.parent / "synthseg_anatomy.nii.gz",
        ]
    )
    for sp in synth_candidates:
        if sp.is_file():
            synth = np.asanyarray(nib.load(str(sp)).dataobj)
            break
    patient_brain = estimate_brain_mask(t1_data)
    patient_vent, vent_source = estimate_patient_ventricle_mask(
        t1_data,
        patient_brain,
        t2=t2_data,
        synthseg=synth,
        atlas_ventricle=atlas_vent,
    )
    ventricle_dice = (
        dice_bool(atlas_vent, patient_vent)
        if np.any(atlas_vent) and np.any(patient_vent)
        else 0.0
    )
    quality, note = flag_registration_quality(
        brain_dice=reg.brain_dice,
        ventricle_dice=ventricle_dice,
        surface_p95_mm=reg.surface_p95_mm,
        coverage=reg.coverage,
        stage=reg.stage,
        ventricle_source=vent_source,
    )
    reg.quality = quality
    reg.quality_note = note
    reg.ventricle_dice = float(ventricle_dice)
    reg.ventricle_source = vent_source
    # Persist finalized QC JSON.
    qc_path = reg_dir / "registration_qc.json"
    try:
        meta = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.is_file() else {}
    except json.JSONDecodeError:
        meta = {}
    meta.update(
        {
            "ventricle_dice": float(ventricle_dice),
            "ventricle_source": vent_source,
            "quality": quality,
            "quality_note": note,
            "surface_mean_mm": reg.surface_mean_mm,
            "surface_p95_mm": reg.surface_p95_mm,
            "coverage": reg.coverage,
            "coverage_partial": reg.coverage_partial,
            "stage": reg.stage,
        }
    )
    qc_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _save_qc_overlay(
        t1_data,
        cortical,
        reg_dir / "registration_qc.png",
        title=(
            f"{case_id} · HO cortical on T1 · stage={reg.stage} · "
            f"MI {reg.mi_before:.3f}→{reg.mi_after:.3f} · "
            f"brain={reg.brain_dice:.3f} vent={ventricle_dice:.3f} "
            f"p95={reg.surface_p95_mm:.1f}mm cov={reg.coverage:.2f} ({quality})"
        ),
    )

    ventricle = atlas_vent

    # Optional JHU tracts → patient space.
    cst_mask = None
    optic_mask = None
    jhu_note = "JHU atlas not present — CST / optic-radiation distances skipped."
    try:
        jhu_mni = jhu_tract_mask_mni(bundle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JHU prep failed: %s", exc)
        jhu_mni = {}
    if jhu_mni:
        jhu_note = "JHU tracts warped to patient space."
        mni_aff = nib.load(str(bundle.mni_t1)).affine
        from location.register import warp_array_to_patient

        cst_l = jhu_mni.get("corticospinal_tract_L")
        cst_r = jhu_mni.get("corticospinal_tract_R")
        if cst_l is not None or cst_r is not None:
            cst_mni = np.zeros(nib.load(str(bundle.mni_t1)).shape[:3], dtype=bool)
            if cst_l is not None:
                cst_mni |= cst_l
            if cst_r is not None:
                cst_mni |= cst_r
            cst_mask = warp_array_to_patient(
                cst_mni.astype(np.uint8),
                mni_aff,
                t1_path,
                reg.transform,
                affine_tfm=reg.affine_tfm,
            ).astype(bool)
        opt_mni = np.zeros(nib.load(str(bundle.mni_t1)).shape[:3], dtype=bool)
        has_opt = False
        for k in ("optic_radiation_L", "optic_radiation_R"):
            if k in jhu_mni:
                opt_mni |= jhu_mni[k]
                has_opt = True
        if has_opt:
            optic_mask = warp_array_to_patient(
                opt_mni.astype(np.uint8),
                mni_aff,
                t1_path,
                reg.transform,
                affine_tfm=reg.affine_tfm,
            ).astype(bool)

    region_payload: dict[str, Any] = {}
    for key, short in REGION_KEYS:
        mask = regions.get(key)
        if mask is None:
            mask = regions.get(short)
        if mask is None:
            continue
        block = region_location_block(
            np.asarray(mask, dtype=bool),
            affine=affine,
            cortical=cortical,
            cortical_labels=bundle.cortical_labels,
            subcortical=subcortical,
            subcortical_labels=bundle.subcortical_labels,
            ventricle_mask=ventricle,
            cst_mask=cst_mask,
            optic_mask=optic_mask,
            spacing_mm=spacing,
            registration_quality=reg.quality,
        )
        if (
            reg.stage == "centered_init"
            and block.get("summary")
            and reg.quality
            not in {"poor", "registration unreliable"}
        ):
            summary = str(block["summary"])
            if not summary.lower().startswith("coarse"):
                block["summary"] = f"coarse (translation only): {summary}"
        region_payload[key] = block

    payload = {
        "case_id": case_id,
        "enabled": True,
        "status": "descriptive",
        "plausibility_gated": False,
        "plausibility": gate,
        "location_statement": None,
        "registration": reg.as_dict(),
        "atlas_warp": (
            "affine_inverse (B-spline used for intensity QC; not analytically "
            "invertible in SimpleITK for NN atlas propagation)"
        ),
        "jhu": jhu_note,
        "regions": region_payload,
        "atlas_licenses": license_notes(bundle),
        "disclaimer": (
            "Atlas-based location estimates depend on registration quality. "
            "No clinical claims for cases flagged poor/caution without visual QC."
        ),
    }
    loc_path = out_dir / "location.json"
    loc_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", loc_path)

    inject_location_into_report(out_dir / "report.html", payload)

    if reg.quality != "registration unreliable":
        lobe_masks = _lobe_masks_from_atlas(cortical, bundle.cortical_labels)
        try:
            add_lobes_to_mesh_html(
                out_dir,
                lobe_masks,
                {
                    k: np.asarray(v, dtype=bool)
                    for k, v in regions.items()
                    if v is not None
                },
                case_id=case_id,
                spacing_mm=spacing,
                registration_quality=reg.quality,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lobe mesh colouring failed: %s", exc)
    else:
        logger.info("Skipping lobe meshes — registration unreliable.")

    return payload
