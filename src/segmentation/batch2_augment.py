"""Domain augmentations for batch2 fine-tuning (clinical → BraTS domain gap).

Applied to both batch2 cases and BraTS replay cases during training:

1. Thick-slice simulation (downsample along z by 3–6×, upsample back to 1 mm)
2. Intensity: random gamma, smooth multiplicative bias-field-like change, Gaussian noise
3. Modality dropout: replace T1c with a copy of T1 (prob 0.1)

Each family is independently switchable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import MapTransform, RandomizableTransform

from preprocessing.h5_to_nifti import MODALITY_NAMES

logger = logging.getLogger(__name__)

# FLAIR=0, T1=1, T1c=2, T2=3
T1_CHANNEL = list(MODALITY_NAMES).index("T1")
T1C_CHANNEL = list(MODALITY_NAMES).index("T1c")


@dataclass(frozen=True)
class DomainAugmentFlags:
    """Which domain-augmentation families are enabled."""

    thick_slice: bool = True
    intensity: bool = True
    modality_dropout: bool = True
    t1c_dropout_prob: float = 0.1
    thick_factor_min: int = 3
    thick_factor_max: int = 6
    gamma_range: tuple[float, float] = (0.7, 1.5)
    bias_magnitude: float = 0.3
    noise_std: float = 0.05
    # Weak-sup shortcut countermeasures (off by default for pixel-supervised FT).
    weaksup_t1c_hide: bool = False
    weaksup_t1c_hide_pos_prob: float = 0.5
    weaksup_t1c_hide_neg_always: bool = True
    inplane_rescale: bool = False
    inplane_scale_range: tuple[float, float] = (0.85, 1.15)
    blur: bool = False
    blur_sigma_range: tuple[float, float] = (0.0, 1.2)
    zscore_brain: bool = False

    def enabled_names(self) -> list[str]:
        names: list[str] = []
        if self.thick_slice:
            names.append(
                f"thick_slice(z_factor={self.thick_factor_min}-{self.thick_factor_max})"
            )
        if self.intensity:
            names.append(
                f"intensity(gamma={self.gamma_range}, bias={self.bias_magnitude}, "
                f"noise_std={self.noise_std})"
            )
        if self.modality_dropout:
            names.append(f"modality_dropout(T1c→T1 p={self.t1c_dropout_prob})")
        if self.weaksup_t1c_hide:
            names.append(
                f"weaksup_t1c_hide(pos_p={self.weaksup_t1c_hide_pos_prob}, "
                f"neg_always={self.weaksup_t1c_hide_neg_always})"
            )
        if self.inplane_rescale:
            names.append(f"inplane_rescale({self.inplane_scale_range})")
        if self.blur:
            names.append(f"blur(sigma={self.blur_sigma_range})")
        if self.zscore_brain:
            names.append("zscore_within_brain_mask")
        return names

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_augment_flags(flags: DomainAugmentFlags) -> None:
    enabled = flags.enabled_names()
    if not enabled:
        logger.info("Domain augmentations: ALL OFF")
    else:
        logger.info("Domain augmentations ON: %s", "; ".join(enabled))


def thick_slice_simulate(
    image: torch.Tensor,
    *,
    factor: int,
    label: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Downsample along z by ``factor``, then upsample back (shape unchanged).

    ``image`` / ``label`` are channel-first ``(C, H, W, D)``.
    Image uses trilinear; label uses nearest.
    """
    if factor < 2:
        return image, label
    if image.ndim != 4:
        raise ValueError(f"Expected (C,H,W,D), got shape {tuple(image.shape)}")

    c, h, w, d = image.shape
    d_low = max(1, int(round(d / float(factor))))
    img_5d = image.unsqueeze(0).float()
    low = F.interpolate(img_5d, size=(h, w, d_low), mode="trilinear", align_corners=False)
    restored = F.interpolate(low, size=(h, w, d), mode="trilinear", align_corners=False)
    out_img = restored.squeeze(0).to(dtype=image.dtype)

    out_lbl = label
    if label is not None:
        if label.ndim != 4:
            raise ValueError(f"Expected label (C,H,W,D), got {tuple(label.shape)}")
        lbl_5d = label.unsqueeze(0).float()
        low_l = F.interpolate(lbl_5d, size=(h, w, d_low), mode="nearest")
        rest_l = F.interpolate(low_l, size=(h, w, d), mode="nearest")
        out_lbl = rest_l.squeeze(0).to(dtype=label.dtype)
    return out_img, out_lbl


def mean_abs_slice_diff(volume: np.ndarray | torch.Tensor, axis: int = -1) -> float:
    """Mean absolute difference between consecutive slices (smoothness proxy)."""
    if isinstance(volume, torch.Tensor):
        arr = volume.detach().float().cpu().numpy()
    else:
        arr = np.asarray(volume, dtype=np.float64)
    # Average over channels if present.
    if arr.ndim == 4:
        arr = arr.mean(axis=0)
    diffs = np.abs(np.diff(arr, axis=axis))
    return float(np.mean(diffs)) if diffs.size else 0.0


def _smooth_bias_field(
    spatial: tuple[int, int, int],
    *,
    magnitude: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Low-frequency multiplicative field via upsampled coarse noise."""
    coarse = tuple(max(2, s // 16) for s in spatial)
    noise = rng.normal(0.0, 1.0, size=coarse).astype(np.float32)
    t = torch.from_numpy(noise).unsqueeze(0).unsqueeze(0)
    field = F.interpolate(t, size=spatial, mode="trilinear", align_corners=False)
    field = field.squeeze(0)  # (1, H, W, D)
    field = 1.0 + magnitude * torch.tanh(field)
    return field


def apply_intensity_augment(
    image: torch.Tensor,
    *,
    gamma_range: tuple[float, float],
    bias_magnitude: float,
    noise_std: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Gamma + bias-field-like + Gaussian noise on channel-first image."""
    out = image.float().clone()
    # Per-channel gamma on positive intensities.
    g_lo, g_hi = gamma_range
    for c in range(out.shape[0]):
        gamma = float(rng.uniform(g_lo, g_hi))
        ch = out[c]
        sign = torch.sign(ch)
        mag = ch.abs().clamp_min(1e-8)
        # Normalize to [0,1]-ish via max for stable gamma.
        peak = float(mag.max().item()) if float(mag.max().item()) > 0 else 1.0
        normed = (mag / peak).clamp(0.0, 1.0)
        out[c] = sign * (normed**gamma) * peak

    bias = _smooth_bias_field(
        tuple(int(s) for s in out.shape[1:]),
        magnitude=bias_magnitude,
        rng=rng,
    )
    out = out * bias
    if noise_std > 0:
        noise = torch.as_tensor(
            rng.normal(0.0, noise_std, size=tuple(out.shape)),
            dtype=out.dtype,
        )
        out = out + noise
    return out.to(dtype=image.dtype)


def apply_t1c_dropout(
    image: torch.Tensor,
    *,
    prob: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """With probability ``prob``, replace T1c channel with T1."""
    if prob <= 0 or rng.random() >= prob:
        return image
    out = image.clone()
    out[T1C_CHANNEL] = out[T1_CHANNEL]
    return out


def apply_inplane_rescale(
    image: torch.Tensor,
    *,
    scale_range: tuple[float, float],
    rng: np.random.Generator,
    label: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Random in-plane (H,W) rescale then crop/pad back to original size."""
    lo, hi = scale_range
    scale = float(rng.uniform(lo, hi))
    if abs(scale - 1.0) < 1e-3:
        return image, label
    c, h, w, d = image.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img5 = image.unsqueeze(0).float()
    scaled = F.interpolate(img5, size=(nh, nw, d), mode="trilinear", align_corners=False)
    out = _center_crop_or_pad(scaled.squeeze(0), (h, w, d)).to(dtype=image.dtype)
    out_lbl = label
    if label is not None:
        lbl5 = label.unsqueeze(0).float()
        sl = F.interpolate(lbl5, size=(nh, nw, d), mode="nearest")
        out_lbl = _center_crop_or_pad(sl.squeeze(0), (h, w, d)).to(dtype=label.dtype)
    return out, out_lbl


def _center_crop_or_pad(
    vol: torch.Tensor, target: tuple[int, int, int]
) -> torch.Tensor:
    """Center crop or pad a (C,H,W,D) volume to ``target`` spatial size."""
    c, h, w, d = vol.shape
    th, tw, td = target
    # Pad first if smaller
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    pad_d = max(0, td - d)
    if pad_h or pad_w or pad_d:
        # F.pad order: (d_before, d_after, w_b, w_a, h_b, h_a)
        vol = F.pad(
            vol,
            (
                pad_d // 2,
                pad_d - pad_d // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )
        c, h, w, d = vol.shape
    sh = max(0, (h - th) // 2)
    sw = max(0, (w - tw) // 2)
    sd = max(0, (d - td) // 2)
    return vol[:, sh : sh + th, sw : sw + tw, sd : sd + td]


def apply_blur(
    image: torch.Tensor,
    *,
    sigma_range: tuple[float, float],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Light Gaussian blur via small-kernel smoothing (approx)."""
    lo, hi = sigma_range
    sigma = float(rng.uniform(lo, hi))
    if sigma < 1e-3:
        return image
    # 3-tap separable-ish blur by average pooling upsample (cheap approx).
    img5 = image.unsqueeze(0).float()
    # Depthwise channel blur using average pool + upsample
    pooled = F.avg_pool3d(img5, kernel_size=3, stride=1, padding=1)
    mix = min(1.0, sigma / 1.2)
    out = (1.0 - mix) * img5 + mix * pooled
    return out.squeeze(0).to(dtype=image.dtype)


def apply_zscore_brain(image: torch.Tensor) -> torch.Tensor:
    """Per-channel z-score within a brain mask (any channel > 0)."""
    out = image.float().clone()
    brain = (out.abs().sum(dim=0) > 0)
    if not bool(brain.any()):
        return image
    for c in range(out.shape[0]):
        ch = out[c]
        vals = ch[brain]
        mean = vals.mean()
        std = vals.std().clamp_min(1e-6)
        out[c] = torch.where(brain, (ch - mean) / std, ch)
    return out.to(dtype=image.dtype)


class DomainAugmentd(RandomizableTransform, MapTransform):
    """MONAI dict transform applying domain augmentations to ``image`` (+ ``label``)."""

    def __init__(
        self,
        keys: Sequence[Hashable] | Hashable = ("image",),
        *,
        label_key: str = "label",
        flags: DomainAugmentFlags | None = None,
        allow_missing_keys: bool = False,
        is_negative_key: str = "is_negative",
    ) -> None:
        keys_t = (keys,) if isinstance(keys, (str, bytes)) else tuple(keys)
        MapTransform.__init__(self, keys_t, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1.0)
        self.label_key = label_key
        self.is_negative_key = is_negative_key
        self.flags = flags or DomainAugmentFlags()
        self._np_rng = np.random.default_rng(0)

    def randomize(self, data: Any = None) -> None:
        seed = int(self.R.randint(0, 2**31 - 1))
        self._np_rng = np.random.default_rng(seed)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        self.randomize()
        flags = self.flags
        is_neg_raw = d.get(self.is_negative_key, False)
        if isinstance(is_neg_raw, torch.Tensor):
            is_neg = bool(is_neg_raw.reshape(-1)[0].item())
        else:
            is_neg = bool(is_neg_raw)

        for key in self.key_iterator(d):
            image = d[key]
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)
            label = d.get(self.label_key)
            if label is not None and not isinstance(label, torch.Tensor):
                label = torch.as_tensor(label)

            if flags.thick_slice:
                factor = int(
                    self._np_rng.integers(flags.thick_factor_min, flags.thick_factor_max + 1)
                )
                image, label = thick_slice_simulate(image, factor=factor, label=label)

            if flags.inplane_rescale:
                image, label = apply_inplane_rescale(
                    image,
                    scale_range=flags.inplane_scale_range,
                    rng=self._np_rng,
                    label=label,
                )

            if flags.blur:
                image = apply_blur(
                    image, sigma_range=flags.blur_sigma_range, rng=self._np_rng
                )

            if flags.intensity:
                image = apply_intensity_augment(
                    image,
                    gamma_range=flags.gamma_range,
                    bias_magnitude=flags.bias_magnitude,
                    noise_std=flags.noise_std,
                    rng=self._np_rng,
                )

            if flags.zscore_brain:
                image = apply_zscore_brain(image)

            if flags.weaksup_t1c_hide:
                if is_neg and flags.weaksup_t1c_hide_neg_always:
                    image = apply_t1c_dropout(image, prob=1.0, rng=self._np_rng)
                elif not is_neg:
                    image = apply_t1c_dropout(
                        image,
                        prob=float(flags.weaksup_t1c_hide_pos_prob),
                        rng=self._np_rng,
                    )
            elif flags.modality_dropout:
                image = apply_t1c_dropout(
                    image, prob=flags.t1c_dropout_prob, rng=self._np_rng
                )

            d[key] = image
            if label is not None and self.label_key in d:
                d[self.label_key] = label
        return d
