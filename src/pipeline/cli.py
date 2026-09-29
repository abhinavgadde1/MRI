"""CLI: ingest → segment (v3, lcc) → reconstruct → validate → report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import PROJECT_ROOT, load_config
from pipeline.run_case import DEFAULT_CHECKPOINT, run_case


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.cli",
        description=(
            "Run the MRI pipeline on one DICOM study or registered NIfTI folder. "
            "Default checkpoint is v3 (brats_scale_full)."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="End-to-end single-case pipeline")
    run_p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="DICOM study directory, or folder with 4 registered 1mm NIfTIs / "
        "04_registered_1mm/",
    )
    run_p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (e.g. outputs/<case>/)",
    )
    run_p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)})",
    )
    run_p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.yaml (reads inference.checkpoint if set)",
    )
    run_p.add_argument(
        "--postprocess",
        choices=("raw", "lcc", "lcc_min"),
        default="lcc",
    )
    run_p.add_argument("--device", type=str, default=None)
    run_p.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="If checkpoint is missing, initialize a random SegResNet (smoke/demo only)",
    )
    run_p.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore --input and build a synthetic 4-channel volume under --out "
        "(for make demo / fresh-env smoke)",
    )
    run_p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.command != "run":
        parser.error(f"Unknown command {args.command}")

    checkpoint = args.checkpoint
    if checkpoint is None and args.config is not None:
        # Prefer explicit YAML key if present (optional section)
        import yaml

        raw = yaml.safe_load(Path(args.config).read_text()) or {}
        inf = raw.get("inference") or {}
        if inf.get("checkpoint"):
            checkpoint = Path(inf["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = PROJECT_ROOT / checkpoint
    if checkpoint is None:
        try:
            cfg = load_config(args.config)
            inf = getattr(cfg, "inference", None)
            if inf is not None and getattr(inf, "checkpoint", None):
                checkpoint = Path(inf.checkpoint)
        except Exception:  # noqa: BLE001
            pass
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT

    report = run_case(
        input_dir=None if args.synthetic else args.input,
        output_dir=args.out,
        checkpoint=checkpoint,
        postprocess=args.postprocess,
        device=args.device,
        allow_random_weights=args.allow_random_weights or args.synthetic,
        synthetic=args.synthetic,
    )
    print(f"Report → {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
