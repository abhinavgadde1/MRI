#!/bin/zsh
# Tar key BraTS checkpoints with SHA256 into ~/Desktop/mri_backup_<date>/.
# Read-only on checkpoints — does not modify model files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="$(date +%Y%m%d)"
DEST="${HOME}/Desktop/mri_backup_${DATE}"
mkdir -p "${DEST}"

TAR="${DEST}/mri_checkpoints_${DATE}.tar"
HASHES="${DEST}/SHA256SUMS.txt"
PYTHON="${ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON=python3

echo "Creating ${TAR}"
(
  cd "${ROOT}"
  tar -cf "${TAR}" \
    checkpoints/brats_pretrain/best_model.pt \
    checkpoints/brats_v2_loss_fix/best_model.pt \
    checkpoints/brats_scale_full/best_model.pt
)

"${PYTHON}" - "${ROOT}" "${TAR}" "${HASHES}" <<'PY'
"""Print and write SHA256 for the three best_model.pt files + the tar."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
tar_path = Path(sys.argv[2])
hashes_path = Path(sys.argv[3])
rels = [
    "checkpoints/brats_pretrain/best_model.pt",
    "checkpoints/brats_v2_loss_fix/best_model.pt",
    "checkpoints/brats_scale_full/best_model.pt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


lines = ["SHA256 of checkpoint files and tar:"]
print("SHA256:")
for rel in rels:
    path = root / rel
    if not path.is_file():
        line = f"MISSING  {rel}"
    else:
        line = f"{sha256(path)}  {rel}"
    lines.append(line)
    print(line)

tar_line = f"{sha256(tar_path)}  {tar_path.name}"
lines.append(tar_line)
print(tar_line)
hashes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {hashes_path}")
PY

echo "Backup complete → ${DEST}"
