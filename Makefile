# MRI pipeline — common targets
.PHONY: paper test demo patients-demo setup serve synthetic-case

# Prefer the project venv so extras such as fast-simplification (trimesh 5
# quadric decimation) are importable. Override with `make PYTHON=python3 test`.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := src

setup:
	./setup.sh

paper:
	$(PYTHON) scripts/make_paper_assets.py

test:
	$(PYTHON) -m pytest tests/ -q

# Synthetic single-case smoke (no BraTS/DICOM; random weights if checkpoint absent)
demo:
	$(PYTHON) -m pipeline.cli run --synthetic --allow-random-weights \
		--input /tmp/unused --out outputs/demo_synthetic/

# Build ZIP used by the web UI / API smoke path
synthetic-case:
	$(PYTHON) scripts/make_synthetic_case.py --out outputs/synthetic_case

# Qualitative real-patient demo (requires local data + v3 checkpoint)
patients-demo:
	$(PYTHON) scripts/real_patients_demo.py

# Local web UI + API (http://127.0.0.1:8000). Job state is in-memory.
# Package layout is src/-rooted, so the app module is api.app (not src.api.app).
serve:
	PYTHONPATH=src $(PYTHON) -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
