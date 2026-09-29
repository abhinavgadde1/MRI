# MRI pipeline — common targets
.PHONY: paper test demo patients-demo setup

PYTHON ?= python3
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

# Qualitative real-patient demo (requires local data + v3 checkpoint)
patients-demo:
	$(PYTHON) scripts/real_patients_demo.py
