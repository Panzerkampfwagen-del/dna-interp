# dna-interp project tasks
# Override PY to point at a different interpreter, e.g.:
#   make test PY=/home/aryan/anaconda3/envs/dna-interp/bin/python
PY ?= /home/aryan/anaconda3/envs/dna-interp/bin/python
# The single dna-interp env covers real-model runs too; override for a GPU build.
GPU_PY ?= $(PY)

# ── environment ────────────────────────────────────────────────────────────────

env:
	conda env create -f environment.yml
	@echo "Activate with: conda activate dna-interp"

env-update:
	conda env update -f environment.yml --prune

# ── tests (fully offline) ──────────────────────────────────────────────────────

test:
	$(PY) -m pytest -q

test-v:
	$(PY) -m pytest -v

# ── linting ────────────────────────────────────────────────────────────────────

lint:
	$(PY) -m ruff check .

lint-fix:
	$(PY) -m ruff check --fix .

# ── offline synthetic demos ────────────────────────────────────────────────────

# Full end-to-end pipeline on synthetic data (no network, no GPU required)
demo-synthetic:
	$(PY) scripts/run_synthetic_demo.py

# Multi-label TF binding demo
demo-tf:
	$(PY) scripts/run_tf_demo.py

# ── real DNABERT-2 pipeline (needs GPU + network on first run) ────────────────

# Step 1: download DNABERT-2 weights and JASPAR motifs
download:
	$(GPU_PY) scripts/download_assets.py

# Step 2: fine-tune DNABERT-2 on NT enhancers (saves checkpoint to results/)
finetune:
	$(GPU_PY) scripts/finetune_dnabert2_enhancer.py

# Step 3: run mechanistic interpretability on the saved checkpoint
# Usage: make interp CKPT=results/checkpoints/dnabert2_enhancer_best.pt
interp:
	$(GPU_PY) scripts/run_dnabert2_interp.py $(CKPT)

# Step 4: explain test failures via activation patching
patch-failures:
	$(GPU_PY) scripts/run_test_failure_patching.py $(CKPT)

# Render saved .npz arrays to PNG (when matplotlib was absent during the run)
render:
	$(GPU_PY) scripts/render_figures.py

# ── housekeeping ───────────────────────────────────────────────────────────────

clean-results:
	rm -rf results/figures/* results/cache/* results/checkpoints/*

.PHONY: env env-update test test-v lint lint-fix demo-synthetic demo-tf \
        download finetune interp patch-failures render clean-results
