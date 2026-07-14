# dna-interp

[![ci](https://github.com/Panzerkampfwagen-del/dna-interp/actions/workflows/ci.yml/badge.svg)](https://github.com/Panzerkampfwagen-del/dna-interp/actions/workflows/ci.yml)

Mechanistic interpretability of a DNA language model (DNABERT-2) on the enhancer
classification task. The goal is not benchmark accuracy but understanding what
the model learns: attention-head specialization, causal activation patching,
layer-wise probing, and correlation with JASPAR transcription-factor motifs.

**Demonstrates:** activation patching, layer-wise probing, attention-head
specialization, JASPAR motif correlation, DNABERT-2 hook engineering. Bridges to
the grokking interpretability toolkit (`../grokking`) via `grokking_bridge.py`
(shared hook API, activation-patching logic, and CKA).

## Key finding

Fine-tuned DNABERT-2 (117M) reaches **AUROC 0.82 / MCC 0.55** on the held-out
enhancer test set, but in-distribution validation climbs to **MCC 0.89** over five
epochs while test stays flat — a train/test generalization gap that further
training does not close. Mechanistically the enhancer signal is largely
**compositional**: GC content is linearly decodable at **R²=0.99 from the
embedding layer**, with CpG (acc 0.95), TATA box (acc 0.97), and discriminative
k-mers (acc 0.87) all readable from the first few layers. DNABERT-2's attention is
**already sharp before fine-tuning** (a pretrained-backbone control gives 123/144
heads at entropy < 0.8, dropping to 106/144 *after* fine-tuning — the task does
not create specialization, it slightly reduces it, suggesting some heads
de-specialize under task pressure). Against a shuffled-motif noise floor, only a
few heads track real TF motifs — most clearly **L9H8 ↔ RFX1 (r=0.30 vs shuffled
max 0.13)**, with weaker E2F7 and SOX21 heads. Activation patching on GC-matched
test failures (main − null) localizes the generalization gap to **L0–L1**,
consistent with a distributional mismatch in early composition features rather
than missing TF-motif detectors — matching the probing result that composition is
decodable from L0.

## Results: real DNABERT-2 on real enhancer data

The intended benchmark, BEND, is no longer reachable (its repository is gone and
it is not on PyPI). The accessible same-task substitute is the Nucleotide
Transformer downstream `enhancers` task: real 200 bp sequences, binary enhancer
label, balanced 14968 train / 400 test (`data/bend_loader.py:load_nt_enhancer`).

Fine-tune (`scripts/finetune_dnabert2_enhancer.py`, fp32 weights + BF16 autocast,
`max_length=48` since 200 bp ≈ 42 BPE tokens, fits the 6 GB card at batch 32):

| epoch | val MCC | val AUROC | held-out **test** |
|-------|---------|-----------|-------------------|
| 0 | 0.81 | 0.97 | — |
| 1 | 0.86 | 0.98 | — |
| 4 | **0.89** | 0.98 | **MCC 0.55 / AUROC 0.82 / acc 0.78** |

Validation improves every epoch; test is unchanged from the one-epoch model.
The extra epochs buy in-distribution memorization, not generalization.

Interpretability of the fine-tuned checkpoint (`scripts/run_dnabert2_interp.py
<checkpoint>`, real enhancer sequences):

- **Probing** — every discriminative feature is linearly decodable from the
  *earliest* layers and does not deepen: GC R²=0.99 @ L0, CpG 0.95 @ L1, TATA
  0.97 @ L3, k-mer 0.87 @ L2. The signal is sequence composition.
  *(These are the leakage-fixed real-model values: the CpG/k-mer median thresholds
  and discriminative-k-mer selection are fit on the probe train slice only, not the
  full probe set. Regenerated on the DNABERT-2 checkpoint with that fix in place —
  every accuracy moved by ≤0.004, i.e. unchanged at this precision.)*
- **Attention** — 106/144 heads specialized (entropy < 0.8). The pretrained
  backbone control on the same sequences gives 123/144, so fine-tuning does not
  create specialized heads; DNABERT-2 is already sharp.
- **Motif correlation** — 879 real JASPAR 2024 motifs vs 40 column-shuffled
  controls as a noise floor (the scanner is vectorized, ~23× faster than the
  per-window loop, numerically identical):

  | head | top real motif | r | shuffled max | verdict |
  |------|----------------|------|--------------|---------|
  | L9H8 | RFX1 | +0.30 | +0.13 | above noise |
  | L8H9 | E2F7 | +0.14 | +0.05 | above noise |
  | L1H0 | SOX21 | +0.12 | +0.03 | above noise |
  | L1H8 | Rarg | +0.11 | +0.17 | within noise |
  | L0H4 | — | ≈0 | — | null |

  Real but sparse per-head motif tracking, concentrated in a couple of heads —
  not a clean distributed grammar. (With crude consensus motifs and no noise
  floor an earlier pass found ~nothing; the real PWMs plus the control are what
  surface L9H8↔RFX1.) Caveats: best-of-879 against only 40 shuffled controls is
  an estimate of the floor; BPE makes motif positions approximate; this is one
  attention metric.

## Test-failure patching: why the model misclassifies

To explain the train/test gap mechanistically, activation patching transfers the
residual stream from confident true-positives into GC-matched confident
false-negatives and localizes the recovered enhancer logit
([`interp/test_failure_patching.py`](interp/test_failure_patching.py),
[`scripts/run_test_failure_patching.py`](scripts/run_test_failure_patching.py)).
GC-matching is essential because composition is the dominant probe signal; without
it, patching would just restore GC content.

This required making patching work on DNABERT-2 for the first time: its layers
unpad tokens internally, so a packed-aware hook remaps padded positions to packed
tokens via each layer's `indices` tensor (self-patch Δlogit = 3e-6 — machinery
validated). Result over 43 confident FN←TP pairs at confidence threshold 0.3 (the
test set is small; raising to 0.4 adds pairs with noisier logits — results are
consistent):

- **The null control is not clean.** Patching *any* GC-matched donor raises the
  FN logit (mean |Δ| 0.09, max +0.80), so much of the raw effect is non-specific:
  FN residuals are simply low and almost any in-distribution activation helps. The
  enhancer-specific effect is therefore `main − null`, not raw `main`.
- **The enhancer-specific signal is early-layer.** main − null peaks at +0.63 in
  **L0–L1** (token positions 3, 30, 12); late layers L8–L11 contribute little, and
  the top causal sites do **not** overlap RFX1 or other JASPAR motifs.
- **Outcome B (early-layer composition).** Even GC-matched failures differ in the
  low-level features the model reads in its first layers — not in the late-layer
  TF-motif heads. This matches the probing result (composition decodable from L0):
  the generalization gap is a distributional mismatch in early features, not a
  missing motif detector. Outputs in `results/test_failure_patching/`
  (`delta_specific_heatmap.png` is the key figure).

## Setup

One-step environment creation (requires [conda](https://docs.conda.io)):

```bash
conda env create -f environment.yml   # creates the 'dna-interp' env
conda activate dna-interp
```

For GPU training swap the `torch` index URL in `environment.yml` to the matching
CUDA build (e.g. `https://download.pytorch.org/whl/cu124` for CUDA 12.4).

Common tasks via `make`:

```bash
make test             # run the full test suite (CPU, fully offline)
make demo-synthetic   # end-to-end synthetic pipeline (no network needed)
make demo-tf          # TF-binding multi-label demo
make finetune         # fine-tune real DNABERT-2 (needs GPU + network)
make interp CKPT=results/checkpoints/dnabert2_enhancer_best.pt
make patch-failures CKPT=results/checkpoints/dnabert2_enhancer_best.pt
```

## Environment

A single `dna-interp` conda environment (Python 3.11, `environment.yml`) covers
tests, synthetic demos, and real-model runs. No scikit-learn / scipy dependency:
metrics and probes are implemented in numpy/torch from scratch (`metrics.py`,
`models/probes.py`). matplotlib is an optional import in `visualize/` — when
absent only `.npz` arrays are written; render them later with
`scripts/render_figures.py`.

Tests run fully offline (39 tests, CPU, untrained tiny BERT):

```bash
pytest -q   # or: make test
```

## Two modes

The code runs in two modes so the interpretability machinery can be developed and
validated without network access.

1. **Real mode** — DNABERT-2 (`zhihan1996/DNABERT-2-117M`, `trust_remote_code`),
   the NT `enhancers` task (`load_nt_enhancer`), and JASPAR 2024 CORE vertebrates
   (`data/jaspar.py:download_jaspar`). Fine-tune with
   `scripts/finetune_dnabert2_enhancer.py`, then `scripts/run_dnabert2_interp.py
   <checkpoint>`.

2. **Synthetic mode** (fully offline, no download) — `make_synthetic_enhancer_dataset`
   builds enhancer-like sequences with *planted* biology (GATA / TATA / SP1 motifs
   and a GC bias in the positive class) and trains a tiny local BERT. The same
   attention, patching, probing, and motif-scan code then confirms the tools
   recover the planted signal — this is what validates the machinery itself.

   ```bash
   make demo-synthetic   # or: python scripts/run_synthetic_demo.py
   ```

### Synthetic validation (tooling check, not a benchmark claim)

`scripts/run_synthetic_demo.py` (6-layer local BERT, 3k planted-motif sequences)
recovers the planted signal end to end, which is how we trust the same code on
DNABERT-2:

- Stage 1: test MCC / AUROC = 1.0 (synthetic data is highly separable).
- Stage 3 (patching): causal effect concentrates in early layers (L0–L2) and the
  top causal token positions fall on planted-motif sites.
- Stage 4 (probing): GC content is linearly decodable (R² ≈ 1.0) at every layer;
  the centered-TATA probe rises with depth (0.88 → 0.93).
- Stage 5 (JASPAR): every analyzed head correlates most with the GC-rich SP1
  motif (r = 0.31–0.45) and stays near zero on the DECOY control (r ≈ 0).

This synthetic model solves the task with surface statistics (GC / GC-rich
motifs), and the real DNABERT-2 result above shows the same theme — composition
dominates — with the addition of a few genuine motif-tracking heads.

### Task 2: TF binding (multi-label)

`make_synthetic_tf_binding_dataset` plants each TF's motif independently and
labels which are present; the trainer uses `BCEWithLogitsLoss` and evaluation
reports mean AUROC (`task="tf_binding"`). The interp tools carry over: patching
takes a `target_class` so a single TF's logit can be localized. The stub k-mer
tokenizer can split a motif across token boundaries, so synthetic Task-2 AUROC is
only modestly above chance; DNABERT-2's BPE tokenizer does not have this limit.

`scripts/run_tf_demo.py` trains a 4-TF multi-label model (test mean AUROC ≈ 0.74)
and, for each TF, runs activation patching targeting that TF's logit. The causal
token positions land on that TF's planted motif (SP1 3/3, EBOX 3/3, TATA 1/1,
GATA 1/2), confirming `target_class` patching does per-TF causal attribution.

BF16 + gradient checkpointing (`DNAClassifier.enable_efficiency`) casts the whole
module so head and backbone share a dtype; the trainer upcasts logits for the
loss. The five analysis stages live in `interp/pipeline.py:run_all_interp`, so the
same stack runs on the synthetic model or a real DNABERT-2 checkpoint in one call.

## DNABERT-2 engineering notes

- Its bundled Triton flash-attention breaks on current Triton
  (`tl.dot(..., trans_b=...)` was removed) and never exposes attention weights.
  `models/dnabert2_patch.py:enable_eager_attention()` sets the module's
  `flash_attn_qkvpacked_func` to None, triggering DNABERT-2's own PyTorch
  attention path (CPU/GPU, explicit softmax).
- Its custom `BertModel` returns a tuple and unpads tokens internally, so it does
  not support `output_hidden_states` / `output_attentions`. `DNAClassifier`
  detects remote-code models and captures the residual stream and attention with
  hooks (`ResidualCapture`, `AttentionCapture`), re-padding via the `indices` each
  layer receives. Cache shapes: resid `[13, B, S, 768]`, attention
  `[12, B, 12, S, S]` with rows summing to 1. Needs `einops`.

## Layout

```
data/        enhancer loading + tokenization, synthetic generators, JASPAR PWMs
models/      DNAClassifier wrapper with activation cache, linear probes
finetune/    training loop, MCC/AUROC evaluation
interp/      attention analysis, activation patching, probing, motif scan, pipeline
visualize/   heatmap plots for each analysis
scripts/     runnable entry points (fine-tune, real interp, synthetic + TF demos)
tests/       pytest suite that passes fully offline
results/     checkpoints, figures, cached arrays (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
