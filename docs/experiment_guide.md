# Experiment Guide for Parameter Golf

This document defines how we run, track, and learn from experiments in this project.
Every experiment should be traceable from hypothesis to result.

---

## 1. Hypothesis Formation

Before running anything, write down:

1. **What do you believe?** — A specific, falsifiable claim.
   - Good: "Increasing MLP_MULT from 2 to 3 with INT6 quantization will improve val_bpb by ~0.01 while staying under 16MB"
   - Bad: "Let's try a bigger MLP and see what happens"

2. **Why do you believe it?** — Evidence or reasoning.
   - "Top leaderboard entries all use MLP_MULT=3 with INT6. The extra capacity pays for itself because INT6 compresses ~33% better than INT8."

3. **What would disprove it?** — Define failure criteria upfront.
   - "If val_bpb improves by less than 0.005, or artifact exceeds 16MB, the hypothesis is wrong."

4. **What's the expected effect size?** — Prevents wasting time on noise.
   - Use the leaderboard and our baseline progression table as anchors.

### Where to look for ideas

- **Leaderboard diffs**: What changed between consecutive SOTA entries?
- **records/ READMEs**: Authors explain their reasoning. Read them.
- **Diminishing returns table** (see baseline_repro.md): Where are the biggest gaps?
- **Quantization headroom**: Current INT8+zlib uses 15.87MB. INT6+zstd could save ~0.3MB = room for more params.
- **Evaluation tricks**: Sliding window eval, test-time training — these improve score without changing training.
- **Architecture**: More layers, wider MLP, different attention patterns.
- **Training schedule**: Warmdown length, learning rates, weight decay, EMA/SWA.

---

## 2. Experiment Design

### Naming Convention

Every experiment gets a unique `RUN_ID` in this format:

```
exp_<NNN>_<short_description>
```

Examples: `exp_001_mlp3x_int6`, `exp_002_11layers`, `exp_003_sliding_eval`

### Environment Variables Template

Copy this block and modify for each experiment. This is the complete set of tunables:

```bash
# === EXPERIMENT: exp_NNN_description ===
# HYPOTHESIS: <one line>
# EXPECTED: <val_bpb target, artifact size estimate>

RUN_ID=exp_NNN_description \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=20000 \
WARMDOWN_ITERS=1200 \
WARMUP_STEPS=20 \
SEED=1337 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=500 \
VAL_BATCH_SIZE=524288 \
TRAIN_LOG_EVERY=100 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
VOCAB_SIZE=1024 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
MUON_MOMENTUM=0.95 \
MUON_BACKEND_STEPS=5 \
LOGIT_SOFTCAP=30.0 \
ROPE_BASE=10000.0 \
QK_GAIN_INIT=1.5 \
python3 train_gpt.py
```

### Controlling Experiment Duration

Full 20K-step runs take ~4 hours on 1xA100. Use shorter runs for fast iteration:

| Purpose | ITERATIONS | VAL_LOSS_EVERY | Approx Time |
|---------|-----------|----------------|-------------|
| Smoke test (does it crash?) | 100 | 50 | ~2 min |
| Quick signal (is the direction right?) | 2000 | 500 | ~25 min |
| Medium run (compare to baseline) | 5000 | 1000 | ~1 hour |
| Full run (final result) | 20000 | 500 | ~4 hours |

**Rule**: Never run a full 20K-step experiment before a 2K-step signal check confirms the direction.

### Controlling for Confounds

- **Change one thing at a time.** If you change MLP_MULT and NUM_LAYERS simultaneously,
  you won't know which helped.
- **Use the same seed** (1337) unless testing seed sensitivity.
- **Compare at the same step count**, not the same wall time.
- **Watch artifact size** throughout, not just at the end.

---

## 3. Experiment Journal

All experiments are logged in `docs/experiment_journal.md`. Each entry follows this template:

```markdown
## Exp NNN: Short Title

**Date:** YYYY-MM-DD
**RUN_ID:** exp_NNN_description
**Status:** running | completed | abandoned
**Log:** logs/exp_NNN_description.txt

### Hypothesis
<What and why>

### Changes from baseline
<Exact env var overrides or code changes, as a diff or list>

### Key Results
| Metric | Baseline | This Exp | Delta |
|--------|----------|----------|-------|
| val_bpb (step 2000) | 1.3250 | ? | ? |
| val_bpb (step 5000) | 1.2758 | ? | ? |
| val_bpb (final) | 1.2195* | ? | ? |
| artifact_size | 15,865,596 | ? | ? |
| step_avg (ms) | 741 | ? | ? |

*post INT8+zlib roundtrip

### Observations
<What happened? Anything surprising?>

### Conclusion
<Was the hypothesis confirmed or rejected? What did we learn?>

### Next Steps
<What experiment does this suggest?>
```

### Reference Baseline Numbers (Current Dataset Export)

These are from our 1xA100 20K-step run. Use these for comparison, NOT the original
Naive Baseline numbers (which used an older dataset export with different val tokens).

| Step | val_bpb | val_loss |
|------|---------|----------|
| 0 | 4.1077 | 6.9357 |
| 500 | 1.4795 | 2.4981 |
| 1000 | 1.3834 | 2.3359 |
| 2000 | 1.3250 | 2.2373 |
| 5000 | 1.2758 | 2.1541 |
| 10000 | 1.2516 | 2.1132 |
| 15000 | 1.2376 | 2.0896 |
| 20000 | 1.2110 | 2.0448 |
| **Final (INT8+zlib)** | **1.2195** | **2.0591** |

---

## 4. What to Observe and Log

### During Training (automated in train.log)

- `train_loss` every 100 steps — is training stable?
- `val_loss` and `val_bpb` every 500 steps — is the model improving?
- `step_avg` — did your change make steps slower/faster?
- `train_time` — total elapsed time

### After Training (check manually)

- `final_int8_zlib_roundtrip val_bpb` — **this is the official metric**
- `Serialized model int8+zlib: N bytes` — must be < 16,000,000 with code
- `Code size: N bytes` — included in the 16MB budget
- `Total submission size int8+zlib: N bytes` — the number that matters
- `peak memory allocated` — are we GPU-memory limited?
- Quantization penalty: pre-quant val_bpb vs post-quant val_bpb

### Red Flags to Watch For

- **train_loss spikes**: Learning rate too high, or instability from architecture change
- **val_bpb stops improving but train_loss keeps dropping**: Overfitting
- **step_avg increased significantly**: Architecture change is too expensive
- **Artifact > 16MB**: Need better quantization or fewer parameters
- **Large quantization penalty** (> 0.01 BPB): Model has outlier weights, consider QAT or clipping

---

## 5. Avoiding Repeated Experiments

### Before starting any experiment:

1. **Check the journal** (`docs/experiment_journal.md`) — has this been tried?
2. **Check the leaderboard records** (`records/track_10min_16mb/`) — has someone else tried it?
3. **Check git log** — was this attempted and reverted?

### What counts as a "repeat":

- Same architecture + same hyperparameters = definitely a repeat
- Same idea but different hyperparameters = NOT a repeat (it's a sweep)
- Same architecture but different quantization = NOT a repeat

### Experiment Index

Maintain a quick-reference table at the top of `docs/experiment_journal.md`:

```markdown
| # | Name | Key Change | val_bpb | Artifact | Status |
|---|------|-----------|---------|----------|--------|
| 000 | baseline_1xA100 | - | 1.2195 | 15.87MB | done |
| 001 | mlp3x_int6 | MLP_MULT=3, INT6 | ? | ? | running |
```

---

## 6. Reflection and Next-Hypothesis Formation

After every experiment (or batch of experiments), write a reflection:

### Questions to answer:

1. **What surprised me?** — Surprises are the most valuable signal.
2. **Where is the biggest remaining gap?** — Compare to SOTA (currently 1.1194 BPB).
3. **What's the limiting factor right now?** — Artifact size? Training time? Architecture?
4. **What's the lowest-hanging fruit?** — Prioritize impact-per-effort.
5. **What combination of proven techniques haven't I tried?** — The leaderboard is built on stacking wins.

### Decision Framework for What to Try Next

```
Is artifact size < 15.5MB?
  YES → Can fit more parameters. Try more layers or wider MLP.
  NO  → Need better compression. Try INT6, zstd, or GPTQ-lite.

Is val_bpb still improving at the end of training?
  YES → Train longer (more iterations or better warmdown schedule).
  NO  → The model has converged. Need architectural changes.

Is quantization penalty > 0.01 BPB?
  YES → Try QAT (quantization-aware training) or GPTQ-lite.
  NO  → Quantization is not the bottleneck.

Is step time > 800ms?
  YES → Architecture too expensive for 1xA100. Simplify or optimize.
  NO  → Can afford more computation per step.
```

### Proven Techniques from Leaderboard (by impact)

These are ordered roughly by how much they contributed across submissions:

1. **INT6 + zstd-22 compression** (~0.3MB savings → room for bigger model)
2. **11 layers instead of 9** (more depth = better quality)
3. **3x MLP expansion** (more capacity per layer)
4. **Sliding window evaluation** (stride=64, ~0.02 BPB free improvement)
5. **EMA/SWA weight averaging** (stabilizes final weights)
6. **Muon weight decay 0.04** (improves generalization)
7. **Longer warmdown** (3000-3500 steps vs 1200)
8. **FP16 tied embeddings** (reduces quantization penalty on embeddings)
9. **Longer sequence length** (2048 for train+eval)
10. **Test-time training** (TTT on already-evaluated tokens)
11. **GPTQ-lite / QAT** (INT6 with minimal quality loss)
12. **Partial RoPE, XSA, SmearGate** (advanced attention mechanisms)

---

## 7. File Organization

```
parameter-golf/
├── train_gpt.py              # Main training script (modify for experiments)
├── logs/                      # Training logs (auto-created, gitignored)
│   ├── baseline_1xA100.txt
│   └── exp_NNN_description.txt
├── docs/
│   ├── baseline_repro.md      # Baseline reproduction report
│   ├── experiment_guide.md    # This file
│   └── experiment_journal.md  # Running log of all experiments
├── final_model.pt             # Latest raw model (gitignored)
├── final_model.int8.ptz       # Latest compressed model (gitignored)
└── data/
    ├── datasets/              # Symlink to /bigdata/data/datasets/
    └── tokenizers/            # Tokenizer files
```

---

## 8. Quick Reference

**Our baseline:** val_bpb = 1.2195 (INT8+zlib), artifact = 15.87MB
**Current SOTA:** val_bpb = 1.1194 (LeakyReLU² + TTT + Parallel Muon)
**Gap to close:** 0.1001 BPB
**Artifact budget remaining:** 134 KB (with INT8+zlib)
**16MB limit:** 16,000,000 bytes (decimal, not MiB)
**Hardware:** 1x NVIDIA A100 80GB PCIe, ~741ms/step at baseline config
**Dataset:** fineweb10B_sp1024, 80 shards (8B tokens), val = 62,021,632 tokens
