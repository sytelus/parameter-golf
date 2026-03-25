# Experiment Journal

## Index

| # | Name | Key Change | val_bpb (post-quant) | Artifact | Status |
|---|------|-----------|----------------------|----------|--------|
| 000 | baseline_1xA100 | Default train_gpt.py, 20K steps | 1.2195 | 15.87MB | done |

---

## Exp 000: Baseline Reproduction

**Date:** 2026-03-25
**RUN_ID:** baseline_1xA100
**Status:** completed
**Log:** logs/baseline_1xA100.txt
**Report:** docs/baseline_repro.md

### Hypothesis
The default train_gpt.py baseline, trained for 20K iterations on 1xA100, will produce
val_bpb close to the published Naive Baseline (1.2244).

### Changes from baseline
- `MAX_WALLCLOCK_SECONDS=0` (disabled wallclock, use iteration-based warmdown)
- `VAL_LOSS_EVERY=500` (more frequent validation)
- `TRAIN_LOG_EVERY=100`

### Key Results
| Metric | Value |
|--------|-------|
| val_bpb (pre-quant, step 20000) | 1.2110 |
| val_bpb (INT8+zlib roundtrip) | 1.2195 |
| val_loss (INT8+zlib roundtrip) | 2.0591 |
| artifact_size | 15,865,596 bytes |
| step_avg (ms) | 741.37 |
| peak memory | 10,889 MiB |
| quantization penalty | +0.0085 BPB |

### Observations
- Not directly comparable to Naive Baseline due to different dataset export (62M vs 64M val tokens)
- Warmdown phase (steps 18800-20000) contributed -0.023 BPB in just 1200 steps
- Diminishing returns plateau between steps 12K-18K
- INT8+zlib leaves only 134KB headroom under 16MB limit

### Conclusion
Baseline successfully reproduced on 1xA100. This serves as our reference point for all future experiments.
Gap to SOTA (1.1194) is 0.1001 BPB.

### Next Steps
- Try INT6 + zstd compression to free up artifact space
- Try 11 layers and/or MLP_MULT=3
- Try sliding window evaluation
