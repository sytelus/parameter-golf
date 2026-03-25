# CLAUDE.md

## Project Overview

This is a fork of OpenAI's **Parameter Golf** challenge. The goal is to train the best
language model that fits in a 16MB artifact, evaluated by bits-per-byte (BPB) on the
FineWeb validation set.

## Key Constraints

- **Artifact limit:** 16,000,000 bytes (code + compressed model, decimal not MiB)
- **Metric:** val_bpb (lower is better), measured after INT8+zlib (or INT6+zstd, etc.) roundtrip
- **Hardware:** We run on 1x NVIDIA A100 80GB PCIe (~741ms/step, ~4h for 20K steps)
- **Dataset:** fineweb10B_sp1024, 80 shards, stored at /bigdata/data/datasets/fineweb10B_sp1024/
  (symlinked from ./data/datasets/fineweb10B_sp1024/)

## Important Files

- `train_gpt.py` — Main training script. All hyperparameters configurable via env vars.
- `docs/experiment_guide.md` — **Read this before running any experiment.** Covers hypothesis
  formation, experiment design, naming conventions, journaling, and reflection process.
- `docs/experiment_journal.md` — Running log of all experiments with results. Check this
  before starting a new experiment to avoid repeats.
- `docs/baseline_repro.md` — Baseline reproduction report with detailed analysis.
- `records/` — Leaderboard submissions with READMEs, train logs, and code. Study these for ideas.

## Running Experiments

```bash
# Quick signal check (~25 min)
RUN_ID=exp_NNN_name MAX_WALLCLOCK_SECONDS=0 ITERATIONS=2000 VAL_LOSS_EVERY=500 python3 train_gpt.py

# Full run (~4 hours)
RUN_ID=exp_NNN_name MAX_WALLCLOCK_SECONDS=0 ITERATIONS=20000 VAL_LOSS_EVERY=500 python3 train_gpt.py
```

Always run a 2K-step signal check before a full 20K-step run.

## Current State

- **Our baseline:** val_bpb = 1.2195 (INT8+zlib), 15.87MB artifact
- **SOTA:** val_bpb = 1.1194 (gap: 0.1001 BPB)
- **Next priorities:** See experiment_journal.md for planned experiments

## Workflow Rules

1. Follow the experiment guide: hypothesis first, then design, then run.
2. Log every experiment in the journal — even failures are valuable.
3. Change one variable at a time unless there's strong reason to combine.
4. Compare results at the same step count using the baseline reference table.
5. Check artifact size — exceeding 16MB is a hard failure.
