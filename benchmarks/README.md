# benchmarks/ — device and speed results

`results/` holds one JSON file per benchmark run, named `<UTC time>_<device>.json`. They are
written by

```bash
.venv/Scripts/python.exe -m possession.video.bench          # Windows
.venv/bin/python -m possession.video.bench                  # Linux / WSL
```

The benchmark always reads the **tactical** video, `data/tactical/BAR-ATH_tactical.mp4`. That is
the reference timeline (DATA.md); the broadcast video is never used. It times:
- sequential decoding;
- one training step of the ball detector (96 px crops, batch 128);
- one inference batch at crop sizes 96, 160 and 320.

It runs on every available device, and on ROCm with MIOpen both on and off.

`scripts/run_all.sh` also writes `<UTC time>_run-all.json` here (`possession/run_record.py`): the time
and status of every step of the round, software and hardware versions, git state and the aggregate
scores of each prediction on the public window. Nothing frame-level.

## What may live here (hackathon rules)

| allowed in `results/` | never here |
|---|---|
| timings, throughput, software and hardware versions, git commit | submission CSVs: they go in `submissions/` or `evaluation/results/`, both gitignored |
| aggregate scores (spellF1, macroF1 on the public window), if added later | crops, frames, ball positions, detections, features, model weights |

RULES.md forbids publishing derived data that would reconstruct the match ("frame-level tracking
exports, clipped video, and similar"). The JSON written by `bench.py` contains none of that, so it
can be committed. `.gitignore` lets only `*.json` through in `results/`, so a stray CSV or array
cannot be committed by accident. Don't paste the output into a hosted AI assistant anyway: the
T&C forbid giving it extracts of the data, and keeping that habit costs nothing.

## Findings so far

Measured on an AMD Radeon RX 9060 XT (gfx1200, 16 GB), Windows 11, torch 2.14.0+rocm10.1.0, using
**random tensors**, not the video. Run `bench.py` to get the numbers on real crops.

| setting | train step, 96 px × 128 | inference, 320 px × 32 |
|---|---|---|
| CPU | 505 ms | — |
| GPU, MIOpen on | fails: the batch-norm kernel does not compile (`'type_traits' file not found`) | — |
| GPU, MIOpen for convolutions only | 278 ms | 166 ms |
| **GPU, MIOpen off (the default)** | **179 ms** | **33 ms** |

So `possession.video.model.pick_device` switches MIOpen off on ROCm. Re-test with `miopen=True`
after a ROCm or driver update.

**Whole match on the real video** (RX 9060 XT, `run_all`): the detector self-check took 4 minutes;
the detection pass over all 190,198 frames took 45 and 46 minutes in two runs (about 70 frames/s)
and found a ball in 12,414 of the 64,687 frames it searched, the same 12,414 at the same positions
both times (`possession.video.fullframe --compare`: identical, none moved by 1 cm).

**Reproducibility.** Training on the GPU is not bit-for-bit repeatable. The submission path keeps
inference on the CPU (`infer.py`), so the CSV is identical on any machine *given the same
weights*. That describes the first, dropped ROI detector. The full-frame detector the submission uses
runs on the GPU. Its detections repeat exactly on one card (two full runs on the RX 9060 XT gave the
same CSV byte for byte) and differ between GPU families (TECHNICAL.md, section 5).
