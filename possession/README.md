# possession/ — the solution package for PLAN.md

Run `make test` (no data needed), then `make demo`. On the real bundle: `make audit`,
`make check-baselines`, `make semantics`, `make submission NAME=<team>`, `make score`.

What counts as dead ball and what counts as possession: [SITUATION.md](../SITUATION.md).

## Running the whole match

```
.venv\Scripts\python -m possession.full_run --name <team>
```

It runs every step in order:
1. `check`: the bundle and the GPU.
2. `audit`: alignment checks.
3. `baselines`: the organiser baselines, byte-identical.
4. `predict_tracking`: tracking-only prediction.
5. `train`: the ball detector, on the GPU.
6. `gate`: does the detector recover hidden balls?
7. `detect`: search the video on every missing-ball frame, on the CPU.
8. `predict_video`: prediction with the video detections.
9. `final`: copy the chosen file to `submissions/<team>.csv`.
10. `reports`: semantics study and diagnosis.

Steps 5-8 (the video detector) run only with `--with-video`. Without it, the submission is the
deterministic tracking-only file, and `final` prints its SHA-256 so a rerun can confirm it is the
same file. Use `--only`, `--skip` or `--from <step>` to run part of it; an interrupted run resumes with
`--from`. If any video step fails, the run carries on, and `final` uses the tracking-only file.

Where to look:

| | |
|---|---|
| console | one line per step (`==>` start, `<==` end, with time and memory) and progress lines with rate and ETA |
| `logs/<UTC>_<command>.log` | the console plus DEBUG detail, the environment (versions, git commit, config hash), warnings, full tracebacks and native crash stacks. Gitignored |
| `benchmarks/results/<UTC>_full-run.json` | each step's status and time, and the aggregate scores |

Every command sets up the same logging: `run`, `audit`, `semantics`, `diagnose`, `harness`,
`video.train`, `video.detect`, `video.bench`. Add `--verbose` to `full_run` to see DEBUG lines on
the console too.

## Built (version 1, rule-based) and what covers it

| PLAN.md | module | tested by |
|---|---|---|
| L0 ingest, team mapping, alignment | `ingest.py`, `audit.py` | `test_ingest`, `test_load_match` (fake bundle) |
| L1 player cleaning, collective state | `players.py` | pipeline tests |
| L2 ball: clean, smooth, gaps, touches/control, out-of-play, stationary | `ball/` | `test_ball` |
| L2 ball scorecard (masked-ball accuracy, no ground truth) | `ball/scorecard.py` | `test_scorecard_audit` |
| L3 features, restart formations, candidate boundaries | `features.py`, `restarts.py`, `candidates.py` | pipeline tests |
| Smart Tagging prior, calibrated on silver frames | `tags.py` | `test_tags` |
| L4 rules emission | `emission_rules.py` | `test_pipeline` (no evidence from an unknown ball) |
| L5 semi-Markov decoder (Viterbi + posterior sampling) | `decode.py` | `test_decode` |
| L6 MBR on spellF1, output policy | `mbr.py`, `policy.py` | `test_mbr` (equals the official scorer) |
| V1–V3 harness, experiment registry | `harness.py` | `test_pipeline` (reproduces README's baseline numbers) |
| Week-1 semantics study (SITUATION.md Q1–Q4): true boundaries vs detector events | `semantics.py` | `test_semantics` |
| On-screen match clock ↔ video frame (review aid; labels stay on video frames) | `clock.py` | `test_clock` |
| Ball-free stoppage evidence, line-crossing extrapolation, restart sides, kick-off team, tag pooling, `--set` | `calm.py`, `ball/events.py`, `candidates.py`, `tags.py`, `config.py` | `test_fixes` |
| V3 labelling tool: change points → ground-truth CSV | `labels.py` | `test_fixes` |
| ATD noise simulator, synthetic match | `simulate_atd.py`, `synthetic.py` | everywhere |

## Video ball recovery (`video/`) — used in the submission

Full description, filters and inspection tools: [VIDEO_AND_DATA.md](../VIDEO_AND_DATA.md).

* **In the submission:** `video/fullframe.py` runs Roboflow's pretrained `football-ball-detection`
  (YOLOv8, Ultralytics, no training) on whole frames, on the GPU, for every graded frame without an
  ATD ball; keeps on-pitch, non-static, steady tracks; `--evaluate` picks the confidence threshold
  on frames where the ATD does see the ball. `video/write_atd.py` writes our ATD (the provided file
  with only missing ball fields filled) and verifies it. `make final` runs both, then the rules.
* **Tried, dropped:** the ROI detector (`dataset.py`, `model.py`, `train.py`, `infer.py`,
  `evaluate.py`, `detect.py`): a small net searching a crop around the gap filler's guess. In long
  gaps the guess is ~13 m off, so it recovered 0.02% of frames. Code kept; `full_run --with-video`
  still drives it.
* **Also kept, switched off:** `audio.py` (referee whistle; not audible on this track) and
  `shape.py` (team from player shape, LightGBM).

Needs `uv sync --extra video` (or `requirements-video.txt`); the rest of the package does not.

## NOT built — needs the data, other datasets, or people

* **Version 2 (learned)**: `emission_gbm.py`, `external.py` (DFL / Metrica loaders). Needs the open
  datasets, LightGBM, and the simulator accepted against this match's real statistics (PLAN 5.5).
* **Tuning of every threshold in `config.py`.** All values are first guesses from PLAN.md, chosen
  without seeing the match. They must be set on V2/V3, never on the public 5 minutes.
* **Hand-labelled windows (V3)** and the timing offsets (`turnover_offset`, `stop_offset`) they set.
  Confirm with the organisers that validation-only hand labels are allowed (RULES.md) first.
* Hull / last-line team descriptors; `review.py` video overlays; broadcast sync.
* A repro check of the video stage on a different GPU or driver, and Python 3.11 testing.

## What the synthetic results do and do not say

On synthetic matches the pipeline beats the nearest-player + majority-vote reference by a wide
margin (spellF1 ≈ 0.75 vs ≈ 0.43–0.57). That shows the layers fit together and the grammar/decoder
logic works. It says nothing about the real match: the generator was written by the same people
who wrote the detectors.

## Data hygiene

`cache/`, `submissions/` and `labels_private/` are git-ignored; they hold frame-level data derived
from the match. The match data and anything extracted from it must not be given to hosted AI
assistants (the terms forbid it, "or any extracts thereof") — develop against `synthetic.py`.
