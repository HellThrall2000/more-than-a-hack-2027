# possession/ — the solution package for PLAN.md

Run `make test` (no data needed), then `make demo`. On the real bundle: `make audit`,
`make check-baselines`, `make submission NAME=<team>`, `make score`.

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
| ATD noise simulator, synthetic match | `simulate_atd.py`, `synthetic.py` | everywhere |

## Optional: video ball detector (`video/`)

Recovers ball positions on missing-ball frames from the tactical video. Labels are free: where the
ATD sees the ball, the homography projects it to a pixel. A small fully-convolutional net finds the
ball in a crop of three stacked frames; at inference the crop is centred on where the ball layer
expects the ball. `python -m possession.video.detect --evaluate` is the go / no-go gate: it hides
visible balls and measures whether the detector recovers them (precision >= 0.9 required).
Tested on a rendered synthetic clip only (`tests/test_video.py`); NOT run on the real video.
Needs `requirements-video.txt`; the rest of the package does not.

## NOT built — needs the data, other datasets, or people

* **Version 2 (learned)**: `emission_gbm.py`, `external.py` (DFL / Metrica loaders). Needs the open
  datasets, LightGBM, and the simulator accepted against this match's real statistics (PLAN 5.5).
* **Tuning of every threshold in `config.py`.** All values are first guesses from PLAN.md, chosen
  without seeing the match. They must be set on V2/V3, never on the public 5 minutes.
* **Hand-labelled windows (V3)** and the timing offsets (`turnover_offset`, `stop_offset`) they set.
  Confirm with the organisers that validation-only hand labels are allowed (RULES.md) first.
* Extrapolated line crossings for a ball lost on its way out; hull / last-line team descriptors;
  `review.py` video overlays; `TECHNICAL.md`; stretch modules (broadcast sync, ROI ball recovery).
* The real-data run itself, the two-machine `make repro-check`, and Python 3.11 testing.

## What the synthetic results do and do not say

On synthetic matches the pipeline beats the nearest-player + majority-vote reference by a wide
margin (spellF1 ≈ 0.75 vs ≈ 0.43–0.57). That shows the layers fit together and the grammar/decoder
logic works. It says nothing about the real match: the generator was written by the same people
who wrote the detectors.

## Data hygiene

`cache/`, `submissions/` and `labels_private/` are git-ignored; they hold frame-level data derived
from the match. The match data and anything extracted from it must not be given to hosted AI
assistants (the terms forbid it, "or any extracts thereof") — develop against `synthetic.py`.
