# Video, data improvement and inspection tools

Companion to [TECHNICAL.md](TECHNICAL.md). TECHNICAL.md explains *why* the system looks the way it
does and what it scores; this file lists *what we used*: the tools that read the tactical video,
the steps that turn it into a better ball track ("our ATD"), and the commands we used to examine
the video and the data while building it. Every command below runs locally; none of them prints
frame-level data unless stated.

---

## 1. Visual extraction — reading the tactical video

| what | where | used for |
|---|---|---|
| **OpenCV** (`opencv-python-headless` 5.0) | `possession/video/frames.py` | sequential decoding of `BAR-ATH_tactical.mp4` with `grab()` / `retrieve()`. No seeking: H.264 seeking is slow and inexact, and one frame off shifts every label. Only the frames needed are converted (grey, or BGR for the detector) |
| **Homography** (provided, one matrix per frame) | `possession/video/geometry.py` | pitch metres ↔ image pixels, vectorised over frames. Used both ways: ATD ball → pixel (free labels), detection pixel → pitch metres |
| **Roboflow `football-ball-detection`** (YOLOv8, [roboflow/sports](https://github.com/roboflow/sports)) via **Ultralytics** 8.4 | `possession/video/fullframe.py` | the ball detector in the submission. Pretrained, **no fine-tuning, no training on this match**. Downloaded once with **gdown** from the link in Roboflow's own `examples/soccer/setup.sh`; SHA-256 logged on every run |
| **PyTorch** 2.14 (CUDA 13 wheels) | `fullframe.py` | whole 1080p frames, `imgsz=1280`, fp16, batches of 8, up to 5 candidates per frame. Decoding runs in a background thread so the GPU is not left waiting |
| our own ROI detector (`BallNet`, CenterNet-style) | `possession/video/{dataset,model,train,infer,evaluate,detect}.py` | first attempt; **dropped** (see section 4) |

Commands:

```bash
make video-check     # python -m possession.video.fullframe --evaluate   (self-check, picks the threshold)
make video-detect    # python -m possession.video.fullframe              (every missing-ball graded frame)
```

Outputs (gitignored, frame-level): `cache/video_gate_fullframe.json`, `cache/video_ball_fullframe.npz`.

## 2. Improving the data — from raw detections to our ATD

The detector returns up to five boxes per frame. Four steps turn them into ball positions we trust,
none of which uses a possession label:

1. **Pixels → pitch.** Box centres are projected through that frame's homography
   (`fullframe.to_pitch`). The projection is onto the ground plane, so an airborne ball lands beyond
   where it really is; the later filters allow for that.
2. **Plausibility** (`fullframe.plausible`):
   - *on the pitch* — a candidate more than 2 m outside the lines is a spare ball or a ball boy's;
   - *not static* — candidates are counted on a 0.5 m grid (with the 8 neighbouring cells); a cell
     holding detections on more than 1% of frames and at least 250 frames (10 s) is a painted spot
     or a parked ball, and is removed. A ball resting for a restart is shorter than that.
3. **Steady tracks** (`fullframe.tracklets`): candidates above the confidence threshold are chained
   frame to frame only where the jump is one a ball can make (`ball_max_speed` plus slack, gaps of
   at most 5 frames). Chains shorter than 8 detections — a boot, a head, a flicker — are dropped;
   where several chains survive in a frame, the longest wins.
4. **Threshold from a self-check, not from labels** (`--evaluate`): the detector runs on 3,000
   frames (3 s windows) where the ATD *does* see the ball, and its positions are compared with the
   ATD's. At confidence 0.5: 45% coverage, 72.9% within 2 m, 85.3% within 5 m. The chosen threshold
   is stored in `cache/video_gate_fullframe.json`.

Then `possession/video/write_atd.py` writes **our ATD**:

```bash
python -m possession.video.write_atd --out-root cache/atd_ours_inside --inside-m 1.0
MTAH_DATA_DIR=cache/atd_ours_inside python -m possession.run --out submissions/x.csv
```

- the organisers' file, line for line; every frame id and player group byte-identical;
- only a ball group that is missing (starts with `N`) **and** has a detection at or above the
  threshold is replaced, as `x,y,speed` in the same normalised coordinates and decimals;
- `--inside-m 1.0`: only detections at least 1 m inside the lines are written, so a video detection
  alone can never put the ball out of play;
- metadata, homography, Smart Tagging and video are linked, not copied;
- `verify` re-parses both files and checks: players identical, original balls identical, line count
  identical, all new values finite. It prints the share of frames with a ball before and after.

Result: 8,795 frames filled; graded frames with a ball 59.3% → 64.8%. The rest of the pipeline reads
our ATD exactly as it reads the original (`MTAH_DATA_DIR`), so nothing downstream had to change.

`make compare-atd` (`scripts/compare_atd.sh`) runs the given ATD and our ATD side by side, each with
the version-3 rules and the newer rules, and writes a totals-only summary to `reports/`.

## 3. Examining the video and the data

Tools we used to understand the match, find bugs and check each change. All print aggregates
(counts, rates, percentiles) unless noted, so their output can be shared without redistributing
the data.

| command | module | what it tells us |
|---|---|---|
| `make audit` | `audit.py` | first run on the bundle: ball channels in the ATD metadata, ball-gap lengths, slot lifetimes, objects per frame, camera pan, ball position relative to image centre, alignment and time-base checks |
| `make check-baselines` | `starter/baseline.py` | rebuilds the organiser baselines; must be byte-identical (proves our ingest reads the data as they do) |
| ball scorecard | `ball/scorecard.py` | hides visible stretches of the ATD ball using the real gap-length distribution and measures how well the gap filler reconstructs them, by gap length. Judges the ball layer without the decoder |
| `python -m possession.clock 51:35 --half 2` / `--frame N` | `clock.py` | on-screen match clock ↔ video frame (anchors 00:00 = frame 11185, 45:00 = frame 113742). Used to find a moment seen on screen in the data, and to look up a frame in the video |
| `python -m possession.labels <file>.txt` | `labels.py` | turns hand-written change points (video frame, video time or match clock) into a ground-truth CSV for a validation window. Hand labels validate only, never go into a submission, and stay in gitignored `labels_private/` |
| `make semantics` | `semantics.py` | where the public ground truth puts each boundary relative to our detector events (out of play, rest, calm, control, set pieces): signed offsets per kind of event. Answers what the label definition is; not used to tune |
| `make diagnose` | `diagnose.py` | where points are lost vs a ground truth, layer by layer: confusion matrix, emission alone vs decoded, whether a candidate boundary exists near each true transition, what the ball layer saw. For finding bugs, not tuning |
| `make report` | `report.py` | totals-only cross-check of a full-match prediction against Smart Tagging set pieces and whistles (e.g. 58 of 59 set pieces fall at one of our dead-ball stretches) |
| `make audio` | `audio.py` | referee-whistle detector on the video's audio track (2–4 kHz narrow-band tone, plain NumPy). Found 18 events in 2 h 13 min, barely better than chance against tagged set pieces: **dropped** |
| `python -m possession.video.bench` | `video/bench.py` | decoding, training-step and inference timings per device on real tactical crops; writes `benchmarks/results/*.json` (timings and environment only) |
| `make run-all` | `scripts/run_all.sh` | everything above in one go, full logs in `logs/`, a totals-only summary in `reports/` |

Logging: every command writes `logs/<UTC>_<command>.log` (structlog) with the environment (versions,
git commit, config hash), warnings and full tracebacks. `logs/`, `reports/`, `cache/` and
`labels_private/` are gitignored.

## 4. Tried and dropped

| attempt | why it was dropped |
|---|---|
| **ROI ball detector trained on this match** (`video/train.py` etc.): a small fully-convolutional net on three stacked grey frames (t−3, t, t+3), labels for free from the ATD ball projected through the homography, crops centred on the gap filler's guess | in long gaps the guess is about 13 m off, so the crop rarely contains the ball; gate coverage 0.02%. The full-frame pretrained detector replaced it |
| **Referee whistle from audio** (`audio.py`) | the referee is not audible on this track (section 3) |
| **Team from player shape** (`shape.py`, LightGBM on player positions, trained on frames where control is known; 82% held out) | neutral or worse on the public window; code kept, switched off (`emit_shape_weight=0`) |
| restart team from the taker, goalkeeper catch vs parry | same; switched off by the `V3_SET` in the Makefile |
| broadcast video and ATD | not synchronised with the tactical timeline |

## 5. Environment

- `pyproject.toml` + `uv.lock` are the source of truth: `make setup` = `uv sync --extra video`.
  The `video` extra adds torch / torchvision (CUDA 13.0 index), ultralytics, opencv-python-headless,
  gdown and lightgbm. `requirements-*.txt` remain for pip users (and the AMD ROCm instructions).
- Tested on Linux, Python 3.12–3.14, NVIDIA RTX 4070 SUPER (12 GB). The earlier AMD track (RX 9060 XT,
  ROCm, Windows) is documented in [benchmarks/README.md](benchmarks/README.md).
- Determinism: detections rounded to 1 cm and 0.001 confidence, stable sorts, fixed seeds. Two full
  detection runs in two environments gave identical detections (10,824 of 10,824).
