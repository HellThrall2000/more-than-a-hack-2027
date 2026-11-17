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
| **PyTorch** 2.14 (AMD ROCm 10.1 wheels, RX 9060 XT / gfx1200) | `fullframe.py` | whole 1080p frames, `imgsz=1280`, fp16, batches of 8, up to 5 candidates per frame. Decoding runs in a background thread so the GPU is not left waiting |
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
   ATD's. No threshold reaches the 90% bar, so `--force` takes the best one: 0.4 on the RX 9060 XT
   (0.5 on the NVIDIA card). At 0.4: 48% coverage, 72.1% within 2 m, 85.1% within 5 m (at 0.5: 45%,
   71.8%, 84.6%). The self-check rows are stored in `cache/video_gate_fullframe.json`.

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

Result on the RX 9060 XT: 9,958 frames filled; graded frames with a ball 59.3% → 65.6% (8,795 and
64.8% on the NVIDIA card: detections differ between GPU vendors). The rest of the pipeline reads
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
| `make run-all` | `scripts/run_all.sh` | everything above in one go: full logs in `logs/`, a totals-only summary in `reports/`, the submission file in `submissions/`, a timing/score record in `benchmarks/results/` |

**Live progress.** Each step of `run_all.sh` shows what it is doing on the console while it runs: a bar
with percent, rate, ETA and elapsed time (`video_detect [##########--------] decoding for ball
detection: 58% 110400/190198 frames 79 frames/s ETA 16:49 | 24:12`), the last log line for steps
without a percentage, and a warning when a log has not changed for two minutes. The four parallel
predictions get one status line (`predictions 0:21 | old_rules done | new_rules L4 emission ...`).
Ctrl-C stops the running step and everything it started. On Windows, run it with Git Bash:
`powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 TotalFootball` (`-SkipVideo` reuses the
last ball detections). Stand-alone commands such as `python -m possession.video.fullframe` print a
progress line every 5 seconds.

What a `run_all` round leaves behind: `submissions/<label>.csv` (the file to hand in: a validated
copy of the video-assisted prediction, or of the tracking-only one when there are no detections),
the candidate files it was copied from, `reports/summary_<time>.txt` (aggregates only, safe to
paste) and `benchmarks/results/<time>_run-all.json` (step timings, versions, scores; no match
data). The whistle step needs ffmpeg (`winget install Gyan.FFmpeg`); without it that step fails and
the rest runs unchanged. `write_atd` links the data files into `cache/atd_ours_inside` by symlink,
else hard link, else copy (Windows refuses symlinks without a special privilege). The parsed ATD is
cached in `cache/` under a name that includes a hash of the file, so a new detection run never
reads an old parse.

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
  The `video` extra adds torch / torchvision (AMD's ROCm index, `device-gfx1200`), ultralytics,
  opencv-python-headless, gdown and lightgbm. `requirements-*.txt` remain for pip users.
- This branch (`feat-amd-track2`) targets the AMD Radeon RX 9060 XT (16 GB, ROCm, Windows 11 or Linux,
  Python 3.12–3.13). torch sees the card and trains on it (see [benchmarks/README.md](benchmarks/README.md));
  the full detection run has been made on it twice (Windows 11, torch 2.14.0+rocm10.1.0): 45 and 46
  minutes for 190,198 frames at about 70 frames/s. The NVIDIA setup (RTX 4070 SUPER, CUDA 13,
  Python 3.12–3.14) is on `feat-nvidia-track`. MIOpen is switched off on ROCm (`pick_device`).
- Determinism: detections rounded to 1 cm and 0.001 confidence, stable sorts, fixed seeds. Two full
  detection runs in two NVIDIA environments gave identical detections (10,824 of 10,824). The AMD
  card gives different ones (9,958 frames filled against 8,795; it also settles on another
  threshold, 0.4 against 0.5), so the video-assisted CSV is byte-exact only on the same GPU family.
  Two full runs on the AMD card agree with each other: 12,414 of 12,414 detections, none moved by
  1 cm (`fullframe --compare`), and the same CSV byte for byte (SHA-256 starts `4d550df38280c15e`).
