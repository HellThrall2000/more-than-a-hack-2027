"""Device benchmark of the video ball detector on the TACTICAL video -- timings only.

    python -m possession.video.bench                       # every available device
    python -m possession.video.bench --devices cpu,cuda --frames 300

Decodes a stretch of first-half frames of the tactical video (games.asset_paths(source="tactical")),
cuts crops of the sizes training and inference use, and times sequential decoding, one training
step and one inference batch per device -- and, on ROCm, with MIOpen on and off.

Writes benchmarks/results/<UTC time>_<device>.json. That file holds timings and the software and
hardware environment only: no pixels, positions, labels or anything else derived frame by frame
from the match (RULES.md, "Do not publish derived data"), so it may be committed. Keep the printed
table to yourself all the same if you are working with a hosted assistant.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import torch

from .. import ROOT
from .. import log as plog
from . import VideoConfig
from .frames import crop_stack, gray_frames
from .model import BallNet, loss_fn

log = plog.get(__name__)
OUT_DIR = os.path.join(ROOT, "benchmarks", "results")
N_VIDEO_FRAMES = 199319             # DATA.md: the tactical video; anything else is the wrong file


def decode(path, start, n, offsets):
    """Grey frames start .. start + n (plus the stacking offsets), and the sequential decode rate."""
    lo, hi = min(offsets), max(offsets)
    needed = list(range(max(0, start + lo), start + n + hi))
    t = time.perf_counter()
    frames = dict(gray_frames(path, needed, N_VIDEO_FRAMES))
    sec = time.perf_counter() - t
    grabbed = needed[-1] + 1        # gray_frames grabs every frame from 0, so this is the real cost
    return frames, {"first_frame": needed[0], "frames_grabbed": grabbed, "seconds": round(sec, 2),
                    "fps": round(grabbed / sec, 1),
                    "full_video_minutes_estimate": round(N_VIDEO_FRAMES / (grabbed / sec) / 60, 1)}


def crops(frames, centres, size, offsets, rng, n):
    """n stacked crops of `size` pixels around random frames and jittered image positions."""
    keys = [k for k in frames if all(k + o in frames for o in offsets)]
    out = []
    for i in rng.randint(len(keys), size=n):
        cx, cy = centres + rng.randn(2) * 200
        out.append(crop_stack([frames[keys[i] + o] for o in offsets], cx, cy, size)[0])
    return torch.from_numpy(np.stack(out))


def timed(fn, warmup, repeat, sync):
    for _ in range(warmup):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    return round((time.perf_counter() - t) / repeat * 1e3, 1)


def run(device, miopen, X_train, X_infer, vcfg: VideoConfig, repeat):
    """One device / MIOpen setting: ms per training step and per inference batch, or the error."""
    row = {"device": device, "miopen": miopen, "train_crop": int(X_train.shape[-1]),
           "train_batch": int(len(X_train)), "train_step_ms": None, "infer_batch_ms": {}, "error": None}
    if device == "cuda":
        torch.backends.cudnn.enabled = bool(miopen)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    torch.manual_seed(vcfg.seed)
    model = BallNet(X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=vcfg.lr)
    x = X_train.to(device)
    uv = torch.full((len(x), 2), x.shape[-1] / 2.0)
    present = torch.ones(len(x), dtype=torch.bool)

    def step():
        loss = loss_fn(model(x.float()), uv, present, x.shape[-1], vcfg.heat_sigma)
        opt.zero_grad()
        loss.backward()
        opt.step()

    try:
        model.train()
        row["train_step_ms"] = timed(step, 2, repeat, sync)
        model.eval()
        with torch.inference_mode():
            for size, Xi in X_infer.items():
                xi = Xi.to(device)
                row["infer_batch_ms"][str(size)] = timed(lambda: model(xi), 2, repeat, sync)
    except RuntimeError as exc:
        row["error"] = str(exc).strip().splitlines()[0][:200]
    finally:
        torch.backends.cudnn.enabled = True
    return row


def environment():
    env = {"python": platform.python_version(), "os": platform.platform(terse=True),
           "cpu": platform.processor(), "torch": torch.__version__, "hip": torch.version.hip,
           "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads()}
    try:
        import cv2
        env["opencv"] = cv2.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=torch.cuda.get_device_name(0), gpu_arch=getattr(p, "gcnArchName", None),
                   vram_gb=round(p.total_memory / 2 ** 30, 1))
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip() or None
    except OSError:
        env["git_commit"] = None
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--devices", default="auto", help="comma list of cpu,cuda; auto = all available")
    ap.add_argument("--start", type=int, default=11185, help="first video frame (default: kick-off)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)
    plog.setup("video_bench")

    from games import asset_paths
    path = asset_paths(data_dir=args.data_dir, source="tactical")["video"]
    if not os.path.isfile(path):
        log.error("tactical video not found: %s", path)
        return 1
    devices = (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])) if args.devices == "auto" \
        else [d.strip() for d in args.devices.split(",") if d.strip()]

    vcfg = VideoConfig()
    rng = np.random.RandomState(vcfg.seed)
    with plog.stage("decode {} frames of the tactical video from frame {}".format(args.frames, args.start)):
        frames, dec = decode(path, args.start, args.frames, vcfg.offsets)
    h, w = next(iter(frames.values())).shape
    centres = np.array([w / 2.0, h / 2.0])
    X_train = crops(frames, centres, vcfg.crop, vcfg.offsets, rng, vcfg.batch)
    X_infer = {s: crops(frames, centres, s, vcfg.offsets, rng, vcfg.infer_batch)
               for s in (vcfg.infer_min_crop, 160, vcfg.infer_max_crop)}
    del frames

    rows = []
    for device in devices:
        for miopen in ((False, True) if device == "cuda" and torch.version.hip else (None,)):
            with plog.stage("time {}{}".format(device, "" if miopen is None else
                                               ", MIOpen " + ("on" if miopen else "off"))):
                rows.append(run(device, miopen, X_train, X_infer, vcfg,
                                args.repeat if device != "cpu" else max(2, args.repeat // 3)))
            if rows[-1]["error"]:
                log.warning("  %s failed: %s", device, rows[-1]["error"])

    now = datetime.datetime.now(datetime.timezone.utc)
    env = environment()
    result = {"benchmark": "possession.video.bench", "when_utc": now.isoformat(timespec="seconds"),
              "video": "tactical", "environment": env, "decode": dec, "runs": rows}
    os.makedirs(args.out_dir, exist_ok=True)
    tag = re.sub(r"[^a-z0-9]+", "-", (env.get("gpu") or "cpu").lower()).strip("-")
    out = os.path.join(args.out_dir, "{}_{}.json".format(now.strftime("%Y%m%dT%H%M%SZ"), tag))
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)

    log.info("decode: {fps} fps sequential -> full video in ~{full_video_minutes_estimate} min".format(**dec))
    log.info("{:<6} {:<7} {:>10}  {}".format("device", "miopen", "train ms", "inference ms per batch of {} by crop".format(
        vcfg.infer_batch)))
    for r in rows:
        log.info("{:<6} {:<7} {:>10}  {}".format(r["device"], str(r["miopen"]), str(r["train_step_ms"]),
                                                 r["error"] or r["infer_batch_ms"]))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
