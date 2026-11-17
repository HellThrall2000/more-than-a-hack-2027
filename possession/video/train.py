"""Train the ball locator on crops whose labels come from the ATD + homography.

    python -m possession.video.train --device auto          # AMD GPU via ROCm, or CPU

Writes models/ball_detector.pt, including the held-out report and the confidence threshold at
which precision reaches the PLAN.md gate. Training is NOT bit-for-bit reproducible across GPUs;
what the submission depends on is the saved weights plus CPU inference (infer.py defaults to CPU).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import numpy as np
import torch

from .. import ROOT
from .. import log as plog
from ..config import DEFAULT
from . import VideoConfig
from .dataset import CropSet
from .model import BallNet, decode, loss_fn, pick_device

_log = plog.get(__name__)


def split(cs: CropSet, vcfg: VideoConfig):
    """Train / validation indices by contiguous time blocks, so near-duplicate frames never leak."""
    seg = cs.frame // vcfg.val_segment_frames
    val = (seg % vcfg.val_every) == (vcfg.val_every - 1)
    if not val.any() or val.all():
        raise ValueError("cannot hold out validation blocks: {} distinct blocks of {} frames".format(
            len(np.unique(seg)), vcfg.val_segment_frames))
    return np.flatnonzero(~val), np.flatnonzero(val)


@torch.no_grad()
def predict_crops(model, X, device, batch=256):
    model.eval()
    conf, u, v = [], [], []
    for i in range(0, len(X), batch):
        out = model(torch.from_numpy(X[i:i + batch]).to(device))
        c, uu, vv = decode(out)
        conf.append(c), u.append(uu), v.append(vv)
    return np.concatenate(conf), np.concatenate(u), np.concatenate(v)


def sweep(conf, u, v, uv, present, vcfg: VideoConfig):
    """Precision / recall over confidence thresholds; picks the lowest one that meets the gate."""
    err = np.hypot(u - np.nan_to_num(uv[:, 0]), v - np.nan_to_num(uv[:, 1]))
    rows = []
    for thr in np.round(np.arange(0.05, 1.0, 0.05), 2):
        det = conf >= thr
        correct = det & present & (err <= vcfg.pixel_tol)
        tp = int(correct.sum())
        fp = int((det & ~present).sum() + (det & present & ~correct).sum())
        rows.append({"thr": float(thr), "precision": tp / max(1, tp + fp),
                     "recall": tp / max(1, int(present.sum())),
                     "false_alarm_rate_on_empty": float((det & ~present).sum()) / max(1, int((~present).sum()))})
    ok = [r for r in rows if r["precision"] >= vcfg.gate_precision and r["recall"] > 0]
    chosen = ok[0] if ok else max(rows, key=lambda r: r["precision"])
    return {"thr": chosen["thr"], "meets_gate": bool(ok), "at_thr": chosen, "sweep": rows,
            "median_px_err": float(np.median(err[present & (conf >= chosen["thr"])]))
            if (present & (conf >= chosen["thr"])).any() else None}


def fit(cs: CropSet, vcfg: VideoConfig, device, log=None):
    log = log or _log.info
    torch.manual_seed(vcfg.seed)
    rng = np.random.RandomState(vcfg.seed)
    tr, va = split(cs, vcfg)
    size = cs.X.shape[-1]
    model = BallNet(cs.X.shape[1]).to(device)
    steps = vcfg.epochs * int(np.ceil(len(tr) / vcfg.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=vcfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=vcfg.lr, total_steps=steps)
    log("training on {} crops, validating on {} ({} positive)".format(
        len(tr), len(va), int(cs.present[va].sum())))
    t_start = time.perf_counter()
    for epoch in range(vcfg.epochs):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        total = 0.0
        t_epoch = time.perf_counter()
        batches = plog.Progress(int(np.ceil(len(perm) / vcfg.batch)), "epoch {} batches".format(epoch + 1),
                                "batches", every=30.0)
        for i in range(0, len(perm), vcfg.batch):
            b = perm[i:i + vcfg.batch]
            x = torch.from_numpy(cs.X[b]).to(device)
            uv = torch.from_numpy(cs.uv[b]).clone()
            present = torch.from_numpy(cs.present[b])
            flip = torch.from_numpy(rng.rand(len(b)) < 0.5)
            x = torch.where(flip.to(device)[:, None, None, None], x.flip(-1), x)
            uv[flip & present, 0] = (size - 1) - uv[flip & present, 0]
            x = x.float() + torch.randn(x.shape, device=device) * 2.0
            loss = loss_fn(model(x), uv, present, size, vcfg.heat_sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(b)
            batches.update()
        if not np.isfinite(total):
            raise FloatingPointError("training loss is not finite in epoch {}".format(epoch + 1))
        took = time.perf_counter() - t_epoch
        left = (time.perf_counter() - t_start) / (epoch + 1) * (vcfg.epochs - epoch - 1)
        log("  epoch {:2d}/{}  loss {:.4f}  {:.0f} crops/s  {}  ETA {}{}".format(
            epoch + 1, vcfg.epochs, total / len(perm), len(perm) / took, plog.duration(took),
            plog.duration(left), plog.memory()))
    conf, u, v = predict_crops(model, cs.X[va], device)
    report = sweep(conf, u, v, cs.uv[va], cs.present[va], vcfg)
    log("held-out: threshold {:.2f}  precision {:.3f}  recall {:.3f}  median error {} px  gate {}".format(
        report["thr"], report["at_thr"]["precision"], report["at_thr"]["recall"],
        "--" if report["median_px_err"] is None else "{:.2f}".format(report["median_px_err"]),
        "MET" if report["meets_gate"] else "NOT MET"))
    return model, report


def save_model(path, model, vcfg, report):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state": model.state_dict(), "in_ch": model.body[0][0].in_channels,
                "vcfg": dataclasses.asdict(vcfg), "report": report}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vcfg = VideoConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["vcfg"].items()})
    model = BallNet(ck["in_ch"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), vcfg, ck["report"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ball_detector.pt"))
    ap.add_argument("--crops", default=os.path.join(ROOT, "cache", "video", "crops.npz"),
                    help="cache of the extracted crops (frame-level data: stays in cache/, never committed)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args(argv)
    plog.setup("video_train")

    from games import asset_paths
    from ..ball import clean as ball_clean
    from ..ingest import load_match
    from ..players import clean_players
    from . import dataset

    changes = {k: v for k, v in (("epochs", args.epochs), ("stride", args.stride)) if v}
    vcfg = dataclasses.replace(VideoConfig(), **changes)
    device = pick_device(args.device)
    _log.info("device: %s %s", device, "(ROCm/HIP {}, MIOpen {})".format(
        torch.version.hip, "on" if torch.backends.cudnn.enabled else "off") if torch.version.hip else "")

    if os.path.isfile(args.crops):
        cs = CropSet.load(args.crops)
        _log.info("loaded %d cached crops from %s (delete it to rebuild)", len(cs.X), plog.rel(args.crops))
    else:
        with plog.stage("build training crops from the tactical video"):
            inputs = load_match(DEFAULT, args.data_dir)
            video = args.video or asset_paths(data_dir=args.data_dir)["video"]
            balls = ball_clean.clean(inputs.tracking.ball_xy, DEFAULT)
            players = clean_players(inputs.tracking.players_xy, inputs.slot_team, DEFAULT)
            cs = dataset.build(video, balls, players, inputs.homography, inputs.blocks, DEFAULT, vcfg,
                               expected_frames=inputs.n_frames)
            os.makedirs(os.path.dirname(args.crops), exist_ok=True)
            cs.save(args.crops)
    with plog.stage("train the ball detector on {}".format(device)):
        model, report = fit(cs, vcfg, device)
    save_model(args.out, model, vcfg, report)
    _log.info("saved %s", plog.rel(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
