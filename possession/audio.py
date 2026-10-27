"""The referee's whistle from the tactical video's audio track (SITUATION.md section 10.3).

    python -m possession.audio --video data/tactical/BAR-ATH_tactical.mp4

The audio is read and analysed on this machine only (FAQ.md: the match media never goes to a hosted
model). The command prints aggregates -- how many whistles, how long -- never the frame list.

A whistle is a short, loud, narrow tone around 2-4 kHz. Crowd noise is broadband, so a frame counts
only if most of its energy is in the band AND most of the band's energy sits next to one peak. The
detector is plain numpy on the CPU: a two-hour track takes seconds, and the result has to be the
same file on any machine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from . import ROOT
from . import log as plog
from .config import Config, DEFAULT
from .util import runs

log = plog.bind(__name__)

CACHE_DIR = os.path.join(ROOT, "cache", "audio")
CHUNK_FRAMES = 4096                 # spectra per FFT batch: bounds memory at ~100 MB


class AudioError(RuntimeError):
    """The audio could not be read; the message says what to do about it."""


class _Transient(AudioError):
    """Worth retrying: a busy disk, a killed child process, a timeout."""


def _run(cmd, timeout):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise AudioError("{} not found: install ffmpeg (sudo apt install ffmpeg)".format(cmd[0])) from None
    except subprocess.TimeoutExpired as exc:
        raise _Transient("{} timed out after {} s".format(cmd[0], timeout)) from exc


_retry = retry(retry=retry_if_exception_type(_Transient), stop=stop_after_attempt(3),
               wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True,
               before_sleep=before_sleep_log(plog.get(__name__), 30))


@_retry
def probe(video):
    """{'codec', 'channels', 'sample_rate', 'duration_s'} of the first audio stream, or None."""
    if not os.path.isfile(video):
        raise AudioError("no such video: {}".format(video))
    res = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                "stream=codec_name,channels,sample_rate,duration", "-of", "json", video], 60)
    if res.returncode != 0:
        raise _Transient("ffprobe failed ({}): {}".format(res.returncode, res.stderr.strip()[:200]))
    streams = json.loads(res.stdout or "{}").get("streams", [])
    if not streams:
        return None
    st = streams[0]
    return {"codec": st.get("codec_name"), "channels": int(st.get("channels", 0)),
            "sample_rate": int(st.get("sample_rate", 0)), "duration_s": float(st.get("duration", "nan"))}


@_retry
def extract(video, cfg: Config = DEFAULT, cache_dir=CACHE_DIR):
    """Mono 16-bit WAV at cfg.audio_rate in cache/audio/, written atomically; reused when newer
    than the video. Returns its path."""
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, "{}_{}hz.wav".format(os.path.splitext(os.path.basename(video))[0], cfg.audio_rate))
    if os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(video):
        log.info("audio cached", wav=plog.rel(out))
        return out
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=cache_dir)
    os.close(fd)
    try:
        res = _run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(cfg.audio_rate),
                    "-c:a", "pcm_s16le", tmp], 3600)
        if res.returncode != 0:
            raise _Transient("ffmpeg failed ({}): {}".format(res.returncode, res.stderr.strip()[:300]))
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    log.info("audio extracted", wav=plog.rel(out), mb=round(os.path.getsize(out) / 2 ** 20, 1))
    return out


def read_wav(path):
    """(samples float32 in [-1, 1], sample rate)."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise AudioError("{}: expected mono 16-bit PCM".format(path))
        raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, w.getframerate()


def tonal_features(samples, rate, cfg: Config = DEFAULT):
    """Per video frame: (band_ratio, narrow, level_db, peak_hz), each (n_frames,).

    band_ratio: share of the spectrum's energy inside the whistle band. narrow: share of the band's
    energy within +/- whistle_peak_hz of its strongest bin. level_db: RMS re full scale. peak_hz:
    the strongest bin in the band. One spectrum per video frame (hop = rate / fps), centred on it.
    """
    hop = rate // cfg.fps
    win = cfg.audio_win
    n_frames = len(samples) // hop
    x = np.concatenate([np.zeros(win // 2, np.float32), samples, np.zeros(win, np.float32)])
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    band = (freqs >= cfg.whistle_lo_hz) & (freqs <= cfg.whistle_hi_hz)
    bf = freqs[band]
    window = np.hanning(win).astype(np.float32)
    win_ms = float(np.mean(window ** 2))
    ratio, narrow, level, peak_hz = (np.zeros(n_frames, np.float32) for _ in range(4))
    idx = np.arange(win)[None, :]
    for c0 in range(0, n_frames, CHUNK_FRAMES):
        c1 = min(c0 + CHUNK_FRAMES, n_frames)
        seg = x[(np.arange(c0, c1) * hop)[:, None] + idx] * window
        power = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        in_band = power[:, band]
        band_sum = in_band.sum(axis=1) + 1e-20
        peak = in_band.argmax(axis=1)
        near = np.abs(bf[None, :] - bf[peak][:, None]) <= cfg.whistle_peak_hz
        ratio[c0:c1] = band_sum / (power.sum(axis=1) + 1e-20)
        narrow[c0:c1] = (in_band * near).sum(axis=1) / band_sum
        level[c0:c1] = 10.0 * np.log10(np.mean(seg ** 2, axis=1) / win_ms + 1e-20)
        peak_hz[c0:c1] = bf[peak]
    return ratio, narrow, level, peak_hz


def whistle_frames(samples, rate, cfg: Config = DEFAULT):
    """Boolean per video frame: this frame's audio looks like a whistle tone (before duration checks)."""
    ratio, narrow, level, _ = tonal_features(samples, rate, cfg)
    return (level > cfg.whistle_floor_db) & (ratio > cfg.whistle_band_ratio) & (narrow > cfg.whistle_peak_ratio)


def events_from_mask(mask, cfg: Config = DEFAULT):
    """[(start, stop)] runs of `mask`, merged across short holes, kept if whistle_min_s..whistle_max_s long."""
    hole = int(round(cfg.whistle_hole_s * cfg.fps))
    merged = []
    for a, b in zip(*runs(mask)):
        if merged and a - merged[-1][1] <= hole:
            merged[-1][1] = int(b)
        else:
            merged.append([int(a), int(b)])
    lo, hi = cfg.whistle_min_s * cfg.fps, cfg.whistle_max_s * cfg.fps
    return [(a, b) for a, b in merged if lo <= b - a <= hi]


def detect_whistles(samples, rate, cfg: Config = DEFAULT):
    """[(start_frame, stop_frame)] of whistles on the audio's own timeline (frame = seconds * fps).

    The sound reaches the microphone late and the file may carry an offset: apply it with
    `shift` before comparing with video events (SITUATION.md section 10.3).
    """
    return events_from_mask(whistle_frames(samples, rate, cfg), cfg)


def shift(whistles, frames):
    """The same whistles moved earlier by `frames` (sound travels ~343 m/s: 4-5 frames from 60 m)."""
    return [(a - frames, b - frames) for a, b in whistles]


def calibrate(samples, rate, tags, cfg: Config = DEFAULT):
    """Aggregate-only table: which thresholds make whistles line up with the SET PIECES tags?

    A set piece restarts play, so the whistle that stops it sits inside or just before its tag.
    `lift` = hit rate / chance (the share of the match covered by those tag windows): 1.0 means the
    whistles are no closer to set pieces than random times. Smart Tagging is independent of the
    possession labels, so this does not tune anything to the answer key.
    """
    ratio, narrow, level, peak_hz = tonal_features(samples, rate, cfg)
    fps, n = cfg.fps, len(ratio)
    cover = np.zeros(n, dtype=bool)
    lead, tail = int(8 * fps), int(3 * fps)
    for t in tags:
        if t["code"].lower() == "set pieces":
            cover[max(0, t["start_frame"] - lead):min(n, t["end_frame"] + tail)] = True
    chance = float(cover.mean())
    log.info("tonal features", band_ratio_pctl=np.percentile(ratio, [50, 90, 99, 99.9]).round(2).tolist(),
             narrow_pctl=np.percentile(narrow, [50, 90, 99, 99.9]).round(2).tolist(),
             level_db_pctl=np.percentile(level, [10, 50, 90, 99]).round(1).tolist(),
             set_piece_window_share=round(chance, 4))
    rows = []
    for br in (0.15, 0.2, 0.25, 0.3, 0.35):
        for nr in (0.3, 0.4, 0.5, 0.6):
            for floor in (-55.0, -45.0, -35.0):
                events = events_from_mask((level > floor) & (ratio > br) & (narrow > nr), cfg)
                if len(events) < 20:
                    continue
                hit = float(np.mean([cover[a] for a, _ in events]))
                rows.append((hit / max(chance, 1e-9), len(events), br, nr, floor, hit))
    rows.sort(reverse=True)
    for lift, count, br, nr, floor, hit in rows[:12]:
        log.info("candidate", events=count, band_ratio=br, narrow=nr, floor_db=floor,
                 hit_rate=round(hit, 3), lift=round(lift, 2))
    top = [e for e in events_from_mask((level > -45) & (ratio > 0.25) & (narrow > 0.4), cfg)]
    if top:
        peaks = np.array([np.median(peak_hz[a:b]) for a, b in top])
        log.info("peak frequency of those events (Hz)", p10_50_90=np.percentile(peaks, [10, 50, 90]).round(0).tolist())
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", default=None, help="defaults to the tactical video of the data bundle")
    ap.add_argument("--calibrate", action="store_true",
                    help="print how threshold choices line up with the SET PIECES tags (aggregates only)")
    args = ap.parse_args(argv)
    plog.setup("audio")
    video = args.video
    if video is None:
        from games import asset_paths
        video = asset_paths()["video"]
    try:
        info = probe(video)
        if info is None:
            log.error("the video has NO audio stream", video=os.path.basename(video),
                      hint="whistle evidence is unavailable; the rest of the pipeline is unaffected")
            return 2
        log.info("audio stream", **info)
        samples, rate = read_wav(extract(video))
    except AudioError as exc:
        log.error("audio unavailable", reason=str(exc))
        return 1
    if args.calibrate:
        from .ingest import load_match
        calibrate(samples, rate, load_match(DEFAULT).tags)
        return 0
    with plog.stage("find whistles"):
        whistles = detect_whistles(samples, rate)
    dur = np.array([b - a for a, b in whistles]) / DEFAULT.fps
    log.info("whistles found", count=len(whistles), total_s=round(float(samples.size / rate), 1),
             median_len_s=round(float(np.median(dur)), 2) if len(dur) else None,
             longest_s=round(float(dur.max()), 2) if len(dur) else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
