"""L0 -- read the provided assets into plain arrays on the submission's timeline (PLAN.md 5.1).

Everything is indexed by *video frame* (row i of a submission). The ATD counts from 1, so
video_frame = frame_id - 1; the per-half offsets in games.py belong to the ground truth's counter
and are not used here. Coordinates leave this module in metres, origin at a pitch corner.

The raw ATD is streamed by hand rather than through kloppy: kloppy needs minutes and gigabytes for
199,319 frames x 46 slots, this is one pass over the file.
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional, Tuple

import numpy as np

from . import ROOT  # noqa: F401  (puts the repository root on sys.path)
from . import log as plog
from .config import Config

log = plog.get(__name__)

FIELD_SEP = ":"
GROUP_SEP = ";"


@dataclasses.dataclass
class Tracking:
    players_xy: np.ndarray      # (N, S, 2) float32, metres, NaN where untracked
    players_speed: np.ndarray   # (N, S) float32, m/s as reported by the ATD
    ball_xy: np.ndarray         # (N, 2) float32, metres, NaN where the ball is not seen
    ball_extra: np.ndarray      # (N, K) float32, any ball fields beyond x,y (speed, height, ...)


@dataclasses.dataclass
class MatchInputs:
    """Everything the pipeline consumes, independent of where it came from (real or synthetic)."""
    tracking: Tracking
    slot_team: np.ndarray                       # (S,) 0 home, 1 away
    blocks: List[Tuple[int, int, int]]          # [(half, first_frame, stop_frame)] stop exclusive
    tags: List[dict]                            # Smart Tagging intervals (load_smart_tagging format)
    homography: Optional[np.ndarray] = None     # (N, 3, 3) normalised pitch -> pixels
    homography_valid: Optional[np.ndarray] = None
    kickoffs: Optional[dict] = None             # {half: first alive frame} when known
    kickoff_team: Optional[dict] = None         # {half: 0 home / 1 away} kicking off, when known
    video_ball: Optional[tuple] = None          # (xy (N, 2), conf (N,), thr) from possession.video
    n_frames: int = 0

    def __post_init__(self):
        if not self.n_frames:
            self.n_frames = len(self.tracking.ball_xy)


def parse_atd(raw_path, n_frames, pitch, max_ball_fields=4) -> Tracking:
    """Stream the raw FIFA/EPTS text: `<frame_id>:<slot>;...;<slot>:<ball>`, groups `x,y,speed`.

    A missing observation is `NaN...`; only the first character is tested, as baseline.py does.
    Frames absent from the file stay NaN, so a short file degrades instead of shifting rows.
    """
    length, width = pitch
    with open(raw_path) as handle:
        first = next((ln for ln in handle if ln.count(FIELD_SEP) == 2), None)
    if first is None:
        raise ValueError("{}: no ATD records found".format(raw_path))
    n_slots = len(first.split(FIELD_SEP)[1].split(GROUP_SEP))

    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, max_ball_fields), np.nan, dtype=np.float32)

    skipped = 0
    with open(raw_path) as handle, plog.Progress(n_frames, "parsing ATD", "frames") as progress:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                skipped += 1
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                skipped += 1
                continue
            progress.set(frame + 1)
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots or not group or group[0] == "N":
                    continue
                values = group.split(",")
                xy[frame, slot, 0] = float(values[0])
                xy[frame, slot, 1] = float(values[1])
                if len(values) > 2:
                    speed[frame, slot] = float(values[2])
            group = parts[2]
            if group and group[0] != "N":
                values = group.split(",")[:max_ball_fields]
                ball[frame, :len(values)] = [float(v) for v in values]

    if skipped:
        log.warning("ATD: %d lines skipped (malformed, or frame outside 1..%d)", skipped, n_frames)
    xy[..., 0] *= length
    xy[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return Tracking(xy, speed, ball[:, :2].copy(), ball[:, 2:].copy())


def graded_blocks(game="BAR-ATH") -> List[Tuple[int, int, int]]:
    """The two graded halves as half-open [first, stop) video-frame blocks."""
    from games import half_video_blocks
    return [(half, first, last + 1) for half, first, last in half_video_blocks(game)]


def load_match(cfg: Config, data_dir=None, game="BAR-ATH", use_cache=True, cache_dir=None,
               with_homography=True) -> MatchInputs:
    """Real-data entry point. Needs the gated bundle in ./data (or MTAH_DATA_DIR)."""
    from games import GAMES, require
    from starter.baseline import team_of_each_track
    from starter import load_smart_tagging

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    pitch = GAMES[game]["pitch"]
    n_frames = GAMES[game]["n_video_frames"]
    cache_dir = cache_dir or os.path.join(ROOT, "cache")
    cache_file = os.path.join(cache_dir, "atd_{}_{}.npz".format(os.path.getsize(raw_path), n_frames))

    if use_cache and os.path.isfile(cache_file):
        z = np.load(cache_file)
        tracking = Tracking(z["players_xy"], z["players_speed"], z["ball_xy"], z["ball_extra"])
        log.info("ATD loaded from cache %s", plog.rel(cache_file))
    else:
        log.info("ATD: parsing %s (first run only; cached afterwards)", os.path.basename(raw_path))
        tracking = parse_atd(raw_path, n_frames, pitch)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_file, players_xy=tracking.players_xy,
                                players_speed=tracking.players_speed, ball_xy=tracking.ball_xy,
                                ball_extra=tracking.ball_extra)

    slot_team = team_of_each_track(meta_path)
    if len(slot_team) != tracking.players_xy.shape[1]:
        raise ValueError("metadata lists {} slots but the ATD has {}".format(
            len(slot_team), tracking.players_xy.shape[1]))

    tags = load_smart_tagging.load(game, data_dir)

    homography = valid = None
    if with_homography:
        from starter.load_homography import load as load_h
        homography, _, valid = load_h(game, data_dir)

    blocks = graded_blocks(game)
    graded = np.zeros(n_frames, dtype=bool)
    for _, a, b in blocks:
        graded[a:b] = True
    log.info("match: %d frames, %d graded | %d track slots (%d home, %d away) | ball seen on %.1f%% of "
             "graded frames | %d Smart Tagging intervals | homography %s",
             n_frames, int(graded.sum()), len(slot_team), int((slot_team == 0).sum()),
             int((slot_team == 1).sum()), 100.0 * float((~np.isnan(tracking.ball_xy[graded, 0])).mean()),
             len(tags), "not loaded" if valid is None else "valid on {:.1f}% of frames".format(
                 100.0 * float(np.mean(valid))))
    from .tags import kickoff_teams
    kickoff_team = kickoff_teams(tags)
    log.info("kick-off team by half (Smart Tagging + Law 8): %s",
             {h: "HA"[t] for h, t in sorted(kickoff_team.items())} or "unknown")
    return MatchInputs(tracking, slot_team, blocks, tags, homography, valid,
                       kickoffs={1: blocks[0][1], 2: cfg.second_half_kickoff},
                       kickoff_team=kickoff_team, n_frames=n_frames)
