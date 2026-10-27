"""Every threshold and hyper-parameter in one frozen dataclass (PLAN.md section 8).

The values are first guesses taken from PLAN.md. NONE of them has been tuned on the match data:
that needs the data bundle and the validation sets of PLAN.md section 7, and the public 5 minutes
must never be used to choose between values.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json


@dataclasses.dataclass(frozen=True)
class Config:
    # -- match constants (games.py is authoritative for the real match) ------------------------
    fps: int = 25
    pitch_length: float = 105.3
    pitch_width: float = 68.0
    second_half_kickoff: int = 113742       # FAQ.md: 110808..113741 are graded and all D

    # -- L1 players ----------------------------------------------------------------------------
    player_max_speed: float = 12.0          # m/s; faster implied jumps are dropped
    player_gap_fill: int = 10               # frames
    dup_merge_dist: float = 0.6             # m; same-team detections closer than this are one
    walk_speed: float = 2.0                 # m/s; below this a player counts as walking
    whistle_speed: float = 1.2              # m/s; mean team speed under this means play stopped
    whistle_min_frames: int = 75            # 3 s sustained

    # -- L2 ball: clean / smooth ---------------------------------------------------------------
    ball_max_speed: float = 40.0            # m/s
    ball_jump_slack: float = 1.0            # m of position noise allowed on top of the speed cap
    ball_reject_reset: int = 8              # consecutive rejections after which the anchor is wrong
    ball_tiny_gap: int = 2                  # frames interpolated before smoothing
    sg_window: int = 7
    sg_order: int = 2

    # -- L2 ball: touches and control ----------------------------------------------------------
    control_dist: float = 1.5               # m
    control_rel_speed: float = 2.0          # m/s between ball and nearest player
    control_min_frames: int = 4
    control_hole: int = 2
    control_lookback: int = 40              # frames before a control onset searched for its touch
    touch_dv: float = 3.0                   # m/s velocity change within the touch window
    touch_turn_deg: float = 30.0
    touch_turn_speed: float = 3.0           # m/s; a turn only counts at speed
    touch_dist: float = 2.0                 # m; a player must be this close
    touch_half_window: int = 3              # frames either side of the candidate touch
    touch_nms: int = 5

    # -- L2 ball: gaps -------------------------------------------------------------------------
    gap_short: int = 8                      # frames; interpolated
    gap_long: int = 75                      # frames; beyond this the position is `unknown`
    gap_flight_speed: float = 8.0           # m/s; a gap bounded by a fast ball is a flight
    gap_carrier_switch: float = 8.0         # metres-equivalent cost of switching carrier
    sigma_observed: float = 0.3             # m

    # -- optional video detections (possession/video) -------------------------------------------
    video_conf_min: float = 0.0             # extra floor on top of the validated threshold in the file

    # -- L2 ball: out of play / stationary -----------------------------------------------------
    oob_margin: float = 0.5                 # m past a line
    oob_persist: int = 10                   # frames (airborne balls project long)
    oob_persist_unobserved: int = 20
    oob_extrap_near: float = 4.0            # m from a line where a lost ball may have gone out
    oob_extrap_speed: float = 4.0           # m/s outward, at the last reliable sighting
    oob_extrap_min_gap: int = 25            # frames the ball must stay lost (1 s) to call it out
    stationary_radius: float = 0.6          # m; positions over the window stay inside this
    stationary_frames: int = 12             # 0.5 s

    # -- L3 / L4 emission (rules) --------------------------------------------------------------
    emit_sigma_d: float = 2.0               # m, proximity kernel
    emit_sigma_v: float = 3.0               # m/s, relative-speed kernel
    emit_sigma_max: float = 6.0             # ball position sigma at which confidence hits zero
    emit_kappa_prox: float = 2.0            # logit weight of proximity
    emit_kappa_ctrl: float = 2.0            # logit weight of a control episode
    emit_kappa_dead: float = 6.0            # logit weight of stoppage evidence
    emit_dead_bias: float = 1.5             # dead logit with no evidence is -bias
    emit_calm_weight: float = 0.6           # stoppage evidence from "play has stopped" (calm.py)
    calm_smooth: int = 25                   # frames (+/-) of smoothing of the collective speed
    calm_min_frames: int = 50               # calm must last 2 s before it counts
    emit_tag_weight: float = 1.0            # scales the calibrated Smart Tagging prior
    emit_shape_weight: float = 1.0          # scales the player-shape team logit (shape.py); 0 switches it off
    shape_min_accuracy: float = 0.75        # used only if its held-out accuracy on ball-seen frames beats this
    tag_pool: str = "max"                   # "max": overlapping codes are one situation; "sum": independent
    tag_calib: str = "interval"             # "interval": per tag interval vs control episodes; "silver": per frame
    tag_ramp: int = 50                      # frames over which a tag interval fades in and out
    tag_silver_dist: float = 0.5            # silver frames: nearest player within this
    tag_silver_opp: float = 3.0             # ... and no opponent within this
    tag_weight_cap: float = 2.0

    # -- L3 candidates -------------------------------------------------------------------------
    cand_nms: int = 5
    cand_grid: int = 50                     # fallback grid everywhere
    cand_grid_gap: int = 25                 # finer grid inside long ball gaps
    cand_long_gap: int = 50
    rest_min_frames: int = 25               # ball at rest this long before a restart counts
    rest_leave_speed: float = 3.0           # m/s
    taker_dist: float = 1.5                 # m; the restart taker stands at the ball
    taker_opp_clear: float = 2.0            # m; no opponent nearer (Law 15: 2 m; Laws 13, 16, 17: 9.15 m)
    taker_window: int = 25                  # frames before the restart that are read (1 s)
    taker_min_share: float = 0.8            # share of those frames that must agree on the taker
    goal_mouth_half: float = 4.2            # m; half the goal (3.66 m) plus tracking slack
    goal_kickoff_window: int = 4500         # frames from a goal crossing to its kick-off (3 min, VAR)

    # -- audio: the referee's whistle (SITUATION.md section 10.3) -------------------------------
    audio_rate: int = 16000                 # Hz; one hop of 640 samples is exactly one 25 fps video frame
    audio_win: int = 2048                   # samples per spectrum (128 ms)
    whistle_lo_hz: float = 2000.0
    whistle_hi_hz: float = 4500.0
    whistle_band_ratio: float = 0.35        # share of the frame's energy that lies in the whistle band
    whistle_peak_ratio: float = 0.5         # share of the band's energy within +/- whistle_peak_hz of its peak
    whistle_peak_hz: float = 150.0
    whistle_min_s: float = 0.2              # shorter is a click, not a whistle
    whistle_max_s: float = 3.0              # longer is the crowd or a stadium tone
    whistle_hole_s: float = 0.12            # a blip shorter than this inside a whistle does not split it
    whistle_floor_db: float = -55.0         # quieter than this (re full scale) is silence, not a whistle

    whistle_delay_frames: int = 4           # sound lags the picture (343 m/s over ~50 m); measure and replace
    whistle_corroborate_s: float = 5.0      # a whistle counts as a stop only if a stoppage run starts this soon after
    dec_bonus_stop_audio: float = 3.0       # below dec_bonus_stop_whistle (5): a whistle also starts restarts

    # -- goalkeeper control (SITUATION.md S13) ----------------------------------------------------
    gk_box_depth: float = 16.5              # m from the goal line: a keeper's touch is in the penalty area
    gk_catch_frames: int = 25               # a keeper holds the ball at least this long to have caught it
    gk_parry_factor: float = 0.3            # turnover bonus scale for a short keeper touch (parry, punch)

    # -- L5 decoder ----------------------------------------------------------------------------
    dec_alpha: float = 0.12                 # discount for correlated frames
    dec_boundary_penalty: float = 4.0
    dec_bonus_turnover: float = 8.0
    dec_bonus_touch: float = 5.0
    dec_bonus_stop_oob: float = 10.0
    dec_bonus_stop_still: float = 8.0
    dec_bonus_stop_whistle: float = 5.0
    dec_bonus_restart_leave: float = 10.0
    dec_bonus_restart_oob_end: float = 4.0
    dec_restart_side: float = 3.0           # bonus for "opponent of the last toucher" restarts
    dec_restart_taker: float = 3.0          # restart team seen at the ball; = dec_restart_side, so it swaps a guess for an observation without making restarts easier
    dec_kickoff_bonus: float = 20.0
    dec_dur_median_alive: float = 9.0       # seconds, log-normal
    dec_dur_sigma_alive: float = 1.1
    dec_dur_median_dead: float = 12.0
    dec_dur_sigma_dead: float = 0.8
    dec_max_segment: int = 6000             # 4 minutes
    dec_min_node_gap: int = 5
    dec_kickoff_team_bonus: float = 6.0     # Law 8: the halves are kicked off by different teams
    turnover_offset: int = 0                # calibrated on hand-labelled windows, not here
    stop_offset: int = 0
    restart_offset: int = 0

    # -- L6 output -----------------------------------------------------------------------------
    mbr_samples: int = 200
    mbr_thetas: tuple = (0.0, 0.2, 0.3, 0.4, 0.5)
    seed: int = 0

    def digest(self) -> str:
        """Short hash used as the cache key, so changing any value invalidates the cache."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def replace(self, **changes) -> "Config":
        return dataclasses.replace(self, **changes)

    def with_overrides(self, items) -> "Config":
        """Apply ["key=value", ...] from the command line (--set), typed like the field.

        Every override changes digest(), so the registry and the cache tell experiments apart.
        """
        fields = {f.name: f for f in dataclasses.fields(self)}
        changes = {}
        for item in items or ():
            key, sep, text = item.partition("=")
            key = key.strip()
            if not sep or key not in fields:
                raise ValueError("--set {!r}: expected key=value with key one of the Config fields".format(item))
            current = getattr(self, key)
            if isinstance(current, bool):
                value = text.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(text)
            elif isinstance(current, float):
                value = float(text)
            elif isinstance(current, tuple):
                value = tuple(float(v) for v in text.split(",") if v.strip())
            else:
                value = text.strip()
            changes[key] = value
        return self.replace(**changes) if changes else self


DEFAULT = Config()
