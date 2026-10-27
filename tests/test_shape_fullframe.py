"""Player-shape team model (shape.py) and the full-frame detector's pure parts (linking, self-check)."""

import numpy as np
import pytest

from possession.config import DEFAULT as cfg


def _pushing_teams(n=20000, seed=0):
    """Home pushes up and spreads when it has the ball, away when it has it."""
    rng = np.random.RandomState(seed)
    owner = (np.arange(n) // 500) % 2                         # 20 s spells, alternating
    xy = np.zeros((n, 22, 2), dtype=np.float32)
    base_x = np.r_[np.linspace(10, 45, 11), np.linspace(60, 95, 11)]
    shift = np.where(owner == 0, 12.0, -12.0)[:, None]        # +x is home's attack
    xy[:, :, 0] = base_x[None, :] + shift + rng.randn(n, 22) * 2.0
    xy[:, :, 1] = rng.uniform(5, 63, (n, 22))
    slot_team = np.repeat([0, 1], 11).astype(np.int8)
    return xy, slot_team, owner


def test_shape_model_learns_who_pushes_up_and_beats_chance_on_held_out_blocks():
    pytest.importorskip("lightgbm")
    from possession.shape import team_logit
    xy, slot_team, owner = _pushing_teams()
    ctrl = np.where(np.arange(len(owner)) % 3 == 0, owner, -1).astype(np.int8)    # control seen on a third
    logit, rep = team_logit(xy, slot_team, {1: (1, -1)}, [(1, 0, len(owner))], ctrl, cfg)
    assert rep["used"] and rep["held_out_accuracy"] > 0.9
    assert ((logit > 0) == (owner == 0)).mean() > 0.9


def test_shape_model_stays_silent_with_too_few_labels():
    pytest.importorskip("lightgbm")
    from possession.shape import team_logit
    xy, slot_team, owner = _pushing_teams(3000)
    ctrl = np.full(len(owner), -1, dtype=np.int8)
    logit, rep = team_logit(xy, slot_team, {1: (1, -1)}, [(1, 0, len(owner))], ctrl, cfg)
    assert not rep["used"] and not logit.any()


def test_link_follows_the_track_not_the_brightest_false_positive():
    from possession.video.fullframe import link
    frames = np.arange(5)
    cand = np.array([[[50.0, 30.0], [10.0, 10.0], [np.nan, np.nan]]] * 5)
    for i in range(5):
        cand[i, 0] += [0.4 * i, 0.0]                           # the real ball drifts slowly
    conf = np.array([[0.9, 0.2, 0.0], [0.3, 0.8, 0.0], [0.3, 0.8, 0.0], [0.3, 0.8, 0.0], [0.3, 0.8, 0.0]])
    xy, c = link(frames, cand, conf, 0.25, cfg)
    assert np.allclose(xy[:, 1], 30.0) and (c > 0).all()        # stays on the ball, not the sock at (10, 10)


def test_self_check_picks_the_lowest_threshold_with_precision_0_9():
    from possession.video.fullframe import self_check
    truth = np.zeros((100, 2))
    cand = np.zeros((100, 3, 2))
    conf = np.zeros((100, 3))
    conf[:, 0] = np.r_[np.full(80, 0.6), np.full(20, 0.12)]
    cand[80:, 0] = [30.0, 30.0]                                # the weak ones are wrong
    rows, thr = self_check(cand, conf, truth)
    assert thr == 0.15 and next(r for r in rows if r["thr"] == 0.15)["coverage"] == 0.8


def test_shape_evidence_moves_belief_between_teams_but_never_into_or_out_of_dead():
    from possession.emission_rules import emit
    from possession.features import FeatureSet
    n = 4
    names = ["sigma", "d_h", "d_a", "rv_h", "rv_a", "ctrl_h", "ctrl_a", "carrier_h", "carrier_a", "stationary",
             "walk_frac", "oob", "calm", "ball_speed", "src"]
    F = np.zeros((n, len(names)), dtype=np.float32)
    F[:, names.index("sigma")] = 2.0
    F[:, names.index("d_h")] = F[:, names.index("d_a")] = 5.0
    F[:, names.index("src")] = 4.0                                  # ball unknown
    fs = FeatureSet(F, names)
    tagf = {"poss": np.zeros((n, 2)), "setpiece": np.zeros(n), "goal": np.zeros(n)}
    base = np.exp(emit(fs, tagf, cfg))
    moved = np.exp(emit(fs, tagf, cfg, shape=np.full(n, 3.0, dtype=np.float32)))
    assert np.allclose(base[:, 2], moved[:, 2], atol=1e-6)          # P(dead) unchanged
    assert (moved[:, 0] > base[:, 0]).all()                          # belief moved to home


def test_shape_evidence_is_ignored_where_the_ball_is_seen_or_in_flight():
    from possession.emission_rules import emit
    from possession.features import FeatureSet
    names = ["sigma", "d_h", "d_a", "rv_h", "rv_a", "ctrl_h", "ctrl_a", "carrier_h", "carrier_a", "stationary",
             "walk_frac", "oob", "calm", "ball_speed", "src"]
    F = np.zeros((2, len(names)), dtype=np.float32)
    F[:, names.index("d_h")] = F[:, names.index("d_a")] = 5.0
    F[1, names.index("src")] = 3.0                                     # row 0 observed, row 1 in flight
    fs = FeatureSet(F, names)
    tagf = {"poss": np.zeros((2, 2)), "setpiece": np.zeros(2), "goal": np.zeros(2)}
    assert np.allclose(emit(fs, tagf, cfg), emit(fs, tagf, cfg, shape=np.full(2, 3.0, dtype=np.float32)))


def test_filters_drop_a_spare_ball_off_the_pitch_and_a_painted_spot_that_never_moves():
    from possession.video.fullframe import plausible
    n = 400
    cand = np.full((n, 3, 2), np.nan)
    conf = np.zeros((n, 3))
    cand[:, 0] = np.c_[np.linspace(20, 80, n), np.full(n, 30.0)]     # the match ball, moving
    cand[:, 1] = [50.0, -6.0]                                           # spare ball beyond the touchline
    cand[:, 2] = [11.0, 34.0]                                           # penalty spot, every frame
    conf[:] = [0.4, 0.9, 0.8]
    out, removed = plausible(cand, conf, cfg)
    assert (out[:, 0] == 0.4).all() and not out[:, 1:].any()
    assert removed == {"off_pitch": n, "static": n}


def test_tracklets_keep_a_steady_ball_and_drop_a_flickering_boot():
    from possession.video.fullframe import tracklets
    n = 40
    frames = np.arange(n)
    cand = np.full((n, 2, 2), np.nan)
    conf = np.zeros((n, 2))
    cand[:, 0] = np.c_[30 + 0.4 * np.arange(n), np.full(n, 20.0)]   # the ball, 10 m/s, every frame
    conf[:, 0] = 0.3
    for i in range(0, n, 6):                                         # a boot, one frame in six, far away
        cand[i, 1] = [80.0 - i, 50.0]
        conf[i, 1] = 0.9
    xy, c = tracklets(frames, cand, conf, 0.2, cfg)
    assert (c > 0).all() and np.allclose(xy[:, 1], 20.0)


def test_tracklets_drop_everything_too_short_to_be_a_ball_track():
    from possession.video.fullframe import tracklets
    frames = np.arange(5)
    cand = np.zeros((5, 1, 2))
    conf = np.full((5, 1), 0.9)
    xy, c = tracklets(frames, cand, conf, 0.2, cfg)
    assert not c.any()


def test_our_atd_copies_every_line_and_fills_only_missing_balls(tmp_path):
    from possession.video.write_atd import verify, write
    raw = tmp_path / "a_atd.txt"
    raw.write_text("1:0.10000,0.20000,1.50;NaN,NaN,NaN:0.50000,0.50000,3.20\n"
                   "2:0.10100,0.20000,1.40;0.90000,0.40000,2.00:NaN,NaN,NaN\n"
                   "3:0.10200,0.20000,1.30;0.90100,0.40000,2.10:NaN,NaN,NaN\n")
    out = tmp_path / "b_atd.txt"
    det = np.full((3, 2), np.nan)
    conf = np.zeros(3)
    det[1], conf[1] = [52.65, 34.0], 0.8          # frame 2 gets a detection; frame 3 stays missing
    assert write(str(raw), str(out), det, conf, 0.5, (105.3, 68.0), 25) == 1
    lines = out.read_text().splitlines()
    assert lines[0] == raw.read_text().splitlines()[0] and lines[2] == raw.read_text().splitlines()[2]
    assert lines[1].split(":")[2].startswith("0.50000,0.50000,")
    rep = verify(str(raw), str(out), 3, (105.3, 68.0))
    assert rep["players_identical"] and rep["original_balls_identical"] and rep["frames_filled"] == 1
