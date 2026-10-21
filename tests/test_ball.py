import numpy as np

from possession.ball import clean, events, gaps, smooth
from possession.config import DEFAULT as cfg


def line(n, speed=0.2):
    t = np.arange(n, dtype=np.float32)
    return np.column_stack([10 + speed * t, np.full(n, 30.0)]).astype(np.float32)


def test_clean_rejects_an_isolated_outlier():
    ball = line(100)
    ball[50] = [90.0, 60.0]
    out = clean.clean(ball, cfg)
    assert np.isnan(out[50]).all()
    assert (~np.isnan(out[:, 0])).sum() == 99


def test_clean_recovers_when_the_anchor_was_the_outlier():
    ball = line(40)
    ball[0] = [90.0, 60.0]                 # the first observation is wrong, the rest is the ball
    out = clean.clean(ball, cfg)
    assert np.isnan(out[0]).all()
    assert (~np.isnan(out[:, 0])).sum() >= 38


def test_savgol_is_exact_on_a_quadratic_including_the_edges():
    t = np.arange(30, dtype=float)
    y = 3 + 2 * t + 0.5 * t ** 2
    sm, dv = smooth.smooth_series(y, cfg.sg_window, cfg.sg_order)
    np.testing.assert_allclose(sm, y, atol=1e-8)
    np.testing.assert_allclose(dv, 2 + t, atol=1e-8)


def reconstruct(ball, players=None, mask=()):
    n = len(ball)
    masked = ball.copy()
    for a, b in mask:
        masked[a:b] = np.nan
    pos, vel, observed, interp = smooth.smooth_ball(masked, cfg)
    players = np.full((n, 4, 2), np.nan, dtype=np.float32) if players is None else players
    return gaps.reconstruct(pos, vel, observed & ~np.isnan(pos[:, 0]), interp, players,
                            np.array([0, 0, 1, 1], dtype=np.int8), [(1, 0, n)], cfg)


def test_short_gap_is_interpolated_with_small_error():
    ball = line(120)
    g = reconstruct(ball, mask=[(60, 66)])
    assert (g.src[60:66] == gaps.SRC_INTERP).all()
    assert np.abs(g.xy[60:66] - ball[60:66]).max() < 0.1
    assert g.gap_len[60] == 6 and g.since_seen[60] == 1 and g.until_seen[65] == 1


def test_fast_ball_gap_is_a_flight():
    ball = line(120, speed=0.6)            # 15 m/s
    g = reconstruct(ball, mask=[(50, 75)])
    assert (g.src[50:75] == gaps.SRC_FLIGHT).all()
    assert np.abs(g.xy[50:75] - ball[50:75]).max() < 0.5


def test_carried_gap_follows_the_player_who_has_the_ball():
    n = 120
    ball = line(n, speed=0.1)              # 2.5 m/s: a dribble
    players = np.full((n, 4, 2), 80.0, dtype=np.float32)
    players[:, 2] = ball + np.array([0.4, 0.0], dtype=np.float32)      # slot 2 (team 1) runs with it
    g = reconstruct(ball, players, mask=[(40, 70)])
    assert (g.src[40:70] == gaps.SRC_CARRIED).all()
    assert (g.carrier_team[40:70] == 1).all()
    assert np.abs(g.xy[40:70] - ball[40:70]).max() < 0.6


def test_gap_at_the_start_of_a_block_is_unknown_not_invented():
    g = reconstruct(line(100), mask=[(0, 20)])
    assert (g.src[:20] == gaps.SRC_UNKNOWN).all() and g.sigma[:20].min() > 3.0


def test_out_of_play_needs_persistence():
    n = 200
    est = np.column_stack([np.full(n, 50.0), np.full(n, 30.0)])
    est[100:130, 1] = -2.0                 # 30 frames over the touchline
    est[150:153, 1] = -2.0                 # 3 frames: a lob projected long, not a stoppage
    src = np.full(n, gaps.SRC_OBSERVED, dtype=np.int8)
    oob, _ = events.out_of_play(est, src, np.full(n, 0.3), np.zeros((n, 2)), cfg)
    assert oob[100:130].all() and not oob[150:153].any() and not oob[:100].any()


def test_stationary_ball_is_found_by_displacement_not_by_jittery_speed():
    n = 200
    rng = np.random.RandomState(0)
    est = np.full((n, 2), 40.0) + rng.randn(n, 2) * 0.05
    est[100:, 0] += 0.2 * np.arange(100)           # then it moves at 5 m/s
    src = np.full(n, gaps.SRC_OBSERVED, dtype=np.int8)
    _, still = events.out_of_play(est, src, np.full(n, 0.3), np.zeros((n, 2)), cfg)
    assert still[:90].all() and not still[110:].any()
