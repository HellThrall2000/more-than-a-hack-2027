"""Restart team rules (SITUATION.md section 6, S14): the team seen at the ball, and the kick-off
after a goal going to the team that conceded."""

import numpy as np

from possession.config import DEFAULT as cfg
from possession.restarts import conceding_team, kickoff_restart_team, restart_taker

LEFT_ATTACKS_RIGHT = (1, -1)        # home attacks towards +x, so home defends the x = 0 goal
MID_Y = cfg.pitch_width / 2.0


def _at_ball(n, home, away):
    """(n, 2) nearest-player distance to the ball per team, constant over time."""
    return np.tile(np.array([home, away], dtype=float), (n, 1))


def _oob_at(n, start, xy):
    """Ball out of play from `start` to the end, crossing at `xy`."""
    ball = np.full((n, 2), [50.0, MID_Y])
    oob = np.zeros(n, dtype=bool)
    ball[start:], oob[start:] = xy, True
    return ball, oob


# -- restart_taker -----------------------------------------------------------------------------

def test_throw_in_goes_to_the_team_whose_player_holds_the_ball():
    assert restart_taker(_at_ball(100, 0.5, 4.0), 100, cfg) == 0
    assert restart_taker(_at_ball(100, 6.0, 0.8), 100, cfg) == 1


def test_no_taker_when_an_opponent_stands_at_the_ball_too():
    assert restart_taker(_at_ball(100, 0.5, 1.0), 100, cfg) is None


def test_no_taker_when_nobody_is_at_the_ball():
    assert restart_taker(_at_ball(100, 3.0, 5.0), 100, cfg) is None


def test_no_taker_when_the_ball_is_mostly_unseen():
    at_ball = _at_ball(100, 0.5, 4.0)
    at_ball[80:95] = np.nan                 # 15 of the last 25 frames unseen: too few left
    assert restart_taker(at_ball, 100, cfg) is None


def test_taker_must_hold_most_of_the_last_second():
    at_ball = _at_ball(100, 0.5, 4.0)
    at_ball[75:90] = [4.0, 0.5]             # away at the ball for 15 of the 25 frames
    assert restart_taker(at_ball, 100, cfg) is None


# -- conceding_team ------------------------------------------------------------------------------

def test_ball_out_between_the_posts_names_the_team_defending_that_goal():
    ball, oob = _oob_at(1000, 500, [-1.0, MID_Y + 2.0])
    assert conceding_team(ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) == 0
    ball, oob = _oob_at(1000, 500, [cfg.pitch_length + 1.0, MID_Y])
    assert conceding_team(ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) == 1
    assert conceding_team(ball, oob, 900, (-1, 1), cfg) == 0          # ends swapped


def test_ball_out_wide_of_the_posts_or_over_a_touchline_is_not_a_goal():
    ball, oob = _oob_at(1000, 500, [-1.0, MID_Y + 8.0])
    assert conceding_team(ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) is None
    ball, oob = _oob_at(1000, 500, [40.0, -1.0])
    assert conceding_team(ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) is None


def test_a_crossing_too_long_before_the_kick_off_is_not_its_goal():
    n = cfg.goal_kickoff_window + 1000
    ball, oob = _oob_at(n, 100, [-1.0, MID_Y])
    oob[200:] = False
    assert conceding_team(ball, oob, n - 1, LEFT_ATTACKS_RIGHT, cfg) is None


# -- kickoff_restart_team (S14) -------------------------------------------------------------------

def test_kick_off_after_a_goal_goes_to_the_team_that_conceded():
    ball, oob = _oob_at(1000, 500, [-1.0, MID_Y])             # home conceded
    unseen = np.full((1000, 2), np.nan)
    assert kickoff_restart_team(unseen, ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) == 0
    seen_home = _at_ball(1000, 0.5, 9.5)
    assert kickoff_restart_team(seen_home, ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) == 0


def test_kick_off_with_crossing_and_taker_disagreeing_makes_no_call():
    ball, oob = _oob_at(1000, 500, [-1.0, MID_Y])             # home conceded ...
    seen_away = _at_ball(1000, 9.5, 0.5)                       # ... but away is on the spot
    assert kickoff_restart_team(seen_away, ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) is None


def test_kick_off_without_a_goal_crossing_follows_the_taker():
    ball = np.full((1000, 2), [50.0, MID_Y])
    oob = np.zeros(1000, dtype=bool)
    assert kickoff_restart_team(_at_ball(1000, 9.5, 0.5), ball, oob, 900, LEFT_ATTACKS_RIGHT, cfg) == 1


# -- goalkeeper touch (S13) and whistle corroboration -------------------------------------------

def _keeper_scene(keeper_x=5.0, hold=10):
    """Two teams of 3: home attacks +x, so its deepest player (the keeper) stands near x = 0."""
    from possession.config import DEFAULT as c
    n = 200
    players = np.full((n, 6, 2), np.nan, dtype=np.float32)
    players[:, 0] = [keeper_x, 34.0]           # home keeper
    players[:, 1] = [30.0, 20.0]
    players[:, 2] = [40.0, 40.0]
    players[:, 3] = [90.0, 34.0]               # away keeper
    players[:, 4] = [60.0, 20.0]
    players[:, 5] = [70.0, 40.0]
    slot = np.zeros((n, 2), dtype=np.int16)
    slot[:, 0], slot[:, 1] = 0, 3
    aux = {"players": players, "attack": {1: (1, -1)}, "slot_team": np.array([0, 0, 0, 1, 1, 1]), "slot": slot}
    return aux, [(1, 0, n)], c


def test_a_short_keeper_episode_is_a_save_and_a_long_one_is_a_catch():
    from possession.candidates import _keeper_touch_only
    aux, blocks, c = _keeper_scene()
    assert _keeper_touch_only(50, 60, 0, aux, blocks, c) is True            # 10 frames: parry
    assert _keeper_touch_only(50, 90, 0, aux, blocks, c) is False           # 40 frames: held


def test_an_outfield_player_or_a_keeper_outside_the_box_is_not_a_save():
    from possession.candidates import _keeper_touch_only
    aux, blocks, c = _keeper_scene()
    aux["slot"][:, 0] = 1                                                    # nearest is not the deepest
    assert _keeper_touch_only(50, 60, 0, aux, blocks, c) is False
    aux, blocks, c = _keeper_scene(keeper_x=25.0)                            # deepest, but 25 m out
    assert _keeper_touch_only(50, 60, 0, aux, blocks, c) is False


def test_away_keeper_defends_the_far_goal():
    from possession.candidates import _keeper_touch_only
    aux, blocks, c = _keeper_scene()
    assert _keeper_touch_only(50, 60, 1, aux, blocks, c) is True


def test_a_whistle_adds_a_stop_candidate_only_next_to_a_real_stoppage():
    from possession.candidates import build_candidates
    from possession.ball import build_ball_state
    from possession.features import build_features
    from possession.players import attack_direction, clean_players, descriptors
    from possession.simulate_atd import to_inputs
    from possession.synthetic import make_match
    from possession.util import runs
    m = make_match(6000, seed=3)
    inp = to_inputs(m, cfg, seed=3)
    tr, st, blocks = inp.tracking, inp.slot_team, inp.blocks
    players = clean_players(tr.players_xy, st, cfg)
    ball = build_ball_state(tr.ball_xy, players, st, blocks, cfg)
    fs, aux = build_features(ball, players, descriptors(players, tr.players_speed, st, cfg), st,
                             attack_direction(players, st, blocks), blocks, cfg)
    dead_start = int(runs(m.truth == 2)[0][0])
    far = [(blocks[0][1] + 50, blocks[0][1] + 75)]                          # nowhere near a stoppage
    near = [(dead_start, dead_start + 25)]
    base = build_candidates(ball, fs, aux, blocks, inp.kickoffs, cfg)
    with_far = build_candidates(ball, fs, aux, blocks, inp.kickoffs, cfg, far)
    with_near = build_candidates(ball, fs, aux, blocks, inp.kickoffs, cfg, near)
    assert with_far.stop.sum() == base.stop.sum()
    assert with_near.stop.sum() >= base.stop.sum()


def test_report_gives_totals_only_and_counts_a_tag_at_a_dead_stretch():
    from possession.report import summarise
    from possession.run import predict
    from possession.simulate_atd import to_inputs
    from possession.synthetic import make_match
    from possession.util import runs
    m = make_match(9000, seed=1)
    inp = to_inputs(m, cfg, seed=1)
    a = int(runs(m.truth == 2)[0][0])
    inp.tags = [{"code": "SET PIECES", "start_frame": a, "end_frame": a + 50, "tags": {}}]
    inp.whistles = [(a, a + 25)]
    _, details = predict(inp, cfg, details=True)
    out = summarise(inp, details, cfg)
    assert out["set_piece_tags"] == 1 and out["set_piece_tags_at_a_dead_stretch"] == 1
    assert out["whistles"] == 1 and out["goal_tags"] == 0 and out["goal_agree"] == 0
    assert all(isinstance(v, (int, float, type(None))) for v in out.values())
