"""The fixes of PLAN.md "Next plan of action" step 1, each on a case built to need it."""

import numpy as np
import pytest

from possession import labels as lab
from possession.ball.events import extrapolated_crossings
from possession.calm import calm_probability, fit_two_states
from possession.config import DEFAULT as cfg
from possession.policy import metrics
from possession.run import predict
from possession.simulate_atd import to_inputs
from possession.synthetic import make_match
from possession.tags import build as build_tags, kickoff_teams
from possession.util import runs

pm = metrics()
BLOCKS = [(1, 11185, 90652), (2, 110808, 190198)]


# -- labelling tool ------------------------------------------------------------------------------

def test_change_points_in_every_unit_build_the_window_and_x_elsewhere():
    text = """
    # comment line
    window 11185 11300
    11185 A
    00:07:30.00 D        # video time: 450 s -> frame 11250
    1H 00:04.6 H         # match clock: kick-off + 4.6 s -> 11185 + 115 = 11300
    """
    window, points = lab.parse(text)
    assert window == (11185, 11300) and [f for f, _ in points] == [11185, 11250, 11300]
    out = lab.build(window, points, BLOCKS, 199319)
    assert (out[11185:11250] == "A").all() and (out[11250:11300] == "D").all() and out[11300] == "H"
    assert out[11184] == "X" and out[11301] == "X" and len(out) == 199319


@pytest.mark.parametrize("text, message", [
    ("11185 A", "window"),
    ("window 11185 11300\n11186 A", "first change point"),
    ("window 11185 11300\n11185 A\n11250 D\n11240 H", "increasing"),
    ("window 11185 11300\n11185 Q", "H, A or D"),
    ("window 90000 91000\n90000 A", "one graded half"),
])
def test_bad_change_point_files_are_refused_with_a_reason(text, message):
    with pytest.raises(ValueError, match=message):
        lab.build(*lab.parse(text), BLOCKS, 199319)


# -- kick-off team -------------------------------------------------------------------------------

def _kick(team, half):
    return {"code": "SET PIECES", "start_frame": 0, "end_frame": 1,
            "tags": {"Team": team, "Half": half, "Type": "Kick Off (Start)"}}


def test_kickoff_team_comes_from_the_tags_and_law_8_fills_the_other_half():
    assert kickoff_teams([_kick("Atletic Club", "1st Half")]) == {1: 1, 2: 0}
    assert kickoff_teams([_kick("Barcelona", "2nd Half")]) == {2: 0, 1: 1}
    assert kickoff_teams([_kick("Barcelona", "1st Half"), _kick("Barcelona", "2nd Half")]) == {}
    assert kickoff_teams([]) == {}


def test_kickoff_team_decides_an_otherwise_ambiguous_first_possession():
    from possession.candidates import Candidates
    from possession.decode import decode_match
    n = 400
    logp = np.full((n, 3), np.log(1 / 3.0), dtype=np.float32)
    c = Candidates(np.array([200]), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros((1, 2)))
    for team in (0, 1):
        [(_, best, _)] = decode_match(logp, c, [(1, 0, n)], {1: 0}, cfg, kickoff_team={1: team})
        assert best[0][2] == team


# -- extrapolated line crossing ------------------------------------------------------------------

def test_a_ball_lost_while_heading_out_is_out_from_the_projected_crossing():
    n = 200
    est = np.full((n, 2), np.nan)
    vel = np.zeros((n, 2))
    reliable = np.zeros(n, dtype=bool)
    reliable[:50] = True
    est[:50] = [50.0, 2.0]                      # 2 m from the y = 0 touchline ...
    vel[:50] = [0.0, -10.0]                     # ... moving out at 10 m/s: crosses in 0.2 s = 5 frames
    reliable[150:] = True
    est[150:] = [50.0, 0.0]
    out = extrapolated_crossings(est, reliable, vel, cfg)
    assert out[54:150].all() and not out[:54].any() and not out[150:].any()
    vel[:50] = [0.0, 10.0]                      # moving back into the field: never out
    assert not extrapolated_crossings(est, reliable, vel, cfg).any()
    reliable[60:] = True                        # seen again after 10 frames: too short to call out
    assert not extrapolated_crossings(est, reliable, vel, cfg).any()


# -- calm ---------------------------------------------------------------------------------------

def test_two_state_fit_finds_walking_and_playing_and_refuses_one_state():
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.normal(np.log(1.0), 0.15, 3000), rng.normal(np.log(3.0), 0.15, 7000)])
    mu, var, w = fit_two_states(x)
    assert np.exp(mu[0]) == pytest.approx(1.0, rel=0.1) and np.exp(mu[1]) == pytest.approx(3.0, rel=0.1)
    assert w[0] == pytest.approx(0.3, abs=0.05)
    speed = np.full((5000, 2), 2.5) + rng.normal(0, 0.05, (5000, 2))      # one state only
    assert calm_probability(speed, [(1, 0, 5000)], cfg).max() == 0.0


def test_stoppages_are_found_when_the_ball_is_hidden_in_them():
    m = make_match(6000, seed=4)
    inp = to_inputs(m, cfg, seed=4)
    for a, b in zip(*runs(m.truth == 2)):
        inp.tracking.ball_xy[a:max(a, b - 25)] = np.nan
    truth = pm.label_codes(np.array(["H", "A", "D"])[m.truth])
    on = pm.evaluate(truth, pm.label_codes(predict(inp, cfg)))["per_class"]["D"]
    off = pm.evaluate(truth, pm.label_codes(predict(inp, cfg.replace(emit_calm_weight=0.0))))["per_class"]["D"]
    assert on["recall"] > off["recall"] + 0.2 and on["precision"] > 0.8


# -- Smart Tagging pooling and --set -------------------------------------------------------------

def test_overlapping_tags_count_once_unless_asked_to_sum():
    tags = [{"code": c, "start_frame": 0, "end_frame": 300, "tags": {"Team": "Barcelona"}}
            for c in ("BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")]
    w = {"BALL IN FINAL THIRD": 1.0, "BALL IN THE BOX": 1.5, "PLAYERS IN THE BOX": 0.5}
    pooled = build_tags(tags, 400, w, cfg)["poss"][150, 0]
    summed = build_tags(tags, 400, w, cfg.replace(tag_pool="sum"))["poss"][150, 0]
    assert pooled == pytest.approx(1.5) and summed == pytest.approx(3.0)


def test_set_overrides_are_typed_and_change_the_config_hash():
    c = cfg.with_overrides(["emit_tag_weight=0", "tag_pool=sum", "restart_offset=5"])
    assert c.emit_tag_weight == 0.0 and c.tag_pool == "sum" and c.restart_offset == 5
    assert c.digest() != cfg.digest() and cfg.with_overrides([]) is cfg
    with pytest.raises(ValueError):
        cfg.with_overrides(["no_such_field=1"])


# -- turnover on the controlling touch, kick-off team from the tracking --------------------------

def test_turnover_lands_on_the_new_teams_first_touch_before_control():
    from possession.candidates import _controlling_touch
    tf = np.array([100, 130, 150, 160])
    tt = np.array([0, 1, 1, 1])                 # home touches at 100, then away at 130, 150, 160
    assert _controlling_touch(tf, tt, 170, 1, cfg) == 130       # first away touch of the run
    assert _controlling_touch(tf, tt, 170, 0, cfg) == 170       # home's last touch is not its run
    assert _controlling_touch(tf, tt, 400, 1, cfg) == 400       # too far back: the onset itself


def test_kickoff_team_is_read_from_who_stands_on_the_centre_spot():
    from possession.restarts import kickoff_team_from_players, resolve_kickoff_team
    n, spot = 200, np.array([cfg.pitch_length / 2, cfg.pitch_width / 2])
    xy = np.full((n, 4, 2), np.nan, dtype=np.float32)
    slot_team = np.array([0, 0, 1, 1], dtype=np.int8)
    xy[:, 0] = spot + [0.5, 0.0]                # a home player on the spot
    xy[:, 1] = spot + [-20.0, 5.0]
    xy[:, 2] = spot + [10.0, 0.0]               # away outside the centre circle
    xy[:, 3] = spot + [25.0, -8.0]
    assert kickoff_team_from_players(xy, slot_team, {1: 100}, cfg) == {1: 0}
    xy[:, 2] = spot + [1.0, 0.0]                # both teams on the spot: no call
    assert kickoff_team_from_players(xy, slot_team, {1: 100}, cfg) == {}
    assert resolve_kickoff_team({}, {1: 0}) == {1: 0, 2: 1}
    assert resolve_kickoff_team({1: 1}, {1: 0, 2: 0}) == {1: 1, 2: 0}         # tags win, Law 8 fills
    assert resolve_kickoff_team({}, {1: 0, 2: 0}) == {}                       # contradicts Law 8
