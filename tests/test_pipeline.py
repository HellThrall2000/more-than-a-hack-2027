import numpy as np
import pytest

from possession import harness
from possession.config import DEFAULT as cfg
from possession.ingest import MatchInputs
from possession.policy import metrics, render, validate
from possession.run import predict
from possession.simulate_atd import corrupt, to_inputs
from possession.synthetic import make_match
from starter import baseline

pm = metrics()


@pytest.fixture(scope="module")
def match():
    return make_match(6000, seed=2)


def nearest_codes(inp):
    tr = inp.tracking
    d = np.hypot(tr.players_xy[:, :, 0] - tr.ball_xy[:, None, 0],
                 tr.players_xy[:, :, 1] - tr.ball_xy[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    return np.where(np.isnan(tr.ball_xy[:, 0]), -1, inp.slot_team[d.argmin(axis=1)]).astype(np.int8)


def test_synthetic_pipeline_is_valid_and_beats_the_reference_baseline():
    wins, ours_all, base_all = 0, [], []
    for seed in (0, 1, 2):
        m = make_match(6000, seed=seed)
        inp = to_inputs(m, cfg, seed=seed)
        labels = predict(inp, cfg)
        assert validate(labels, inp.n_frames) is None
        truth = pm.label_codes(np.array(["H", "A", "D"])[m.truth])
        ours = pm.evaluate(truth, pm.label_codes(labels))["spell_f1_mean"]
        base = max(pm.evaluate(truth, pm.label_codes(baseline.smooth(nearest_codes(inp), w)))["spell_f1_mean"]
                   for w in (50, 125))
        ours_all.append(ours)
        base_all.append(base)
        wins += ours > base
    assert wins == 3, (ours_all, base_all)
    assert np.mean(ours_all) > 0.5


def test_pipeline_is_deterministic(match):
    inp = to_inputs(match, cfg, seed=2)
    a, b = predict(inp, cfg), predict(inp, cfg)
    assert (a == b).all()


def test_blocks_kickoffs_and_ungraded_frames(match):
    """Two graded blocks: X outside them, D until the second-half kick-off, alive after it."""
    n = len(match.ball_xy)
    inp = to_inputs(match, cfg, seed=2)
    inp = MatchInputs(inp.tracking, inp.slot_team, [(1, 500, 3000), (2, 3500, n)], inp.tags,
                      kickoffs={1: 500, 2: 4100}, n_frames=n)
    labels = predict(inp, cfg)
    assert (labels[:500] == "X").all() and (labels[3000:3500] == "X").all()
    assert (labels[3500:4100] == "D").all()
    assert labels[4100] in ("H", "A") and labels[500] in ("H", "A")
    assert not np.isin(labels[500:3000], ["X"]).any() and not np.isin(labels[3500:], ["X"]).any()


def test_a_missing_ball_alone_never_creates_a_possession_change(match):
    """With no ball at all the system must not invent team flips from nothing.

    Stoppages may still be found -- the players stopping is evidence that needs no ball (calm.py)
    -- but the D it produces must be real stoppages, and no H <-> A change may appear.
    """
    inp = to_inputs(match, cfg, seed=2, with_tags=False)
    inp.tracking.ball_xy[:] = np.nan
    labels = predict(inp, cfg)
    codes = pm.label_codes(labels)
    flips = [t for t in pm.transitions(codes, pm.valid_runs(codes)) if t[1] < 2 and t[2] < 2]
    assert flips == []
    dead = codes == 2
    if dead.any():
        assert (match.truth[dead] == 2).mean() > 0.7


def test_emission_gives_no_team_evidence_where_the_ball_is_unknown(match):
    inp = to_inputs(match, cfg, seed=2, with_tags=False)
    _, d = predict(inp, cfg, details=True)
    ball, logp, fs = d["ball"], d["logp"], d["features"]
    blind = (np.nan_to_num(ball.sigma, nan=99) >= cfg.emit_sigma_max) & (ball.ctrl_team < 0) \
        & (ball.carrier_team < 0)
    assert blind.sum() > 50
    assert np.allclose(logp[blind, 0], logp[blind, 1], atol=1e-6)
    assert np.allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-4)


def test_public_ground_truth_scores_one_and_baselines_match_the_published_numbers():
    gt = harness.resolve_gt("public")
    labels = pm.read_label_csv(gt)["label"].values
    assert harness.score(labels, "public")["spell_f1_mean"] == pytest.approx(1.0)
    for name, expected in (("baseline_smooth50", 0.3256), ("baseline_smooth125", 0.4242)):
        path = "evaluation/results/{}.csv".format(name)
        from possession import ROOT
        r = harness.score_csv("{}/{}".format(ROOT, path), "public")
        assert r["spell_f1_mean"] == pytest.approx(expected, abs=5e-4)       # README.md's figures


def test_render_marks_everything_outside_the_blocks_as_abstention():
    out = render(10, [[(2, 5, 0), (5, 8, 2)]])
    assert "".join(out) == "XXHHHDDDXX"
    assert validate(out) is None and validate(out, 11) is not None


def test_diagnose_report_runs_and_contains_only_aggregates(match):
    from possession.diagnose import report
    inp = to_inputs(match, cfg, seed=2)
    labels, details = predict(inp, cfg, details=True)
    gt = pm.label_codes(np.array(["H", "A", "D"])[match.truth])
    text = report(labels, details, gt, cfg)
    for needle in ("confusion", "emission argmax", "candidate boundaries", "transitions at", "ball layer"):
        assert needle in text
    assert len(text) < 5000
