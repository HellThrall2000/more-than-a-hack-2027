import numpy as np

from possession.audit import audit_arrays
from possession.ball import scorecard
from possession.config import DEFAULT as cfg
from possession.players import clean_players
from possession.simulate_atd import fit_gap_model, to_inputs
from possession.synthetic import make_match


def test_scorecard_reports_error_by_gap_length_and_source():
    m = make_match(9000, seed=4)
    inp = to_inputs(m, cfg, seed=4)
    tr = inp.tracking
    pl = clean_players(tr.players_xy, inp.slot_team, cfg)
    out = scorecard.run(tr.ball_xy, pl, inp.slot_team, inp.blocks, cfg, n_windows=80, seed=1)
    assert out["n_windows"] > 20 and out["n_frames"] > 300
    short = out["by_length"]["3-8"]
    assert short["median_m"] < 1.0                      # a few frames of a visible ball is easy
    longest = out["by_length"]["76-inf"] if "76-inf" in out["by_length"] else None
    if longest:
        assert longest["median_m"] > short["median_m"]  # and long gaps are not
    assert "observed" not in out["by_src"] or out["by_src"]["observed"]["median_m"] < 0.5


def test_audit_reports_aggregates_only():
    m = make_match(6000, seed=5)
    inp = to_inputs(m, cfg, seed=5)
    rep = audit_arrays(inp, cfg)
    assert 0.4 < rep["ball_visible_graded"] < 0.8 and rep["gap_count"] > 10
    assert 15 < rep["objects_per_graded_frame_mean"] <= 22
    import json
    assert len(json.dumps(rep)) < 4000                  # statistics, nothing frame-sized


def test_gap_model_roundtrip_reproduces_the_visibility_rate():
    from possession.simulate_atd import ball_mask
    m = make_match(6000, seed=6)
    inp = to_inputs(m, cfg, seed=6)
    model = fit_gap_model(inp.tracking.ball_xy)
    mask = ball_mask(60000, np.random.RandomState(0), model)
    real = (~np.isnan(inp.tracking.ball_xy[:, 0])).mean()
    assert abs(mask.mean() - real) < 0.1
