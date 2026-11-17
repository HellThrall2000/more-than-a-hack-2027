import re

import numpy as np

from possession.config import DEFAULT as cfg
from possession.policy import metrics
from possession.run import predict
from possession.semantics import report, signed_nearest
from possession.simulate_atd import to_inputs
from possession.synthetic import make_match

pm = metrics()


def test_signed_nearest_sign_ties_and_search_window():
    out = signed_nearest([10, 50], [12, 30, 49, 200, 8], search=75)
    assert out[0] == -2 and out[2] == 1 and out[4] == 2
    assert out[1] == -20                    # equidistant: the earlier event wins
    assert np.isnan(out[3])                 # nothing within 75 frames
    assert np.isnan(signed_nearest([], [5])).all()


def test_report_counts_every_true_boundary_on_a_synthetic_match():
    m = make_match(6000, seed=1)
    inp = to_inputs(m, cfg, seed=1)
    _, details = predict(inp, cfg, details=True)
    gt = pm.label_codes(np.array(["H", "A", "D"])[m.truth])
    text = report(gt, details, inp.tags, cfg)

    tr = pm.transitions(gt, pm.valid_runs(gt))
    n_stop = sum(b == 2 for _, a, b in tr)
    n_restart = sum(a == 2 for _, a, b in tr)
    n_turn = sum(a < 2 and b < 2 for _, a, b in tr)
    assert "Q1  alive -> D  (n={})".format(n_stop) in text
    assert "Q2  D -> alive  (n={})".format(n_restart) in text
    assert "Q3  H <-> A  (n={})".format(n_turn) in text
    assert n_stop > 0 and n_turn > 0
    # every "found k/n" line is bounded by its n
    for k, n in re.findall(r"found\s+(\d+)/(\d+)", text):
        assert int(k) <= int(n)
    # the synthetic tags put a SET PIECES interval on every stoppage, all typed Throw-in
    assert "Throw-in" in text
