import numpy as np

from possession import tags
from possession.config import DEFAULT as cfg


def tag(code, team, a, b):
    return {"code": code, "start_frame": a, "end_frame": b, "tags": {"Team": team, "Half": "1st Half"}}


def test_team_names_map_to_sides_and_na_is_unknown():
    assert tags.team_index("Barcelona") == 0
    assert tags.team_index("Atletic Club") == 1          # the data's own spelling
    assert tags.team_index("athletic club") == 1
    assert tags.team_index("N/A") is None and tags.team_index(None) is None


def test_defending_tags_credit_the_opponent_and_na_credits_nobody():
    assert tags.possessing_team(tag("BUILD UP", "Barcelona", 0, 10)) == 0
    assert tags.possessing_team(tag("DEFENDING IN MIDDLE THIRD", "Barcelona", 0, 10)) == 1
    assert tags.possessing_team(tag("BUILD UP", "N/A", 0, 10)) is None
    assert tags.possessing_team(tag("SET PIECES", "Barcelona", 0, 10)) is None


def test_calibration_trusts_a_code_that_agrees_with_the_silver_frames_and_not_one_that_disagrees():
    silver = np.full(2000, -1, dtype=np.int8)
    silver[100:400] = 0
    good = [tag("BUILD UP", "Barcelona", 0, 500)]
    bad = [tag("LONG BALL", "Atletic Club", 0, 500)]
    w_good, _ = tags.calibrate(good, silver, cfg)
    w_bad, _ = tags.calibrate(bad, silver, cfg)
    assert w_good["BUILD UP"] > 1.0 and w_bad["LONG BALL"] == 0.0
    w_few, _ = tags.calibrate(good, np.full(2000, -1, dtype=np.int8), cfg)
    assert w_few["BUILD UP"] == 0.0                    # no silver frames, no trust


def test_channels_fade_at_the_edges_and_split_by_kind():
    chan = tags.build([tag("BUILD UP", "Barcelona", 100, 400), tag("SET PIECES", "N/A", 500, 600),
                       tag("GOALS", "Barcelona", 700, 750)], 1000, {"BUILD UP": 2.0}, cfg)
    assert chan["poss"][100, 0] < chan["poss"][250, 0] <= 2.0 and chan["poss"][:, 1].sum() == 0
    assert chan["setpiece"][550] == 1.0 and chan["goal"][725] > 0 and chan["setpiece"][0] == 0
