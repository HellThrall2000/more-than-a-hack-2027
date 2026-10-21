"""The real-data path (load_match -> predict -> CSV), exercised on a miniature fake bundle.

The gated bundle cannot be used in tests, so this writes a few thousand synthetic frames in the
formats DATA.md describes -- FIFA/EPTS metadata XML, the raw `frame:players:ball` text, the
homography CSV, the SportsCode tagging XML -- and points games.py at it. It checks plumbing: that
the loaders, the team mapping, the frame alignment and the output contract all agree.
"""

import numpy as np
import pytest

import games
from possession.config import DEFAULT
from possession.ingest import load_match
from possession.policy import metrics, validate
from possession.run import predict
from possession.simulate_atd import corrupt, synthetic_tags
from possession.synthetic import make_match

pm = metrics()
N = 6000


def write_bundle(root, match, cfg):
    folder = root / "tactical"
    folder.mkdir(parents=True)
    stem = str(folder / "BAR-ATH_tactical")
    tr = corrupt(match, cfg, seed=3)

    teams = "".join('<Team id="T{}"><Name>{}</Name></Team>'.format(i + 1, name)
                    for i, name in enumerate(("Barcelona", "Athletic")))
    players = "".join('<Player teamId="T{}"/>'.format(int(t) + 1) for t in match.slot_team)
    open(stem + "_atd.xml", "w").write("<root>{}<Players>{}</Players></root>".format(teams, players))

    scale = np.array([cfg.pitch_length, cfg.pitch_width])
    with open(stem + "_atd.txt", "w") as handle:
        for f in range(N):
            groups = []
            for s in range(tr.players_xy.shape[1]):
                p = tr.players_xy[f, s]
                groups.append("NaN,NaN,NaN" if np.isnan(p[0]) else "{:.5f},{:.5f},{:.2f}".format(
                    p[0] / scale[0], p[1] / scale[1], tr.players_speed[f, s]))
            b = tr.ball_xy[f]
            ball = "NaN,NaN,NaN" if np.isnan(b[0]) else "{:.5f},{:.5f},0.0".format(b[0] / scale[0], b[1] / scale[1])
            handle.write("{}:{}:{}\n".format(f + 1, ";".join(groups), ball))

    with open(stem + "_homography.csv", "w") as handle:
        handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for f in range(N):
            handle.write("{},1800,0,0,0,1000,0,0,0,1,0.2,1\n".format(f))

    inst = "".join(
        "<instance><start>{}</start><end>{}</end><code>{}</code><label><group>Team</group><text>{}</text></label>"
        "<label><group>Half</group><text>{}</text></label></instance>".format(
            t["start_frame"] / 25.0, t["end_frame"] / 25.0, t["code"], t["tags"]["Team"], t["tags"]["Half"])
        for t in synthetic_tags(match, cfg, 3))
    open(stem + "_smart_tagging.xml", "w").write("<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format(inst))


def test_load_match_end_to_end(tmp_path, monkeypatch):
    cfg = DEFAULT.replace(second_half_kickoff=4100)
    match = make_match(N, seed=3, cfg=cfg)
    write_bundle(tmp_path, match, cfg)

    game = games.GAMES["BAR-ATH"]
    monkeypatch.setitem(game, "n_video_frames", N)
    monkeypatch.setitem(game, "offsets", {1: 0, 2: 0})
    monkeypatch.setitem(game, "halves", {1: (500, 2999), 2: (3500, N - 1)})

    inputs = load_match(cfg, data_dir=str(tmp_path), use_cache=False)
    assert inputs.tracking.players_xy.shape == (N, 22, 2)
    assert inputs.blocks == [(1, 500, 3000), (2, 3500, N)]
    assert list(inputs.slot_team) == [0] * 11 + [1] * 11            # first team in the XML is home
    assert inputs.homography.shape == (N, 3, 3) and inputs.homography_valid.all()
    assert {t["code"] for t in inputs.tags} >= {"BUILD UP", "SET PIECES"}
    ref = corrupt(match, cfg, seed=3).ball_xy                      # frame_id - 1 alignment held
    both = ~np.isnan(ref[:, 0]) & ~np.isnan(inputs.tracking.ball_xy[:, 0])
    assert np.abs(ref[both] - inputs.tracking.ball_xy[both]).max() < 0.01

    labels = predict(inputs, cfg)
    assert validate(labels, N) is None
    assert (labels[:500] == "X").all() and (labels[3000:3500] == "X").all()
    assert (labels[3500:4100] == "D").all()
    truth = np.full(N, "X", dtype="<U1")
    for a, b in ((500, 3000), (3500, N)):
        truth[a:b] = np.array(["H", "A", "D"])[match.truth[a:b]]
    truth[3500:4100] = "D"
    result = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))
    assert result["spell_f1_mean"] > 0.4
