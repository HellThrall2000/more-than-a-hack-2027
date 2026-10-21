import numpy as np

from possession.ingest import parse_atd


def test_parse_atd_roundtrip(tmp_path):
    raw = tmp_path / "atd.txt"
    raw.write_bytes(
        b"1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,3.0:0.25,0.5,5.0,0.3\r\n"
        b"2:0.5,0.5,1.5;NaN;0.1,0.2,3.5:NaN,NaN,NaN\r\n")
    tr = parse_atd(str(raw), n_frames=3, pitch=(100.0, 50.0), max_ball_fields=4)

    assert tr.players_xy.shape == (3, 3, 2)
    np.testing.assert_allclose(tr.players_xy[0, 0], [50.0, 25.0])      # normalised -> metres
    np.testing.assert_allclose(tr.players_xy[0, 2], [10.0, 10.0])
    assert np.isnan(tr.players_xy[0, 1]).all() and np.isnan(tr.players_xy[1, 1]).all()
    np.testing.assert_allclose(tr.players_speed[1, 2], 3.5)
    np.testing.assert_allclose(tr.ball_xy[0], [25.0, 25.0])
    np.testing.assert_allclose(tr.ball_extra[0], [5.0, 0.3])
    assert np.isnan(tr.ball_xy[1]).all()
    assert np.isnan(tr.players_xy[2]).all() and np.isnan(tr.ball_xy[2]).all()    # absent frame stays NaN
