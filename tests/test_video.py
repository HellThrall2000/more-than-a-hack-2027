"""The video ball detector, end to end, on a synthetic clip with a known ball.

A moving white ball, dark player blobs and static white pitch lines are rendered into a real video
file; labels come from the 'ATD' ball projected through a known homography, exactly as on the real
match. This tests the plumbing and that the approach can learn a few-pixel moving ball. It cannot
say how it does on the real broadcast video.
"""

import dataclasses
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from possession.config import DEFAULT as cfg
from possession.video import VideoConfig, dataset
from possession.video.evaluate import evaluate
from possession.video.frames import crop_stack, gray_frames
from possession.video.geometry import pitch_to_px, px_per_metre, px_to_pitch
from possession.video.infer import crop_sizes, detect
from possession.video.model import BallNet, decode, make_targets, out_size
from possession.video.train import fit, load_model, save_model

W, H, N = 640, 360, 640


def render_clip(path, seed=0):
    rng = np.random.RandomState(seed)
    length, width = cfg.pitch_length, cfg.pitch_width
    ball = np.zeros((N, 2))
    pos, vel = np.array([length / 2, width / 2]), rng.randn(2) * 6
    for t in range(N):
        vel += rng.randn(2) * 1.2
        vel = np.clip(vel, -14, 14)
        pos = pos + vel / 25.0
        for a, hi in ((0, length), (1, width)):
            if pos[a] < 3 or pos[a] > hi - 3:
                vel[a] *= -1
                pos[a] = np.clip(pos[a], 3, hi - 3)
        ball[t] = pos
    players = np.zeros((N, 22, 2))
    p = rng.rand(22, 2) * [length, width]
    for t in range(N):
        p = np.clip(p + rng.randn(22, 2) * 0.15, 1, [length - 1, width - 1])
        players[t] = p

    hmat = np.diag([float(W), float(H), 1.0])
    hs = np.repeat(hmat[None], N, axis=0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 25.0, (W, H))
    assert out.isOpened()
    sx, sy = W / length, H / width
    for t in range(N):
        img = np.full((H, W, 3), (60, 120, 60), dtype=np.uint8)
        cv2.rectangle(img, (int(5 * sx), int(10 * sy)), (int(100 * sx), int(58 * sy)), (240, 240, 240), 1)
        cv2.line(img, (W // 2, 0), (W // 2, H), (240, 240, 240), 1)
        cv2.circle(img, (W // 2, H // 2), int(9.15 * sx), (240, 240, 240), 1)
        for i, (x, y) in enumerate(players[t]):
            cv2.circle(img, (int(x * sx), int(y * sy)), 5, (200, 40, 40) if i < 11 else (40, 40, 200), -1)
        cv2.circle(img, (int(round(ball[t, 0] * sx)), int(round(ball[t, 1] * sy))), 2, (255, 255, 255), -1, cv2.LINE_AA)
        img = np.clip(img.astype(np.int16) + rng.randn(H, W, 3) * 3, 0, 255).astype(np.uint8)
        out.write(img)
    out.release()
    return ball, players, hs


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "clip.mp4")
    ball, players, hs = render_clip(path)
    return {"path": path, "ball": ball.astype(np.float32), "players": players.astype(np.float32), "hs": hs}


def test_geometry_roundtrip_and_scale():
    hs = np.repeat(np.diag([W, H, 1.0])[None], 3, axis=0)
    xy = np.array([[10.0, 20.0], [50.0, 30.0], [100.0, 60.0]])
    np.testing.assert_allclose(px_to_pitch(hs, pitch_to_px(hs, xy, cfg), cfg), xy, atol=1e-9)
    assert px_per_metre(hs, xy, cfg)[0] == pytest.approx(0.5 * (W / cfg.pitch_length + H / cfg.pitch_width))


def test_crop_stack_has_the_ball_where_the_label_says_and_pads_outside_the_image():
    frame = np.zeros((50, 80), dtype=np.uint8)
    frame[20, 70] = 255
    crop, x0, y0 = crop_stack([frame, frame, frame], 70, 20, 32)
    assert crop.shape == (3, 32, 32) and crop[0, 20 - y0, 70 - x0] == 255      # label = (px - x0, py - y0)
    near_edge, x0, y0 = crop_stack([frame], 78, 30, 32)                        # window hangs off the image
    assert near_edge.shape == (1, 32, 32) and near_edge[0, 20 - y0, 70 - x0] == 255


def test_decoded_frames_come_in_order_and_skip_the_rest(scene):
    got = [i for i, _ in gray_frames(scene["path"], [0, 5, 6, 100, N - 1])]
    assert got == [0, 5, 6, 100, N - 1]


def test_targets_peak_exactly_at_the_ball_cell_and_decode_inverts_them():
    uv = torch.tensor([[40.0, 52.0], [np.nan, np.nan]])
    present = torch.tensor([True, False])
    heat, cell, off = make_targets(uv, present, 96, 0.8)
    assert heat.shape == (2, 1, out_size(96), out_size(96)) and heat[0].max() == 1.0 and heat[1].max() == 0.0
    assert tuple(cell[0].tolist()) == (10, 13)
    out = torch.zeros(1, 3, 24, 24)
    out[0, 0] = -9.0
    out[0, 0, 13, 10] = 9.0
    out[0, 1, 13, 10], out[0, 2, 13, 10] = float(off[0, 0]), float(off[0, 1])
    conf, u, v = decode(out)
    assert conf[0] > 0.99 and abs(u[0] - 40.0) < 1e-4 and abs(v[0] - 52.0) < 1e-4


@pytest.fixture(scope="module")
def trained(scene):
    vcfg = VideoConfig(stride=1, pos_per_frame=2, neg_per_frame=1, val_segment_frames=80, val_every=4,
                       epochs=14, batch=64, lr=3e-3, pixel_tol=3.0)
    blocks = [(1, 0, N)]
    cs = dataset.build(scene["path"], scene["ball"], scene["players"], scene["hs"], blocks, cfg, vcfg)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model, report = fit(cs, vcfg, torch.device("cpu"), log=lambda *_: None)
    return {"vcfg": vcfg, "cs": cs, "model": model, "report": report, "blocks": blocks}


def test_dataset_labels_land_on_the_rendered_ball(trained):
    cs = trained["cs"]
    pos = cs.present
    assert pos.sum() > 800 and (~pos).sum() > 200 and cs.X.shape[1:] == (3, 96, 96)
    brightest = [np.unravel_index(np.argmax(c[1]), c[1].shape) for c in cs.X[pos][:60]]     # centre frame
    uv = cs.uv[pos][:60]
    hit = [abs(y - v) <= 3 and abs(x - u) <= 3 for (y, x), (u, v) in zip(brightest, uv)]
    assert np.mean(hit) > 0.8       # the label is on the bright blob (lines/players can win a few)


def test_detector_clears_the_gate_on_held_out_crops(trained):
    r = trained["report"]
    assert r["meets_gate"], r["at_thr"]
    assert r["at_thr"]["recall"] > 0.6 and r["median_px_err"] < 2.0


def test_roi_search_recovers_the_ball_near_a_noisy_prior(scene, trained):
    seg = np.arange(N) // trained["vcfg"].val_segment_frames
    frames = np.flatnonzero((seg % 4) == 3)[8:-8:2]
    rng = np.random.RandomState(1)
    prior = scene["ball"][frames] + rng.randn(len(frames), 2) * 1.5          # the filler is ~1.5 m off
    xy, conf = detect(scene["path"], frames, prior, np.full(len(frames), 3.0), scene["hs"], trained["model"],
                      cfg, trained["vcfg"], "cpu")
    thr = trained["report"]["thr"]
    found = conf >= thr
    err = np.hypot(*(xy - scene["ball"][frames]).T)
    assert found.mean() > 0.6
    assert np.median(err[found]) < 1.0 and np.median(err[found]) < np.median(np.hypot(*(prior - scene["ball"][frames]).T))


def test_frames_with_a_vague_prior_are_not_searched(scene, trained):
    frames = np.array([100, 101])
    xy, conf = detect(scene["path"], frames, scene["ball"][frames], np.array([30.0, 1.0]), scene["hs"],
                      trained["model"], cfg, trained["vcfg"], "cpu")
    assert conf[0] == 0.0 and np.isnan(xy[0]).all() and conf[1] > 0.0
    assert crop_sizes(scene["ball"][:2], np.array([1.0, 6.0]), scene["hs"][:2], cfg, trained["vcfg"])[1] > \
        crop_sizes(scene["ball"][:2], np.array([1.0, 6.0]), scene["hs"][:2], cfg, trained["vcfg"])[0]


def test_model_checkpoint_roundtrip_and_fully_convolutional(trained, tmp_path):
    path = str(tmp_path / "m.pt")
    save_model(path, trained["model"], trained["vcfg"], trained["report"])
    model, vcfg, report = load_model(path)
    x = torch.rand(2, 3, 160, 224) * 255
    assert tuple(model(x).shape) == (2, 3, out_size(160), out_size(224))      # any crop size works
    assert vcfg == trained["vcfg"] and report["thr"] == trained["report"]["thr"]
    assert np.allclose(model(x).detach().numpy(), trained["model"].eval()(x).detach().numpy(), atol=1e-5)


def test_video_detections_enter_the_ball_layer_only_where_the_atd_has_none(scene):
    from possession.ball import build_ball_state, merge_video
    from possession.ball.gaps import SRC_OBSERVED
    n = 400
    ball = scene["ball"][:n].copy()
    ball[100:160] = np.nan
    xy = np.full((n, 2), np.nan, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    xy[110:140], conf[110:140] = scene["ball"][110:140], 0.9
    xy[10:20], conf[10:20] = [0.0, 0.0], 0.99                  # ATD already sees the ball here: ignored
    merged, used = merge_video(ball, (xy, conf, 0.5), cfg)
    assert used[110:140].all() and not used[10:20].any() and not used[:100].any()
    np.testing.assert_allclose(merged[110:140], scene["ball"][110:140])
    slot_team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    state = build_ball_state(ball, scene["players"][:n], slot_team, [(1, 0, n)], cfg, (xy, conf, 0.5))
    assert state.video_used[110:140].all() and (state.src[110:140] == SRC_OBSERVED).all()
    without = build_ball_state(ball, scene["players"][:n], slot_team, [(1, 0, n)], cfg)
    err = lambda s: np.nanmedian(np.hypot(*(s.xy[100:160] - scene["ball"][100:160]).T))      # noqa: E731
    assert err(state) <= err(without)


def test_go_no_go_gate_hides_visible_balls_and_measures_recovery(scene, trained):
    """evaluate(): hide held-out visible stretches, fill them, search the video, compare with truth."""
    ball = scene["ball"].copy()
    for a, b in ((20, 40), (130, 160), (200, 235), (360, 395)):    # real-looking gaps, outside the held-out blocks
        ball[a:b] = np.nan
    slot_team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    result = evaluate(trained["model"], trained["vcfg"], trained["report"]["thr"], scene["path"], ball,
                      scene["players"], slot_team, scene["hs"], trained["blocks"], cfg, n_windows=12, seed=2)
    assert result["n_windows"] >= 5 and result["n_frames"] > 100
    o = result["overall"]
    assert o["coverage"] > 0.3 and o["precision"] >= 0.8
    assert o["filler_plus_video_median_m"] < o["filler_median_m"]       # the video actually helps the fill
    assert result["gate"]["met"] and result["gate"]["worthwhile"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no ROCm / CUDA device")
def test_training_on_the_gpu_clears_the_gate_and_cpu_inference_agrees(trained):
    """The intended split: train on the GPU (an AMD card under ROCm reports as cuda), infer on the
    CPU, where the submission has to reproduce exactly."""
    from possession.video.model import pick_device
    from possession.video.train import predict_crops
    gpu = pick_device("cuda")
    model, report = fit(trained["cs"], trained["vcfg"], gpu, log=lambda *_: None)
    assert report["meets_gate"], report["at_thr"]
    X = trained["cs"].X[:256]
    gpu_conf, _, _ = predict_crops(model, X, gpu)
    cpu_conf, _, _ = predict_crops(model.to("cpu"), X, torch.device("cpu"))
    assert np.abs(gpu_conf - cpu_conf).max() < 1e-3


def test_bench_reads_the_tactical_video_and_writes_aggregate_timings_only(scene, tmp_path):
    import json
    import shutil
    import warnings
    from games import asset_paths
    from possession.video import bench
    data = tmp_path / "data"
    target = asset_paths(data_dir=str(data), source="tactical")["video"]
    os.makedirs(os.path.dirname(target))
    shutil.copy(scene["path"], target)
    out = tmp_path / "results"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # the clip is not 199,319 frames long
        assert bench.main(["--data-dir", str(data), "--devices", "cpu", "--start", "10", "--frames", "20",
                           "--repeat", "2", "--out-dir", str(out)]) == 0
    files = list(out.glob("*.json"))
    assert len(files) == 1
    result = json.loads(files[0].read_text())
    assert result["video"] == "tactical" and result["decode"]["fps"] > 0
    row = result["runs"][0]
    assert row["device"] == "cpu" and row["error"] is None and row["train_step_ms"] > 0
    assert set(row["infer_batch_ms"]) == {"96", "160", "320"}
    assert set(result) == {"benchmark", "when_utc", "video", "environment", "decode", "runs"}
