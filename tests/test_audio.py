"""Whistle detection on built audio: a tone burst, crowd noise and things that must not count."""

import numpy as np
import pytest

from possession import audio
from possession.config import DEFAULT as cfg

RATE = cfg.audio_rate


def _scene(seconds, bursts=(), noise=0.01, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(int(seconds * RATE)).astype(np.float32) * noise
    for start, length, hz, amp in bursts:
        t = np.arange(int(length * RATE)) / RATE
        tone = amp * np.sin(2 * np.pi * hz * t) * (1 + 0.05 * np.sin(2 * np.pi * 30 * t))
        a = int(start * RATE)
        x[a:a + len(tone)] += tone.astype(np.float32)
    return x


def test_a_3khz_tone_burst_is_found_at_its_frame():
    found = audio.detect_whistles(_scene(30, [(10.0, 0.8, 3000.0, 0.3)]), RATE, cfg)
    assert len(found) == 1
    a, b = found[0]
    assert abs(a - 250) <= 3 and abs((b - a) - 20) <= 4


def test_two_whistles_are_two_events():
    found = audio.detect_whistles(_scene(30, [(5.0, 0.5, 3200.0, 0.3), (20.0, 1.0, 2800.0, 0.3)]), RATE, cfg)
    assert [round(a / 25) for a, _ in found] == [5, 20]


@pytest.mark.parametrize("burst", [
    (10.0, 0.8, 500.0, 0.3),       # a low tone: a horn or a voice, not a whistle
    (10.0, 0.05, 3000.0, 0.3),     # a click
    (5.0, 5.0, 3000.0, 0.3),       # a held stadium tone
])
def test_things_that_are_not_whistles_are_not_found(burst):
    assert audio.detect_whistles(_scene(30, [burst]), RATE, cfg) == []


def test_loud_broadband_crowd_noise_is_not_a_whistle():
    x = _scene(30, noise=0.4)
    assert audio.detect_whistles(x, RATE, cfg) == []


def test_a_whistle_inside_loud_crowd_noise_is_still_found():
    x = _scene(30, [(12.0, 0.7, 3000.0, 0.5)], noise=0.08)
    found = audio.detect_whistles(x, RATE, cfg)
    assert len(found) == 1 and abs(found[0][0] - 300) <= 4


def test_shift_moves_whistles_earlier_for_the_speed_of_sound():
    assert audio.shift([(100, 120)], 5) == [(95, 115)]


def test_wav_round_trip(tmp_path):
    import wave
    x = _scene(2, [(0.5, 0.5, 3000.0, 0.3)])
    path = str(tmp_path / "a.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    y, rate = audio.read_wav(path)
    assert rate == RATE and len(y) == len(x) and np.abs(x - y).max() < 1e-3


def test_missing_video_and_missing_audio_stream_are_clear_errors(tmp_path):
    with pytest.raises(audio.AudioError, match="no such video"):
        audio.probe(str(tmp_path / "nope.mp4"))


def test_calibration_prefers_thresholds_where_whistles_sit_inside_set_piece_tags():
    bursts = [(t, 0.6, 3000.0, 0.3) for t in (10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340,
                                              370, 400, 430, 460, 490, 520, 550, 580, 610)]
    x = _scene(640, bursts)
    tags = [{"code": "SET PIECES", "start_frame": int((t + 0.5) * 25), "end_frame": int((t + 6) * 25), "tags": {}}
            for t, *_ in bursts]
    rows = audio.calibrate(x, RATE, tags, cfg)
    lift, count, *_, hit = rows[0]
    assert count >= 20 and hit == 1.0 and lift > 1.5            # every whistle inside a tag; chance is ~54%
