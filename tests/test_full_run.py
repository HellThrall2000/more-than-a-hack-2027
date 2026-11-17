import json
import logging

import pytest

from possession import full_run
from possession import log as plog


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def lines():
    handler = Collect()
    logger = plog.get()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def test_progress_reports_rate_eta_and_a_final_line(lines, monkeypatch):
    clock = iter(range(0, 1000, 5))
    monkeypatch.setattr(plog.time, "perf_counter", lambda: float(next(clock)))
    with plog.Progress(100, "decoding", "frames", every=10.0) as p:
        for _ in range(4):
            p.update(10)
    assert any("decoding:" in s and "%" in s and "ETA" in s for s in lines)
    assert "decoding: 40 frames in" in lines[-1]


def test_stage_logs_failure_and_reraises(lines):
    with pytest.raises(ValueError):
        with plog.stage("doomed"):
            raise ValueError("boom")
    assert lines[0] == "==> doomed"
    assert any("doomed FAILED" in s and "boom" in s for s in lines)


def test_runner_skips_continues_and_stops():
    run = full_run.Runner(["a", "b", "c"])

    def skip():
        raise full_run.Skip("not needed")

    def broken():
        raise RuntimeError("nope")

    full_run.STEPS, saved = ("a", "b", "c", "d"), full_run.STEPS
    try:
        assert run.step("a", "first", lambda: {"x": 1}) == {"x": 1}
        assert run.step("b", "second", skip) is None
        assert run.step("c", "third", broken, critical=False) is None
        assert run.step("d", "fourth", lambda: None) is None        # not selected
        with pytest.raises(full_run.StepFailed):
            full_run.Runner(["a"]).step("a", "first", broken)
    finally:
        full_run.STEPS = saved
    status = {r["step"]: r["status"] for r in run.rows}
    assert status["a"] == "ok" and status["b"].startswith("skipped") and status["c"].startswith("FAILED")
    assert status["d"] == "not selected"


def test_missing_bundle_stops_at_check_and_still_writes_the_summary(tmp_path):
    rc = full_run.main(["--data-dir", str(tmp_path / "nothing"), "--summary-dir", str(tmp_path / "out"),
                        "--only", "check,audit"])
    assert rc == 1
    (summary,) = (tmp_path / "out").glob("*_full-run.json")
    s = json.loads(summary.read_text())
    assert s["failed_step"] == "check" and s["video"] == "tactical"
    assert [r["step"] for r in s["steps"]] == ["check"]
    assert s["steps"][0]["status"].startswith("FAILED: FileNotFoundError")


def test_unknown_step_names_are_rejected():
    with pytest.raises(SystemExit):
        full_run.main(["--only", "chek"])


def test_video_steps_are_opt_in(tmp_path, lines):
    full_run.main(["--data-dir", str(tmp_path / "nothing"), "--summary-dir", str(tmp_path / "out")])
    header = next(s for s in lines if s.startswith("full run "))
    assert "predict_tracking" in header and "final" in header
    assert not any(step in header.split("steps ")[1].split("  (")[0] for step in full_run.VIDEO_STEPS)
