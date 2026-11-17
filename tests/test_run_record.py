"""possession.run_record: the aggregate record of a run_all round."""

import json

from possession import run_record

SUMMARY = """\
[OK]   predict old_rules  (60s)
[info     ] result: 263 possessions (mean 12.8 s), 176 dead-ball stretches | graded frames H 33.2% A 19.9% D 46.9%
[info     ] old_rules.csv: spellF1 0.4970  macroF1 0.7770  possessions 16 (gt 17)  config 453315b630
[info     ]   spellF1 bootstrap 5-95%: 0.3273 .. 0.6667
[info     ] self-check, steady tracks only                               coverage=0.4488 precision=0.7175 thr=0.5 within_5m=0.8459
[error    ] GATE NOT PASSED: no threshold reaches precision 0.9 -- do not use the detections
[info     ] detections saved                                             file=cache/x.npz recovered=12414 searched=64687 share=0.1919
[OK]   identical: 78b329d604d86d2b
"""


def test_the_summary_is_turned_into_scores_and_totals(tmp_path):
    path = tmp_path / "summary.txt"
    path.write_text(SUMMARY, encoding="utf-8")
    got = run_record.read_summary(str(path))
    assert got["scores"]["old_rules"]["spell_f1"] == 0.497
    assert got["scores"]["old_rules"]["spell_f1_bootstrap_5_95"] == [0.3273, 0.6667]
    assert got["predictions"]["old_rules"]["dead_ball_stretches"] == 176
    assert got["predictions"]["old_rules"]["graded_share_pct"] == {"H": 33.2, "A": 19.9, "D": 46.9}
    assert got["video_gate"] == [{"coverage": 0.4488, "precision": 0.7175, "thr": 0.5}]
    assert got["detections"] == {"recovered": 12414, "searched": 64687, "share": 0.1919}
    assert got["reproducible"] is True


def test_the_record_lists_failed_steps_and_adds_up_the_time(tmp_path):
    steps = tmp_path / "steps.tsv"
    steps.write_text("tests\tok\t16\t0\nwrite_atd\tfailed\t2\t1\nvideo_detect\treused\t0\t0\n", encoding="utf-8")
    summary = tmp_path / "summary.txt"
    summary.write_text(SUMMARY, encoding="utf-8")
    assert run_record.main(["--label", "t", "--steps", str(steps), "--summary", str(summary),
                            "--out-dir", str(tmp_path)]) == 0
    record = json.loads(next(tmp_path.glob("*_run-all.json")).read_text(encoding="utf-8"))
    assert record["failed_steps"] == ["write_atd"]
    assert record["total_seconds"] == 18
    assert [s["status"] for s in record["steps"]] == ["ok", "failed", "reused"]
