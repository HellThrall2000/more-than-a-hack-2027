# PLAN.md section 9: one command from the data bundle to a validated submission.
#
#   make setup           uv sync: the pinned environment from uv.lock (add EXTRA=video for the GPU parts)
#   make test            unit + integration tests on synthetic fixtures (no data needed)
#   make test-fast       the same without the (slow, torch) video tests
#   make demo            the whole pipeline on a synthetic match, scored against its known truth
#   make audit           FIRST thing to run on the real bundle -- aggregate statistics only
#   make check-baselines rebuild the four committed baselines from the data; must be byte-identical
#   make diagnose        where the pipeline loses points vs the public window (aggregate counts only)
#   make audio           does the video have sound? count the referee's whistles (aggregates only)
#   make video-check     GPU ball detector self-check on frames where the tracking sees the ball
#   make video-detect    GPU ball detection on every missing-ball frame -> cache/video_ball_fullframe.npz
#   make final           THE candidate submission: v3 rules + our ATD (video balls inside the pitch) -> $(OUT)
#   make compare-atd     given ATD vs our ATD (balls filled from the video), v3 rules and new rules
#   make run-all         EVERYTHING in one go; full logs in logs/, aggregate summary in reports/
#   make report          totals-only cross-check against Smart Tagging and whistles (safe to paste)
#   make semantics       SITUATION.md Q1-Q4: where true boundaries sit vs our detectors (keep local)
#   make full-run        the whole match end to end, with progress, logs/ and a summary (NAME=<team>)
#   make submission      predict + validate  ->  submissions/$(NAME).csv
#   make score           score the submission on the public 5 minutes (a CORRECTNESS check only)
#   make repro-check     build twice and compare -- the rules require the same file every time
#
# NAME is the team name: SUBMISSION.md wants <team_name>.csv.
PY   ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)
NAME ?= submission
OUT  ?= submissions/$(NAME).csv

.PHONY: setup final compare-atd video-check video-detect run-all report audio test test-fast demo audit diagnose semantics full-run check-baselines predict validate score submission repro-check

EXTRA ?= video
setup:
	uv sync $(if $(EXTRA),--extra $(EXTRA),)

test:
	$(PY) -m pytest -q tests

test-fast:
	$(PY) -m pytest -q tests --ignore=tests/test_video.py

demo:
	$(PY) -m possession.run --synthetic --seed 0

audit:
	$(PY) -m possession.audit

diagnose:
	$(PY) -m possession.diagnose --gt public

video-check:
	$(PY) -m possession.video.fullframe --evaluate

video-detect:
	$(PY) -m possession.video.fullframe

V3_SET = --set taker_dist=0 --set goal_mouth_half=-1 --set gk_parry_factor=1 --set dec_bonus_stop_audio=0 --set emit_shape_weight=0
OURS   = cache/atd_ours_inside

final:
	$(PY) -m possession.video.fullframe --evaluate || test $$? -eq 3
	$(PY) -m possession.video.fullframe --force
	$(PY) -m possession.video.write_atd --out-root $(OURS) --inside-m 1.0
	MTAH_DATA_DIR=$(OURS) $(PY) -m possession.run --out $(OUT) $(V3_SET)
	$(PY) evaluation/validate_submission.py $(OUT)

compare-atd:
	scripts/compare_atd.sh

run-all:
	scripts/run_all.sh $(NAME)

report:
	$(PY) -m possession.report --audio

audio:
	$(PY) -m possession.audio

semantics:
	$(PY) -m possession.semantics --gt public

full-run:
	$(PY) -m possession.full_run --name $(NAME)

check-baselines:
	mkdir -p cache/baselines
	$(PY) starter/baseline.py --out cache/baselines
	@for f in raw smooth25 smooth50 smooth125; do \
	  b=baseline_$$f.csv; \
	  cmp cache/baselines/$$b evaluation/results/$$b && echo "identical: $$b" || { echo "DIFFERENT: $$b"; exit 1; }; \
	done

predict:
	$(PY) -m possession.run --out $(OUT)

validate:
	$(PY) evaluation/validate_submission.py $(OUT)

score:
	$(PY) -m possession.harness $(OUT) --gt public

submission: predict validate

repro-check:
	$(PY) -m possession.run --out submissions/repro_a.csv
	$(PY) -m possession.run --out submissions/repro_b.csv
	sha256sum submissions/repro_a.csv submissions/repro_b.csv
	cmp submissions/repro_a.csv submissions/repro_b.csv && echo "IDENTICAL"
