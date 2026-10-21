# PLAN.md section 9: one command from the data bundle to a validated submission.
#
#   make test            unit + integration tests on synthetic fixtures (no data needed)
#   make test-fast       the same without the (slow, torch) video tests
#   make demo            the whole pipeline on a synthetic match, scored against its known truth
#   make audit           FIRST thing to run on the real bundle -- aggregate statistics only
#   make check-baselines rebuild the four committed baselines from the data; must be byte-identical
#   make diagnose        where the pipeline loses points vs the public window (aggregate counts only)
#   make submission      predict + validate  ->  submissions/$(NAME).csv
#   make score           score the submission on the public 5 minutes (a CORRECTNESS check only)
#   make repro-check     build twice and compare -- the rules require the same file every time
#
# NAME is the team name: SUBMISSION.md wants <team_name>.csv.
PY   ?= .venv/bin/python
NAME ?= submission
OUT  ?= submissions/$(NAME).csv

.PHONY: test test-fast demo audit diagnose check-baselines predict validate score submission repro-check

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
