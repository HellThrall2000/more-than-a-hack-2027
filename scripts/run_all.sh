#!/usr/bin/env bash
# One command for the whole evaluation round.
#
#   scripts/run_all.sh [label]          # or: make run-all
#
# Full output of every step goes to logs/run_all_<time>/ (gitignored, never shared).
# While a step runs the console shows a live progress bar (scripts/lib_progress.sh); Ctrl-C stops it.
# reports/summary_<time>.txt gets ONLY aggregate lines -- scores, counts, OK/FAIL -- so it can be
# pasted to a hosted assistant (the data terms forbid frame-level extracts, not aggregate scores).
# reports/ is gitignored too.
#
# Steps: tests -> audit -> audio check -> GPU ball detection (self-check, then every missing-ball
# frame; SKIP_VIDEO=1 reuses the last detections) -> four predictions in parallel (old rules, new
# rules, new rules again for the reproducibility check, old rules + video ball) -> validate + score
# each on the public window -> submissions/<label>.csv (the video version when it exists, else the
# tracking-only one; a copy, validated) -> reproducibility check -> totals-only report (with the
# whistle cross-check) -> a timing/score record for benchmarks/results/ (aggregates only).
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY=".venv/Scripts/python.exe"     # Windows (Git Bash)
[ -x "$PY" ] || PY="python3"
LABEL="${1:-round}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="logs/run_all_${STAMP}"
SUM="reports/summary_${STAMP}.txt"
mkdir -p "$RUN" reports submissions
: > "$SUM"

# shellcheck source=scripts/lib_progress.sh
source "$(dirname "$0")/lib_progress.sh"

say()  { printf '%s\n' "$*" | tee -a "$SUM"; }
# keep only lines that carry aggregates; everything else stays in the full log
keep() { grep -E "$1" "$2" 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | sed -E 's/^[0-9:]+ +//' >> "$SUM"; }
# row <name> <ok|failed|reused> <seconds> <exit code>: one line per step for the benchmark record
row()  { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RUN/steps.tsv"; }
# step <name> <command...>: run it in the background with its output in $RUN/<name>.log, show live
# progress on the console while it runs (scripts/lib_progress.sh), then the status in the summary.
# OKCODES="0 3" step ...  lists the exit codes that count as success (default: 0).
# Ctrl-C stops the running step and everything it started, then the script.
step() {
  local name="$1"; shift
  local t0=$SECONDS code=0 pid ok raw
  "$@" > "$RUN/$name.log" 2>&1 &
  pid=$!
  trap 'kill_tree "$pid"; printf "\r\033[K"; say "[STOPPED] '"$name"' (Ctrl-C after $((SECONDS - t0))s) -- log: '"$RUN"'/'"$name"'.log"; exit 130' INT TERM
  watch_step "$pid" "$name" "$RUN/$name.log" "$t0"
  wait "$pid" || code=$?
  trap - INT TERM
  raw=$code
  for ok in ${OKCODES:-0}; do [ "$code" = "$ok" ] && code=0; done
  row "$name" "$([ "$code" = 0 ] && echo ok || echo failed)" "$((SECONDS - t0))" "$raw"
  if [ "$code" = 0 ]; then
    say "[OK]   $name  ($((SECONDS - t0))s)"
  else
    say "[FAIL] $name  (exit $code, $((SECONDS - t0))s)  last lines of its log:"
    grep -vE '^\s*$|Warning|nanmean|nanvar|^\s+[a-z_]+\[' "$RUN/$name.log" | tail -3 | sed 's/^/         /' | tee -a "$SUM"
  fi
}

say "run_all ${LABEL}  ${STAMP}  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)  commit=$(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo '+uncommitted')"
say "cores=$(nproc)  python=$($PY --version 2>&1)"

for f in atd.txt atd.xml homography.csv smart_tagging.xml; do
  [ -f "data/tactical/BAR-ATH_tactical_${f}" ] || say "[MISSING] data/tactical/BAR-ATH_tactical_${f}"
done
[ -f data/tactical/BAR-ATH_tactical.mp4 ] || say "[MISSING] data/tactical/BAR-ATH_tactical.mp4  (audio and video steps will fail)"

say ""; say "== tests"
step tests "$PY" -m pytest -q --ignore=tests/test_video.py
keep '[0-9]+ (passed|failed)' "$RUN/tests.log"

say ""; say "== data audit"
step audit "$PY" -m possession.audit
keep 'ALIGNMENT (OK|FAILED)|match: ' "$RUN/audit.log"

say ""; say "== audio"
step audio "$PY" -m possession.audio
keep 'audio stream|whistles found|NO audio' "$RUN/audio.log"

say ""; say "== video ball detection -> our ATD (GPU, pretrained roboflow/sports football-ball-detection, no training)"
VIDEO=cache/video_ball_fullframe.npz
OURS=cache/atd_ours_inside
if [ "${SKIP_VIDEO:-0}" = 1 ] && [ -f "$VIDEO" ]; then
  say "[SKIP] video detection: reusing $VIDEO"
  row video_detect reused 0 0
else
  OKCODES="0 3" step video_check "$PY" -m possession.video.fullframe --evaluate     # 3 = gate not passed: expected
  keep 'steady tracks|GATE' "$RUN/video_check.log"
  step video_detect "$PY" -m possession.video.fullframe --force
  keep 'detections saved' "$RUN/video_detect.log"
fi
step write_atd "$PY" -m possession.video.write_atd --out-root "$OURS" --inside-m 1.0
keep 'filled from the video|COMPLIANT|FAILED' "$RUN/write_atd.log"

say ""; say "== predictions (4 in parallel)"
OLD=(--set taker_dist=0 --set goal_mouth_half=-1 --set gk_parry_factor=1 --set dec_bonus_stop_audio=0 --set emit_shape_weight=0)
# job <name> <run args...>: one prediction; exit code in <name>.exit and seconds in <name>.secs when done
job() { local name="$1" t=$SECONDS; shift; MTAH_LOG_DIR="$RUN/$name" "$PY" -m possession.run --out "submissions/${name}.csv" "$@" > "$RUN/$name.log" 2>&1; echo $? > "$RUN/$name.exit"; echo $((SECONDS - t)) > "$RUN/$name.secs"; }
job_ours() { local name="$1" t=$SECONDS; shift; MTAH_DATA_DIR="$OURS" MTAH_LOG_DIR="$RUN/$name" "$PY" -m possession.run --out "submissions/${name}.csv" "$@" > "$RUN/$name.log" 2>&1; echo $? > "$RUN/$name.exit"; echo $((SECONDS - t)) > "$RUN/$name.secs"; }
JOBS=(); JOB_PIDS=(); JOBS_T0=$SECONDS
job old_rules "${OLD[@]}" & JOBS+=(old_rules); JOB_PIDS+=($!)
job new_rules & JOBS+=(new_rules); JOB_PIDS+=($!)
job new_rules_b & JOBS+=(new_rules_b); JOB_PIDS+=($!)
if [ -f "$OURS/tactical/BAR-ATH_tactical_atd.txt" ]; then
  job_ours v3_our_atd "${OLD[@]}" & JOBS+=(v3_our_atd); JOB_PIDS+=($!)
else
  say "[SKIP] predict v3_our_atd: $OURS has no ATD (write_atd failed)"
fi
trap 'for p in "${JOB_PIDS[@]}"; do kill_tree "$p"; done; printf "\r\033[K"; say "[STOPPED] predictions (Ctrl-C after $((SECONDS - JOBS_T0))s) -- logs: '"$RUN"'/"; exit 130' INT TERM
watch_jobs "$JOBS_T0" "$RUN" "${JOBS[@]}"
wait
trap - INT TERM
for name in "${JOBS[@]}"; do
  code="$(cat "$RUN/$name.exit" 2>/dev/null || echo '?')"
  secs="$(cat "$RUN/$name.secs" 2>/dev/null || echo 0)"
  row "predict_$name" "$([ "$code" = 0 ] && echo ok || echo failed)" "$secs" "$code"
  if [ "$code" = 0 ]; then say "[OK]   predict $name  (${secs}s)"
  else say "[FAIL] predict $name (exit $code)"; grep -E 'Error|Traceback' "$RUN/$name.log" | tail -2 | sed 's/^/         /' | tee -a "$SUM"; fi
  keep 'result: [0-9]+ possessions|player-shape team model|video detections used' "$RUN/$name.log"
done

say ""; say "== validate and score on the public window (a correctness check, never a way to pick)"
for name in old_rules new_rules v3_our_atd; do
  [ -f "submissions/${name}.csv" ] || continue
  [ -f "$RUN/$name.exit" ] || continue
  step "validate_$name" "$PY" evaluation/validate_submission.py "submissions/${name}.csv"
  step "score_$name" "$PY" -m possession.harness "submissions/${name}.csv" --gt public --bootstrap 1000
  keep 'spellF1|bootstrap' "$RUN/score_$name.log"
done

# The file to hand in. The documented candidate is the version-3 rules on our ATD (TECHNICAL.md,
# `make final`); without it (no video detections) the same rules on the provided ATD.
say ""; say "== submission file: submissions/${LABEL}.csv"
FINAL_FROM=old_rules
[ "$(cat "$RUN/v3_our_atd.exit" 2>/dev/null)" = 0 ] && [ -f submissions/v3_our_atd.csv ] && FINAL_FROM=v3_our_atd
if [ -f "submissions/${FINAL_FROM}.csv" ] && [ "$(cat "$RUN/$FINAL_FROM.exit" 2>/dev/null)" = 0 ]; then
  cp "submissions/${FINAL_FROM}.csv" "submissions/${LABEL}.csv"
  say "   ${LABEL}.csv is a copy of ${FINAL_FROM}.csv ($([ "$FINAL_FROM" = v3_our_atd ] && echo 'rules v3 on our ATD, balls found in the video' || echo 'rules v3 on the provided ATD, tracking only'))"
  step validate_final "$PY" evaluation/validate_submission.py "submissions/${LABEL}.csv"
  say "   sha256 $(sha256sum "submissions/${LABEL}.csv" | cut -c1-16)"
else
  say "[FAIL] no valid prediction to use as submissions/${LABEL}.csv"
fi

say ""; say "== reproducibility (same code, run twice, byte for byte)"
if [ -f submissions/new_rules.csv ] && [ -f submissions/new_rules_b.csv ]; then
  if cmp -s submissions/new_rules.csv submissions/new_rules_b.csv; then say "[OK]   identical: $(sha256sum submissions/new_rules.csv | cut -c1-16)"
  else say "[FAIL] new_rules and new_rules_b DIFFER"; fi
fi

say ""; say "== totals-only report (Smart Tagging and whistle cross-check)"
step report "$PY" -m possession.report --audio
keep '(possessions|dead_|set_piece|goal_|whistles)[a-z_0-9]* +[0-9.]+' "$RUN/report.log"

say ""; say "== benchmark record (timings, versions, aggregate scores -- no match data)"
step record "$PY" -m possession.run_record --label "$LABEL" --steps "$RUN/steps.tsv" --summary "$SUM"
keep 'benchmark record' "$RUN/record.log"

say ""; say "done. full logs: ${RUN}/   summary: ${SUM}"
