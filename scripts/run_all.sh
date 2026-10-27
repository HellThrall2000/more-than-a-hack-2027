#!/usr/bin/env bash
# One command for the whole evaluation round.
#
#   scripts/run_all.sh [label]          # or: make run-all
#
# Full output of every step goes to logs/run_all_<time>/ (gitignored, never shared).
# reports/summary_<time>.txt gets ONLY aggregate lines -- scores, counts, OK/FAIL -- so it can be
# pasted to a hosted assistant (the data terms forbid frame-level extracts, not aggregate scores).
# reports/ is gitignored too.
#
# Steps: tests -> audit -> audio check -> GPU ball detection (self-check, then every missing-ball
# frame; SKIP_VIDEO=1 reuses the last detections) -> four predictions in parallel (old rules, new
# rules, new rules again for the reproducibility check, new rules + video ball) -> validate + score
# each on the public window -> totals-only report.
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
LABEL="${1:-round}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="logs/run_all_${STAMP}"
SUM="reports/summary_${STAMP}.txt"
mkdir -p "$RUN" reports submissions
: > "$SUM"

say()  { printf '%s\n' "$*" | tee -a "$SUM"; }
# keep only lines that carry aggregates; everything else stays in the full log
keep() { grep -E "$1" "$2" 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | sed -E 's/^[0-9:]+ +//' >> "$SUM"; }
step() {                                   # step <name> <command...>   -> $RUN/<name>.log, status in the summary
  local name="$1"; shift
  local t0=$SECONDS code=0
  "$@" > "$RUN/$name.log" 2>&1 || code=$?
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
else
  step video_check bash -c "\"$PY\" -m possession.video.fullframe --evaluate || test \$? -eq 3"
  keep 'steady tracks|GATE' "$RUN/video_check.log"
  step video_detect "$PY" -m possession.video.fullframe --force
  keep 'detections saved' "$RUN/video_detect.log"
fi
step write_atd "$PY" -m possession.video.write_atd --out-root "$OURS" --inside-m 1.0
keep 'filled from the video|COMPLIANT|FAILED' "$RUN/write_atd.log"

say ""; say "== predictions (4 in parallel)"
OLD=(--set taker_dist=0 --set goal_mouth_half=-1 --set gk_parry_factor=1 --set dec_bonus_stop_audio=0 --set emit_shape_weight=0)
job() { local name="$1"; shift; MTAH_LOG_DIR="$RUN/$name" "$PY" -m possession.run --out "submissions/${name}.csv" "$@" > "$RUN/$name.log" 2>&1; echo $? > "$RUN/$name.exit"; }
job old_rules "${OLD[@]}" &
job new_rules &
job new_rules_b &
job_ours() { local name="$1"; shift; MTAH_DATA_DIR="$OURS" MTAH_LOG_DIR="$RUN/$name" "$PY" -m possession.run --out "submissions/${name}.csv" "$@" > "$RUN/$name.log" 2>&1; echo $? > "$RUN/$name.exit"; }
[ -f "$OURS/tactical/BAR-ATH_tactical_atd.txt" ] && job_ours v3_our_atd "${OLD[@]}" &
wait
for name in old_rules new_rules new_rules_b v3_our_atd; do
  [ -f "$RUN/$name.exit" ] || continue
  code="$(cat "$RUN/$name.exit" 2>/dev/null || echo '?')"
  if [ "$code" = 0 ]; then say "[OK]   predict $name"
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

say ""; say "== reproducibility (same code, run twice, byte for byte)"
if [ -f submissions/new_rules.csv ] && [ -f submissions/new_rules_b.csv ]; then
  if cmp -s submissions/new_rules.csv submissions/new_rules_b.csv; then say "[OK]   identical: $(sha256sum submissions/new_rules.csv | cut -c1-16)"
  else say "[FAIL] new_rules and new_rules_b DIFFER"; fi
fi

say ""; say "== totals-only report (Smart Tagging and whistle cross-check)"
step report "$PY" -m possession.report
keep '(possessions|dead_|set_piece|goal_|whistles)[a-z_0-9]* +[0-9.]+' "$RUN/report.log"

say ""; say "done. full logs: ${RUN}/   summary: ${SUM}"
