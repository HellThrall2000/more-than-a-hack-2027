#!/usr/bin/env bash
# Given ATD vs our ATD (missing balls filled from the video), each with the v3 rules and the new rules.
#
#   scripts/compare_atd.sh            # needs cache/video_ball_fullframe.npz (make video-detect)
#
# Full logs: logs/compare_atd_<time>/ ; aggregate summary: reports/compare_atd_<time>.txt (both gitignored).
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="logs/compare_atd_${STAMP}"
SUM="reports/compare_atd_${STAMP}.txt"
OURS="${OURS:-cache/atd_ours}"
WRITE_ARGS=(${WRITE_ARGS:-})
mkdir -p "$RUN" reports submissions
: > "$SUM"
say()  { printf '%s\n' "$*" | tee -a "$SUM"; }
keep() { grep -E "$1" "$2" 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | sed -E 's/^[0-9:]+ +//' | tee -a "$SUM" >/dev/null; }

say "compare_atd ${STAMP}  ours=${OURS} ${WRITE_ARGS[*]}"
say ""; say "== build our ATD"
if "$PY" -m possession.video.write_atd --out-root "$OURS" "${WRITE_ARGS[@]}" > "$RUN/write_atd.log" 2>&1; then say "[OK]   write_atd"; else say "[FAIL] write_atd"; fi
keep 'filled from the video|verification|COMPLIANT|FAILED' "$RUN/write_atd.log"
if MTAH_DATA_DIR="$OURS" "$PY" -m possession.audit > "$RUN/audit_ours.log" 2>&1; then say "[OK]   audit on our ATD"; else say "[FAIL] audit on our ATD"; fi
keep 'ALIGNMENT|match: ' "$RUN/audit_ours.log"

V3=(--set taker_dist=0 --set goal_mouth_half=-1 --set gk_parry_factor=1 --set dec_bonus_stop_audio=0 --set emit_shape_weight=0)
run() {  # run <name> <data dir or ''> [extra args]
  local name="$1" dir="$2"; shift 2
  if [ -n "$dir" ]; then MTAH_DATA_DIR="$dir" MTAH_LOG_DIR="$RUN/$name" "$PY" -m possession.run --out "submissions/$name.csv" "$@" > "$RUN/$name.log" 2>&1
  else MTAH_LOG_DIR="$RUN/$name" "$PY" -m possession.run --out "submissions/$name.csv" "$@" > "$RUN/$name.log" 2>&1; fi
  echo $? > "$RUN/$name.exit"
}
say ""; say "== predictions (4 in parallel)"
run v3_given_atd "" "${V3[@]}" &
run v3_our_atd "$OURS" "${V3[@]}" &
run new_given_atd "" &
run new_our_atd "$OURS" &
wait
for name in v3_given_atd v3_our_atd new_given_atd new_our_atd; do
  [ "$(cat "$RUN/$name.exit")" = 0 ] && say "[OK]   $name" || { say "[FAIL] $name"; grep -E 'Error|Traceback' "$RUN/$name.log" | tail -2 | tee -a "$SUM"; }
  keep 'graded frames by source|result: [0-9]+ possessions|player-shape' "$RUN/$name.log"
  "$PY" evaluation/validate_submission.py "submissions/$name.csv" > "$RUN/validate_$name.log" 2>&1 && say "       valid" || say "       INVALID"
  "$PY" -m possession.harness "submissions/$name.csv" --gt public --bootstrap 1000 > "$RUN/score_$name.log" 2>&1
  keep 'spellF1' "$RUN/score_$name.log"
done
say ""; say "== totals-only cross-check, our ATD"
MTAH_DATA_DIR="$OURS" "$PY" -m possession.report > "$RUN/report_ours.log" 2>&1
keep '(possessions|dead_|set_piece|goal_)[a-z_0-9]* +[0-9.]+' "$RUN/report_ours.log"
say ""; say "== diagnosis on the public window, our ATD (new rules)"
MTAH_DATA_DIR="$OURS" "$PY" -m possession.diagnose --gt public > "$RUN/diagnose_ours.log" 2>&1
grep -E "spellF1|by IoU|alive/dead|transitions at|frame share by source|control team ==" "$RUN/diagnose_ours.log" | sed -E 's/\x1b\[[0-9;]*m//g' | tee -a "$SUM" >/dev/null
say ""; say "done. full logs: ${RUN}/   summary: ${SUM}"
