#!/usr/bin/env bash
# Live progress for long steps. Sourced by run_all.sh; nothing here runs by itself.
#
# A step runs in the background with all of its output in a log file, and watch_step draws what it
# is doing on the console every couple of seconds, so a long step never looks stuck:
#
#   run_all  video_detect [##############--------------] decoding for ball detection: 58% 110400/190198
#            frames 79 frames/s ETA 16:49 | 24:12
#
# It reads the "NN% ... ETA" lines the Python side (possession/log.py: Progress) writes every
# few seconds; a step without a percentage shows its last log line instead. If the log has not
# changed for MTAH_IDLE_WARN seconds the line says so. On a terminal the line is redrawn in place;
# when the output is piped it is printed as a new line every MTAH_WATCH_PLAIN_EVERY seconds.
#
# Knobs (environment): MTAH_WATCH_POLL (2 s), MTAH_WATCH_PLAIN_EVERY (30 s), MTAH_IDLE_WARN (120 s),
# MTAH_FORCE_TTY=1 (redraw in place even when stdout is not a terminal; used by the tests).

fmt_time() {                                   # seconds -> m:ss, or h:mm:ss from an hour on
  local s=$1
  if [ "$s" -ge 3600 ]; then printf '%d:%02d:%02d' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
  else printf '%d:%02d' $((s / 60)) $((s % 60)); fi
}

make_bar() {                                   # make_bar <percent> -> ######------ (24 cells)
  local pct=$1 w=24 f a b
  [ "$pct" -gt 100 ] && pct=100
  f=$((pct * w / 100))
  a="$(printf '%*s' "$f" '')"; b="$(printf '%*s' $((w - f)) '')"
  printf '%s%s' "${a// /#}" "${b// /-}"
}

strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }
tidy()       { sed -E 's/^[0-9:]+ +\[[a-z ]+\] +//; s/ +/ /g'; }      # drop the clock and level the logger prints first

kill_tree() {                                  # stop a step and everything it started
  local pid=$1 win
  if [ -r "/proc/$pid/winpid" ]; then          # Git Bash / MSYS: native Windows children need taskkill
    win="$(cat "/proc/$pid/winpid")"
    taskkill //F //T //PID "$win" >/dev/null 2>&1
  else
    pkill -TERM -P "$pid" 2>/dev/null
  fi
  kill "$pid" 2>/dev/null
}

watch_step() {                                 # watch_step <pid> <name> <logfile> <start-in-SECONDS>
  local pid=$1 name=$2 log=$3 t0=$4
  local poll=${MTAH_WATCH_POLL:-2} plain_every=${MTAH_WATCH_PLAIN_EVERY:-30} idle_warn=${MTAH_IDLE_WARN:-120}
  local tty=0 next_plain=0 cols line pct detail text idle mtime elapsed
  { [ -t 1 ] || [ "${MTAH_FORCE_TTY:-0}" = 1 ]; } && tty=1
  cols=${COLUMNS:-$(tput cols 2>/dev/null || echo 110)}
  while kill -0 "$pid" 2>/dev/null; do
    line="$(tail -n 400 "$log" 2>/dev/null | strip_ansi | grep -aE '[0-9]+%.*ETA' | tail -1)"
    pct=""
    if [ -n "$line" ]; then
      pct="$(printf '%s' "$line" | grep -oE '[0-9]+%' | tail -1 | tr -d '%')"
      detail="$(printf '%s' "$line" | tidy)"
    else
      detail="$(tail -n 30 "$log" 2>/dev/null | strip_ansi | grep -avE '^[[:space:]]*$' | tail -1 | tidy)"
    fi
    elapsed=$((SECONDS - t0))
    if [ -n "$pct" ]; then text="  $name [$(make_bar "$pct")] $detail | $(fmt_time "$elapsed")"
    else text="  $name  $detail | $(fmt_time "$elapsed")"; fi
    mtime="$(stat -c %Y "$log" 2>/dev/null || stat -f %m "$log" 2>/dev/null || echo 0)"
    idle=$(( $(date +%s) - mtime ))
    if [ "$idle" -ge "$idle_warn" ]; then text="$text | no log output for $(fmt_time "$idle"), still running"; fi
    if [ "$tty" = 1 ]; then
      printf '\r%s\033[K' "${text:0:$((cols - 1))}"
    elif [ "$SECONDS" -ge "$next_plain" ]; then
      printf '%s\n' "$text"; next_plain=$((SECONDS + plain_every))
    fi
    sleep "$poll"
  done
  [ "$tty" = 1 ] && printf '\r\033[K'
  return 0
}

watch_jobs() {                                 # watch_jobs <start-in-SECONDS> <rundir> <name>... until every <rundir>/<name>.exit exists
  local t0=$1 dir=$2; shift 2
  local poll=${MTAH_WATCH_POLL:-2} plain_every=${MTAH_WATCH_PLAIN_EVERY:-30}
  local tty=0 next_plain=0 cols name text left stage
  { [ -t 1 ] || [ "${MTAH_FORCE_TTY:-0}" = 1 ]; } && tty=1
  cols=${COLUMNS:-$(tput cols 2>/dev/null || echo 110)}
  while :; do
    left=0; text=""
    for name in "$@"; do
      if [ -f "$dir/$name.exit" ]; then
        text="$text | $name done"
      else
        left=$((left + 1))
        stage="$(tail -n 30 "$dir/$name.log" 2>/dev/null | strip_ansi | grep -avE '^[[:space:]]*$' | tail -1 | tidy | cut -c1-22)"
        text="$text | $name ${stage:-starting}"
      fi
    done
    [ "$left" = 0 ] && break
    text="  predictions $(fmt_time $((SECONDS - t0)))$text"
    if [ "$tty" = 1 ]; then
      printf '\r%s\033[K' "${text:0:$((cols - 1))}"
    elif [ "$SECONDS" -ge "$next_plain" ]; then
      printf '%s\n' "$text"; next_plain=$((SECONDS + plain_every))
    fi
    sleep "$poll"
  done
  [ "$tty" = 1 ] && printf '\r\033[K'
  return 0
}
