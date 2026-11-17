"""The on-screen match clock <-> video frame, the submission's row index.

    python -m possession.clock 51:35 --half 2         # clock -> video frame
    python -m possession.clock --frame 12851          # video frame -> clock

The tactical video shows a match clock in its top-left corner. It reads 00:00 at the first-half
kick-off and 45:00 at the second-half kick-off, and runs on through stoppages and into added time.
Around half-time it is frozen or hidden. **Labels are indexed by video frame, never by this
clock.** This module only converts between the two: a moment seen on screen ("51:35") can be found
in the data, and a frame can be looked up in the video.

Anchors (FAQ.md, DATA.md): clock 00:00 = video frame 11185, clock 45:00 = video frame 113742, at
25 fps. The clock shows whole seconds, so a converted frame is good to about +/- 1 s.

Two traps:
  * Clock readings from 45:00 up exist in BOTH halves (first-half added time runs past 45:00), so
    `half` is required there.
  * Frames 110808-113741 come before the second-half clock starts, yet they are graded, all `D`.
    "No clock on screen" does not mean "skip": only the half-time frames 90652-110807 are ungraded.
"""

from __future__ import annotations

import argparse
import sys

from . import ROOT  # noqa: F401  (puts the repository root on sys.path)
from .config import DEFAULT
from games import GAME, half_video_blocks

FPS = DEFAULT.fps
HALF_START_CLOCK = {1: 0, 2: 45 * 60}                    # clock seconds at each kick-off
BLOCKS = {half: (first, last) for half, first, last in half_video_blocks(GAME)}
KICKOFF = {1: BLOCKS[1][0], 2: DEFAULT.second_half_kickoff}


def parse_clock(text):
    """'51:35' or '51:35.4' -> seconds (float)."""
    minutes, _, seconds = str(text).strip().partition(":")
    if not seconds:
        raise ValueError("expected mm:ss, got {!r}".format(text))
    return int(minutes) * 60 + float(seconds)


def format_clock(seconds):
    minutes, sec = divmod(round(seconds, 1), 60.0)      # round first: 299.96 s is 05:00.0, not 04:60.0
    return "{:02d}:{:04.1f}".format(int(minutes), sec)


def clock_to_frame(clock, half=None):
    """Video frame shown at match-clock reading `clock` ('mm:ss') in `half`."""
    sec = parse_clock(clock)
    if half is None:
        if sec >= HALF_START_CLOCK[2]:
            raise ValueError("{} is ambiguous: first-half added time and the second half both show "
                             "it -- pass half=1 or half=2".format(clock))
        half = 1
    if sec < HALF_START_CLOCK[half]:
        raise ValueError("{} is before the kick-off of half {}".format(clock, half))
    frame = KICKOFF[half] + int(round((sec - HALF_START_CLOCK[half]) * FPS))
    if frame > BLOCKS[half][1]:
        raise ValueError("{} is after the end of half {} (last graded frame {} = {})".format(
            clock, half, BLOCKS[half][1], format_clock(frame_to_clock(BLOCKS[half][1])[1])))
    return frame


def frame_to_clock(frame):
    """(half, clock seconds) for a frame on the running clock, else (None, reason)."""
    for half, (first, last) in BLOCKS.items():
        if KICKOFF[half] <= frame <= last:
            return half, HALF_START_CLOCK[half] + (frame - KICKOFF[half]) / float(FPS)
        if first <= frame < KICKOFF[half]:
            return None, "half {} before its kick-off: no clock, but graded and all D".format(half)
    if frame < BLOCKS[1][0]:
        return None, "before the match: ungraded (X)"
    if frame < BLOCKS[2][0]:
        return None, "half-time: ungraded (X)"
    return None, "after the final whistle: ungraded (X)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clock", nargs="?", help="match clock as mm:ss")
    ap.add_argument("--half", type=int, choices=(1, 2))
    ap.add_argument("--frame", type=int, help="convert a video frame instead")
    args = ap.parse_args(argv)
    if args.frame is not None:
        half, value = frame_to_clock(args.frame)
        print("video frame {} -> {}".format(args.frame, value if half is None else
                                             "half {}, clock {}".format(half, format_clock(value))))
    elif args.clock:
        print("half {}, clock {} -> video frame {}".format(
            args.half or 1, args.clock, clock_to_frame(args.clock, args.half)))
    else:
        ap.error("give a clock reading or --frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
