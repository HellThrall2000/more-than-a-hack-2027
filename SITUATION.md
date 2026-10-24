# SITUATION — when the ball is dead, and who has it when it is not

The labelling rulebook for this solution. It turns the hackathon's definitions (CHALLENGE.md,
EVALUATION.md, FAQ.md, DATA.md) and the Laws of the Game into a short list of decisions the
algorithm implements. It is also the "possession rules" section of the technical document, and
the guide for anyone hand-labelling a validation window (PLAN.md section 7).

**One sentence.** The ball is dead (`D`) from the moment the referee's law takes it out of play
until the restart is taken; at every other graded frame exactly one team has it (`H` or `A`), and
that team changes only when the other team gains *control*.

---

## 1. What the hackathon fixes — not ours to decide

| fact | source |
|---|---|
| Labels: `H` home in possession, ball alive · `A` away in possession, ball alive · `D` ball not in play · `X` no answer | CHALLENGE.md |
| **Three states only.** No "contested", "loose", "transition" or "unknown" in the ground truth | CHALLENGE.md |
| **Who owns a dead ball is not asked.** When the ball is dead the answer is `D`, even though the tracking names a restarting team | CHALLENGE.md |
| **The referee decides when the ball is in play.** While it is, one team is in possession | CHALLENGE.md |
| A loose or contested ball **stays with the last team** that had it | CHALLENGE.md |
| After a stoppage there is always a clear possessor: the restarting team | CHALLENGE.md |
| A change needs **controlled** possession by the opponent. A touch without control, a deflection, or an aerial duel won without follow-up control is not a change | CHALLENGE.md |
| `D` **splits possessions**: `H D H` is two possessions, and so is `A D A` | CHALLENGE.md, `possession_metrics.py` |
| The ground truth is mechanical: `D` wherever Metrica's *corrected* tracking marks the ball dead, otherwise the team it names as owner. No smoothing | CHALLENGE.md |
| Where the wording and the public ground truth disagree, **the ground truth is right** | CHALLENGE.md |
| Graded: video frames 11185–90651 and 110808–190197. Everything else is ungraded → `X` | DATA.md |
| First half: graded from the kick-off itself. Second half: 110808–113741 are graded and **all `D`**; the kick-off is 113742 | FAQ.md |

## 2. What the public 5 minutes show

The committed ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, frames
11185–18684) has 21 segments: 17 possessions and 4 stoppages.

| stoppage | video frames | on-screen clock (±1 s) | length | before → after |
|---|---|---|---|---|
| 1 | 12851–12988 | 01:06.6–01:12.1 | 5.5 s | `A` → `D` → `H` |
| 2 | 15889–16399 | 03:08.2–03:28.6 | 20.4 s | `A` → `D` → `A` |
| 3 | 17380–17837 | 04:07.8–04:26.1 | 18.3 s | `H` → `D` → `A` |
| 4 | 18027–18388 | 04:33.7–04:48.1 | 14.5 s | `A` → `D` → `H` |

What that settles:
- **No `D` at the first kick-off.** Frame 11185 is already `A`: the label is alive from the frame
  the kick-off is struck. Athletic kicked off the first half, so by Law 8 **Barcelona kicks off the
  second** (113742).
- **The same team can restart** (stoppage 2). That still produces two possessions.
- **"Control" is quick.** The shortest possessions last 19 and 33 frames (0.76 s and 1.32 s). The
  ground truth does not wait "a few seconds": a team holds the ball as soon as one of its players
  has it under control.
- Stoppages here run 5.5–20.4 s. That describes these four only, not a prior. Quick free kicks
  and throw-ins can be much shorter.

The type of each stoppage (throw-in, foul, …) is not in the ground truth. Recover it with the
local semantics study (section 8). Do not paste data extracts into a hosted assistant (FAQ.md).

### Video frames and the on-screen match clock

The tactical video (`data/tactical/BAR-ATH_tactical.mp4`, the only video the solution reads) shows
a match clock in its top-left corner. It starts at 00:00 on the first-half kick-off and runs with
the match. Around half-time it freezes or disappears, and there is no clock while the teams are
off the pitch. **Labels are indexed by video frame, not by that clock.** Row *i* of the submission
is frame *i* of the file, counted from the very first frame, pre-match included.

| period | video frames | on-screen clock | label |
|---|---|---|---|
| before the match: walk-on, anthems | 0–11184 | none | `X` (ungraded) |
| **1st half** | **11185–90651** | 00:00 → 52:58 (added time runs past 45:00) | `H` / `A` / `D` |
| half-time | 90652–110807 | frozen or none | `X` (ungraded): **skipped** |
| 2nd half, before its kick-off | 110808–113741 | none | **`D`: graded — do NOT skip** |
| **2nd half** | **113742–190197** | 45:00 → 95:58 | `H` / `A` / `D` |
| after the final whistle | 190198–199318 | none | `X` (ungraded) |

- **"No clock" is not "skip".** The 2,934 frames before the second-half kick-off have no clock
  running, but they are graded and all `D` (FAQ.md). Writing `X` there is worse than a miss: the
  scorer back-fills it with the first label of the half, which stretches the first possession by
  about 117 s.
- **Readings from 45:00 up appear in both halves**, so always say which half: `51:35` in the second
  half is frame 123617.
- **The clock runs through every stoppage** (fouls, throw-ins, ball out of play), checked on the
  video. It is absent only before the match (walk-on, anthems) and at half-time, so it carries no
  dead-ball information inside a half. It is ruled out as a `D` signal; do not build an OCR
  detector for it.
- Convert either way with `python -m possession.clock 51:35 --half 2` or
  `python -m possession.clock --frame 12851`.

---

## 3. DEAD BALL — label `D`

Law 9.1: the ball is out of play when it has **wholly** crossed the touchline or goal line, on the
ground or in the air, or when **the referee stops play**. It comes back into play at the restart
(Law 8 and the restart laws).

| situation | `D` starts | `D` ends (first alive frame) | team after `D` |
|---|---|---|---|
| ball wholly over a **touchline** | the frame it is wholly over | throw-in: the ball **enters the field** from the thrower's hands | opponent of the last toucher |
| ball wholly over a **goal line**, attacker touched last | crossing frame | goal kick **kicked and clearly moves** | defending team |
| ball wholly over a **goal line**, defender touched last | crossing frame | corner **kicked and clearly moves** | attacking team |
| **goal** (own goals included) | the ball wholly over the line between the posts | the **kick-off** is kicked | the team that conceded |
| **foul, handball** → free kick or penalty | the stoppage (see S2) | free kick kicked and moves / penalty kicked and moves forward | the team fouled |
| **offside** | the stoppage (whistle, not the flag) | indirect free kick kicked | the defending team |
| other whistle: **injury, misconduct, second ball, referee hit by the ball** (Law 9.1c), VAR review called during live play | the stoppage | dropped ball **touches the ground**, or the restart the decision gives | dropped ball: team that last touched (the defending goalkeeper if it happened in their area) |
| **second-half wait** 110808–113741 | 110808 | 113742 | Barcelona (`H`) |
| **chained stoppages**: goal → VAR → disallowed → free kick; foul → card → substitution → free kick | the first stoppage | the restart that finally puts the ball back in play | as the final restart |

**Everything during a stoppage is `D`, however the ball moves:** fetching it, rolling it to the
taker, a ball boy's replacement ball, the goalkeeper placing it, a wall forming, a substitution,
treatment, a card, a VAR check. A ball moving next to players is not a restart. A restart is the
taker striking (or throwing) a ball that was set for it.

## 4. BALL IN PLAY — label `H` or `A`, never `D`

Law 9.2: the ball is in play at all other times.

| situation | label |
|---|---|
| ball **on** the line: any part still over the field | in play, team unchanged |
| rebound off a **post, crossbar or corner flag** that stays on the field | in play, team unchanged until someone controls it |
| ball hits the **referee** and stays on, with none of the Law 9.1c outcomes | in play, team unchanged |
| foul with **advantage**, or a foul the referee does not whistle | in play; `D` only if and when play is stopped |
| players stop or appeal, but **no whistle** | in play |
| delayed **offside flag**: play runs on until the whistle | in play until the stoppage |
| player down but play goes on | in play |
| **goalkeeper holding** the ball (in hands, bouncing it, about to release it) | in play, goalkeeper's team |
| **loose or contested ball**, scramble, aerial duel, blocked shot staying on the field | in play, last team in possession |
| **ball in flight**: long ball, cross, clearance staying on the field | in play, passer's team until an opponent controls it |
| ball **not visible** in the tracking | in play, previous label, unless there is positive evidence of a stoppage or a change |
| penalty rebound off the post or goalkeeper | in play |

## 5. Who has the ball while it is alive — `H` ↔ `A`

| counts as a change (to the new team) | is NOT a change (stays with the last team) |
|---|---|
| clean reception or interception by an opponent | deflection that does not stop the pass |
| tackle won **and** the ball kept by the tackler's team | tackle that knocks the ball loose |
| goalkeeper **catches** or holds the ball | goalkeeper parry or punch; the ball stays loose |
| opponent's controlled first touch, including a first-time pass that **reaches a teammate** | touch without control, miscontrol, ricochet |
| a loose ball **picked up and controlled** by the other team | aerial duel won without follow-up control |

**Timing:** the change lands on the controlling touch, not when control is "confirmed" later. A
lag of 12 frames already costs about 0.1 spellF1 (PLAN.md section 1).

## 6. Restart team (`D` → `H`/`A`) — summary

| restart | team | first alive frame |
|---|---|---|
| kick-off, start of a half | 1st half: Athletic (`A`), seen in the public ground truth; 2nd half: Barcelona (`H`), Law 8 | 11185 / 113742 |
| kick-off after a goal | the team that conceded | kicked and clearly moves |
| throw-in | opponent of the last toucher | ball enters the field |
| goal kick | defending team | kicked and clearly moves |
| corner | attacking team | kicked and clearly moves |
| free kick / penalty | team fouled; for offside, the defending team | kicked and clearly moves |
| dropped ball | team that last touched | ball touches the ground |

**Fallback order when the type is unclear:** the Smart Tagging `SET PIECES` team (throw-ins are
tagged `N/A`, so infer those) → restart formation and ball position → opponent of the last
toucher.

---

## 7. The simplifications we commit to

These are operational choices where the rules can't be read off the tracking directly. Each has a
reason in the metric.

| # | decision | why |
|---|---|---|
| **S1** | `D` needs **positive evidence**: ball out of play, a ball at rest with players walking, a whistle (collective deceleration), or a set-piece or goal tag. **A missing ball is never evidence** | A false `D` inside a long possession splits it in two, the most expensive error there is (PLAN.md section 1) |
| **S2** | `D` starts at the **line crossing** for out-of-play balls. For whistled stoppages it starts at the **onset of collective deceleration**, plus one global offset calibrated on validation windows | The whistle cannot be heard in the data; deceleration is the nearest observable. Pending Q1 |
| **S3** | `D` ends on the **frame the ball leaves rest** at a restart spot or formation, or re-enters over the touchline | The ground truth is alive from the strike (the first kick-off shows it). Pending Q2 |
| **S4** | One stoppage is **one `D` run**. No `D` `A` `D` inside a stoppage. The ball being moved during set-up is not a restart | A spurious mini-possession in a stoppage adds a false possession and splits the `D` |
| **S5** | `D` can be **short**: quick free kicks and throw-ins of about 1 s are allowed when the evidence is strong. The duration prior is soft | Missing a stoppage merges two possessions; never predicting `D` costs about 0.19 spellF1 |
| **S6** | **No team during `D`**, whatever the tracking, tags or formation say about who "owns" it. That information is used only to choose the team *after* the restart | CHALLENGE.md: dead-ball ownership is not asked |
| **S7** | **Same-team restarts are new possessions** (`A D A`). The decoder never merges across `D` | That is how the metric counts them |
| **S8** | A team change needs an opponent's **control onset**: control for at least about 4 frames, or a touch followed by a teammate's control. No minimum possession length beyond that | The public window has 19-frame possessions; a long minimum would delete them |
| **S9** | An out-of-play ball needs **persistence**: about 0.5 m past the line for at least 10 observed frames, and longer on the far touchline | Airborne balls project past the line from the camera; a ball on the line is still in play |
| **S10** | **Fixed frames:** 11185 (clock 00:00) is alive, the first-half kick-off; 110808–113741 (no clock) is `D`; 113742 (clock 45:00) is alive. Ungraded frames, half-time included → `X`; graded frames are never `X` | Published boundaries (FAQ.md), not labels learned from the ground truth |

## 8. Open questions — settled by measurement, not by tuning

To answer these, measure the 20 boundaries of the public window (12 turnovers, 4 stoppages,
4 restarts) against what our detectors fire on. **Do it on a local machine and keep the output
local.** This is understanding the definition, which CHALLENGE.md asks for. It is not choosing
between versions: 17 possessions cannot rank anything (EVALUATION.md).

| # | question | how to tell |
|---|---|---|
| Q1 | Does ground-truth `D` start at the **line crossing / foul**, or at the **whistle / visible stop**? | Signed offset from each true `D` start to our out-of-play onset and to our deceleration onset |
| Q2 | Does ground-truth alive start **on the strike**, or some frames after? | Offset from each true restart to the frame the ball leaves rest |
| Q3 | Does a turnover land on the **first touch** of the new team, or after k frames of control? | Offset from each true turnover to the new team's nearest touch and control onset |
| Q4 | What type was each of the 4 stoppages? | Overlap with Smart Tagging `SET PIECES` `Type`, and the ball's position at the stop |

Settle Q1–Q3 once; they become `stop_offset`, the restart offset and `turnover_offset` in
`possession/config.py`. Then confirm them on the hand-labelled windows (PLAN.md section 7). Never
re-fit them to the public window.

## 9. Where each decision lives in the code

| decision | code | status |
|---|---|---|
| out of play, ball at rest | `ball/events.py: out_of_play` | built |
| whistle (deceleration), at rest with walking players, out-of-play stoppage candidates | `candidates.py` | built |
| restart = ball leaving rest next to a player; formations | `candidates.py`, `restarts.py` | built |
| "play has stopped" from the players alone (S1, S2): no ball needed | `calm.py`, `emission_rules.py: calm_evidence`, `candidates.py` | built |
| restart side by stoppage kind: ball out → opponent of the last toucher; whistle → team in control before it | `candidates.py: _side_pref` | built; an offside (→ defending team) still goes the wrong way |
| grammar: no same-label neighbours, so a stoppage is one `D` run | `decode.py` | built |
| fixed second-half `D` prefix, alive from the kick-off | `decode.py: decode_match`, `ingest.py` | built |
| kick-off team alternates by half (Smart Tagging "Kick Off (Start)" + Law 8) | `tags.py: kickoff_teams`, `decode.py` | built, as a preference |
| kick-off after a goal → the team that conceded | — | **to do** |
| extrapolated line crossing for a ball lost on its way out | `ball/events.py: extrapolated_crossings` | built |
| S4 guard: no restart while the set-up is still in progress | — | **to do**: restarts currently need only "leaves rest near a player" |
| offsets Q1–Q3 | `config.py: stop_offset, restart_offset, turnover_offset` | placeholders = 0, set from measurements only |
