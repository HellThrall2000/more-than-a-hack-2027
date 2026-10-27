# SITUATION — when the ball is dead, and who has it when it is not

The labelling rulebook for this solution. It turns the hackathon's definitions (CHALLENGE.md,
EVALUATION.md, FAQ.md, DATA.md), the **IFAB Laws of the Game 2026/27** and the industry's
possession conventions (Opta / Stats Perform, FIFA Enhanced Football Intelligence) into a short
list of decisions the algorithm implements. It is also the "possession rules" section of the
technical document, and the guide for anyone hand-labelling a validation window (PLAN.md
section 7).

**One sentence.** The ball is dead (`D`) from the moment the Laws take it out of play until the
restart is taken; at every other graded frame exactly one team has it (`H` or `A`), and that team
changes only when the other team gains *control*.

## Order of authority

When two sources disagree, the higher one wins:

1. **The organiser's ground truth** (the public 5 minutes, and the private full match it stands for).
   "If your reading of the definition disagrees with it, the ground truth is right" (CHALLENGE.md).
2. **The organiser's written definitions** (CHALLENGE.md, FAQ.md, DATA.md).
3. **IFAB Laws of the Game 2026/27**, in force from 1 July 2026: the current edition, and the
   reference text of this rulebook. See "Law edition" below for the one way the match itself
   limits it.
4. **Industry possession conventions**: Opta / Stats Perform sequences and possessions, FIFA EFI
   possession states. The Laws say when the ball is in play; they do not define team possession,
   so these fill that gap.
5. **Our simplifications** (section 7): operational choices where the tracking cannot show the
   rule directly.

Where a convention is not a Law (for example the fair-play kick-out after an injury) this document
says so.

### Law edition: 2026/27 rules, a 2024/25 match

The match (LaLiga 2024-25, matchday 2, August 2024) was refereed under the **2024/25** Laws, and
the ground truth records what actually happened on the pitch. The definitions this rulebook
depends on most — when the ball is out of play (Law 9.1), when each restart puts it back in play
(Laws 8, 13–17), what a goal is (Law 10) — are **the same in both editions**. The differences are
all about *which restart* a team is given:

| situation | 2024/25 (refereed in this match) | 2026/27 (this rulebook) | effect on a label |
|---|---|---|---|
| goalkeeper holds the ball too long | over 6 s: indirect free kick to the opponents | over 8 s: **corner** to the opponents (Law 17) | none: the same team restarts; only the restart type differs |
| team delays a throw-in | caution only | 5-second visual countdown, then the **throw-in goes to the opponents** (Law 15) | could not happen in this match |
| team delays a goal kick | caution only | 5-second visual countdown, then a **corner to the opponents** (Laws 16, 17) | could not happen in this match |
| dropped ball outside the penalty area | team that **last touched** the ball | team that **has or would have kept or gained possession**, including the team that would have taken the restart; otherwise the last toucher (Law 8.2) | can differ — see the rule below |
| dropped ball after the ball hits a match official or outside agent | position where it was last touched | **position of the contact** (Law 8.2) | none: position only |
| substituted player leaving | no time limit | must leave within **10 s** (Law 3) | none: happens during `D` |
| injured player treated on the field | may stay on (except some cases) | must leave for **one minute** after play restarts, with exceptions (Law 5) | none for labels; fewer tracked players after the restart |
| off-field person (substitute, team official, temporarily off player) touches a ball clearly leaving the field | — | indirect free kick, no sanction (Law 9.2) | `D` from the touch, restart to the opponents |

**Rule: observed beats predicted.** Where the restarting team can be seen (who throws, who takes
the kick, who receives the dropped ball), the label follows what happened, whichever edition's
rule it agrees with. The 2026/27 rules are used to *predict* a restart team only where the
tracking cannot show it; for a dropped ball, where the two editions can disagree, an observed
receiver always wins.

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

### Where the organiser departs from industry conventions

| industry convention | this challenge | consequence |
|---|---|---|
| Opta: a possession runs **through** stoppages until the opponent gains control (a shot saved for a corner and the corner are one possession) | `D` splits possessions | a same-team restart is a **new** possession (S7) |
| FIFA EFI: a fourth state, **In Contest** — in play, neither team in control | no contest label | a contested ball is labelled with the **last team in control**; a contest state may exist inside the decoder only (S11) |
| Traditional TV possession %: time or completed passes | frame-by-frame team label | no percentage, no pass counting |

---

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
- **Passes and flights are not `D`.** 17 possessions of mean 14.2 s contain many passes, yet there
  are only 4 `D` runs, all real stoppages of 5.5–20.4 s.
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

**Law 9.1** — the ball is out of play when:
- it has **wholly** passed over the goal line or touchline, on the ground or in the air;
- play has been **stopped by the referee**;
- it touches a **match official**, remains on the field, and a team starts a promising attack, or
  the ball goes directly into the goal, or the team in possession changes (restart: dropped ball).

It comes back into play at the restart. Every restart has its own "in play" moment:

| restart | Law | ball is in play when |
|---|---|---|
| kick-off | 8.1 | it is **kicked and clearly moves** |
| dropped ball | 8.2 | it **touches the ground** |
| free kick (direct or indirect) | 13 | it is **kicked and clearly moves**. A defending team's free kick in its own penalty area does not have to leave the area |
| penalty kick | 14 | it is **kicked and clearly moves forward** |
| throw-in | 15 | it **enters the field of play**. A throw that does not enter is retaken (still `D`) |
| goal kick | 16 | it is **kicked and clearly moves**. It does not have to leave the penalty area |
| corner kick | 17 | it is **kicked and clearly moves**. It does not have to leave the corner area |

### 3.1 When `D` starts, and who restarts

| situation | `D` starts | `D` ends (first alive frame) | team after `D` | Law |
|---|---|---|---|---|
| ball wholly over a **touchline** | the frame it is wholly over | throw-in enters the field | opponents of the **last player to touch** it | 9, 15 |
| ball wholly over the **goal line, not a goal**, attacker touched last | crossing frame | goal kick kicked and clearly moves | defending team | 9, 16 |
| ball wholly over the **goal line, not a goal**, defender touched last (goalkeeper included) | crossing frame | corner kicked and clearly moves | attacking team | 9, 17 |
| **goal**: ball wholly over the goal line **between the posts and under the crossbar**, own goals included | crossing frame | the **kick-off** is kicked | the team that **conceded** | 10, 8 |
| **foul or handball** → direct free kick or penalty | the stoppage (see S2) | free kick kicked and moves / penalty kicked and moves forward | the team **fouled** | 12, 13, 14 |
| **offside** | the stoppage: the whistle, not the flag | indirect free kick kicked and moves, at the offence position | the **defending** team | 11, 13 |
| **goalkeeper controls the ball with the hands for over 8 seconds** | the stoppage | **corner** kicked and clearly moves | the **attacking** team | 12, 17 |
| goalkeeper handles a deliberate back-pass or a throw-in from a team-mate | the stoppage | indirect free kick kicked and moves | the **attacking** team | 12, 13 |
| team **delays a throw-in** past the referee's 5-second countdown | the original crossing (the ball never came back into play) | throw-in by the **other** team enters the field | the **opposing** team | 15 |
| team **delays a goal kick** past the referee's 5-second countdown | the original crossing | **corner** kicked and clearly moves | the **opposing** (attacking) team | 16, 17 |
| substitute, team official or temporarily-off player touches a ball **clearly leaving** the field | the touch | indirect free kick kicked and moves | the opponents of the person who touched it | 9.2 |
| **injury**: referee stops play | the stoppage | dropped ball touches the ground, or the restart the decision gives | dropped ball: see below | 5, 8.2 |
| **outside interference**: extra ball, spectator, object on the field, ball hits the referee as in Law 9.1 | the stoppage | dropped ball touches the ground | see below | 3, 8.2, 9.1 |
| **VAR review called** after the ball goes out or play is stopped | the original stoppage | the restart the decision gives | as that restart | VAR protocol |
| **VAR**: referee stops live play for a review | the stoppage | the decision's restart, or a dropped ball if no offence | as that restart | VAR protocol |
| **second-half wait** 110808–113741 | 110808 | 113742 | Barcelona (`H`) | FAQ.md, Law 8 |
| **chained stoppages**: goal → VAR → disallowed → free kick; foul → card → substitution → free kick | the first stoppage | the restart that finally puts the ball back in play | as the final restart | — |

**Dropped ball (Law 8.2, 2026/27).** The ball is dropped for the **defending goalkeeper** if,
when play stopped, the ball was in the penalty area. Otherwise it is dropped for **one player of
the team that has or would have kept or gained possession**, including the team that would have
taken the restart had the ball gone out of play, if the referee can determine it; otherwise for
the team that **last touched** it. It is dropped where the ball was when play stopped, or where it
hit a match official or outside agent. All other players stay at least 4 m away. It is in play
when it touches the ground. (In this 2024/25 match the referee applied "last touched"; an observed
receiver wins either way.)

**Substitutions** (Law 3: the outgoing player has 10 s to leave) and **cautions / sending-offs**
(Law 12) happen only while the ball is out of play, so they always sit inside a `D` run, never
start one on their own. An **injured player treated on the field** must leave and stay off for one
minute after the restart (Law 5, 2026/27): that is a change in the number of tracked players, not
a stoppage.

**Everything during a stoppage is `D`, however the ball moves:** fetching it, rolling it to the
taker, a ball boy's replacement ball, the goalkeeper placing it, a wall forming, a substitution,
treatment, a card, a VAR check. A ball moving next to players is not a restart. A restart is the
taker striking (or throwing) a ball that was set for it.

**A retaken restart** (throw-in that does not enter the field, encroachment at a penalty, a free
kick taken before the whistle when one was required) keeps the ball out of play: one `D` run until
the valid restart.

---

## 4. BALL IN PLAY — label `H` or `A`, never `D`

**Law 9.2**: the ball is in play at all other times, including when it touches a match official
and when it rebounds off a goalpost, crossbar or corner flagpost and remains on the field.

| situation | label |
|---|---|
| ball **on** the line: any part still over the field (on the ground or in the air) | in play, team unchanged |
| ball **curls outside a line in the air and back in** | **out** — it wholly passed over the line (Law 9.1). `D` from the crossing |
| rebound off a **post, crossbar or corner flag** that stays on the field | in play, team unchanged until someone controls it |
| ball hits the **referee** and stays on, with none of the Law 9.1 outcomes | in play, team unchanged |
| foul with **advantage** (Law 5), or a foul the referee does not whistle | in play; `D` only if and when play is stopped |
| players stop or appeal, but **no whistle** | in play |
| delayed **offside flag**: play runs on until the whistle | in play until the stoppage |
| player down but play goes on | in play |
| **ball in flight**: pass, long ball, cross, shot, clearance staying on the field | in play, **the team that played it**, until an opponent controls it (section 5.1) |
| **loose or contested ball**, scramble, aerial duel, blocked shot staying on the field | in play, last team in possession |
| **goalkeeper holding** the ball (in hands, bouncing it, about to release it) | in play, goalkeeper's team |
| ball **not visible** in the tracking | in play, previous label, unless there is positive evidence of a stoppage or a change |
| penalty rebound off the post or goalkeeper | in play |

**`D` means "out of play", not "nobody is touching it".** A ball travelling between players is
alive and belongs to the team that last controlled it.

---

## 5. Who has the ball while it is alive — `H` ↔ `A`

The Laws define when the ball is in play, not which team has it. The industry convention (Opta /
Stats Perform; FIFA EFI) and CHALLENGE.md agree:

- **A possession belongs to a team, not a player.** Every action by the same team keeps the label:
  dribbles, passes, tricks, shielding, headers to a team-mate, a goalkeeper's distribution.
- **A possession ends only when the opponent gains control of the ball** (Opta), or when the ball
  goes out of play (this challenge).
- **A defensive action alone is not control.** A tackle or interception starts a possession only
  if a controlled action follows it, such as a pass, carry or shot (Opta).
- **The change is timed at the first ball-control event of the new team** (FIFA EFI "ball recovery
  time": last control of one team to first control of the next).

| counts as a change (to the new team) | is NOT a change (stays with the last team) |
|---|---|
| clean reception or interception by an opponent | deflection that does not stop the pass |
| tackle won **and** the ball kept by the tackler's team | tackle that knocks the ball loose |
| goalkeeper **catches** or holds the ball | goalkeeper parry or punch; the ball stays loose |
| opponent's controlled first touch, including a first-time pass that **reaches a teammate** | touch without control, miscontrol, ricochet |
| a loose ball **picked up and controlled** by the other team (Opta "recovery") | aerial duel won without follow-up control |
| a pass played directly to an opponent, who controls it | an opponent's block that sends the ball loose |

**Timing:** the change lands on the controlling touch, not when control is "confirmed" later. A
lag of 12 frames already costs about 0.1 spellF1 (PLAN.md section 1). The decision may use later
frames (section 10.2), but the label is written back to the controlling touch.

### 5.1 The ball in flight

| flight | label during flight | ends |
|---|---|---|
| pass or long ball to a team-mate | passer's team | team-mate's reception: **no change** |
| pass intercepted | passer's team | **change at the interception** (the opponent's controlling touch) |
| cross, cleared by a defender's header with no follow-up control | attacking team | no change; the contest continues |
| cross, headed and controlled by a defender or caught by the goalkeeper | attacking team | **change at that control** |
| shot | shooting team | goal / out (`D` at the crossing), a catch (change), or a loose rebound (no change) |
| clearance staying on the field | clearing team (only if it had control) — else last team in control | next controlled touch |

Aerial duel without follow-up control is not a change (CHALLENGE.md). A header that reaches a
team-mate who controls it **is** a change, timed at the header, because it is the new team's first
controlled action.

### 5.2 Shots, goalkeeper actions and goals

Law 12 treats a goalkeeper as **in control of the ball with the hands** when the ball is between
the hands, or between a hand and any surface (ground, own body); when holding it in an outstretched
open hand; or when bouncing it or throwing it into the air. It is **not** in control when the ball
**rebounds from the goalkeeper or the goalkeeper has made a save**. That is the line between a save
that wins the ball and one that does not.

| after the shot | label | change happens at |
|---|---|---|
| **goalkeeper catches or holds it** | goalkeeper's team | **the catch frame** |
| goalkeeper parries or punches; **a defender** then controls it | goalkeeper's team | **the defender's control**, not the parry |
| goalkeeper parries; **an attacker** collects it | shooting team | no change at all |
| goalkeeper parries; ball loose in a scramble | shooting team | until someone controls it |
| goalkeeper **tips it over the bar or round the post** | `D` from the crossing; **corner**, attacking team (a new possession) | — |
| goalkeeper pushes it **over the touchline** | `D` from the crossing; throw-in, attacking team | — |
| goalkeeper touches it **and it goes in** | shooting team, then `D` from the crossing; kick-off by the conceding team | — |
| shot **wide or over**, no defender touch | `D` from the crossing; goal kick, defending team | — |
| shot **blocked by a defender** and out over the goal line | `D` from the crossing; corner, attacking team | — |
| goal **disallowed by VAR** | `D` from the crossing; the decision's restart (free kick to defenders, goal kick, …) | — |

**A goalkeeper's touch is never a change by itself.** It is a change only when it is a catch or
hold, or when the keeper's team controls the ball afterwards.

### 5.3 Dribbles, passes, tricks and duels

| action | label |
|---|---|
| dribble or carry, any number of touches | carrier's team |
| feint, step-over, turn, shielding with an opponent close | carrier's team — proximity of an opponent is not control |
| pass, through ball or one-two between team-mates | passing team throughout |
| tackle, ball loose | last team in control, until someone controls it |
| tackle, tackler's team keeps the ball | change at the tackler's (or team-mate's) controlling touch |
| player dispossessed: opponent takes the ball and carries on | change at the opponent's controlling touch |
| 50/50 that runs loose | last team in control |

### 5.4 Customs that are not Laws

| custom | how it looks | label |
|---|---|---|
| **fair-play kick-out**: a team kicks the ball out so an injured player can be treated, and the opponents' throw-in is given back | `D` (injury), throw-in by the opponents, then an immediate long throw or pass to the other team | label each part by the rules above: `D`, the throwing team, then a **change** when the other team controls it |
| ball returned to the opponents after a **dropped ball** | the receiving team plays it back | label by control: a new possession for whoever controls it |

---

## 6. Restart team (`D` → `H`/`A`) — summary

| restart | team | first alive frame | Law |
|---|---|---|---|
| kick-off, start of a half | 1st half: Athletic (`A`), seen in the public ground truth; 2nd half: Barcelona (`H`) | 11185 / 113742 | 8.1 |
| kick-off after a goal | the team that conceded | kicked and clearly moves | 8.1 |
| throw-in | opponents of the last player to touch the ball | ball enters the field | 15 |
| throw-in after a delayed throw-in (5-second countdown) | the opposing team | ball enters the field | 15 |
| goal kick | defending team | kicked and clearly moves | 16 |
| corner kick | attacking team; also after a goalkeeper's 8-second offence or a delayed goal kick | kicked and clearly moves | 17 |
| direct / indirect free kick | team fouled; for offside, the defending team; for a goalkeeper's back-pass offence, the attacking team | kicked and clearly moves | 12, 13 |
| penalty kick | team fouled | kicked and clearly moves forward | 14 |
| dropped ball | defending goalkeeper if in the penalty area; else the team that would have kept or gained possession, else the team that last touched; **an observed receiver wins** | ball touches the ground | 8.2 |

**Fallback order when the type is unclear:** the Smart Tagging `SET PIECES` team (throw-ins are
tagged `N/A`, so infer those) → restart formation and ball position (section 10.1) → opponent of
the last toucher.

---

## 7. The simplifications we commit to

These are operational choices where the rules can't be read off the tracking directly. Each has a
reason in the metric.

| # | decision | why |
|---|---|---|
| **S1** | `D` needs **positive evidence**: ball out of play, a ball at rest with players walking, a whistle (collective deceleration, or the audio whistle of section 10.3), or a set-piece or goal tag. **A missing ball is never evidence** | A false `D` inside a long possession splits it in two, the most expensive error there is (PLAN.md section 1) |
| **S2** | `D` starts at the **line crossing** for out-of-play balls. For whistled stoppages it starts at the **onset of collective deceleration**, plus one global offset calibrated on validation windows | The tracking cannot see the referee's decision; deceleration (and the whistle, if audible) is the nearest observable. Pending Q1 |
| **S3** | `D` ends on the **frame the ball leaves rest** at a restart spot or formation, or re-enters over the touchline | The Laws put the ball in play on the strike (the first kick-off shows the ground truth agrees). Pending Q2 |
| **S4** | One stoppage is **one `D` run**. No `D` `A` `D` inside a stoppage. The ball being moved during set-up is not a restart | A spurious mini-possession in a stoppage adds a false possession and splits the `D` |
| **S5** | `D` can be **short**: quick free kicks and throw-ins of about 1 s are allowed when the evidence is strong. The duration prior is soft | Missing a stoppage merges two possessions; never predicting `D` costs about 0.19 spellF1 |
| **S6** | **No team during `D`**, whatever the tracking, tags or formation say about who "owns" it. That information is used only to choose the team *after* the restart | CHALLENGE.md: dead-ball ownership is not asked |
| **S7** | **Same-team restarts are new possessions** (`A D A`). The decoder never merges across `D` | That is how the metric counts them, unlike Opta |
| **S8** | A team change needs an opponent's **control onset**: control for at least about 4 frames, or a touch followed by a teammate's control. No minimum possession length beyond that | The public window has 19-frame possessions; a long minimum would delete them |
| **S9** | An out-of-play ball needs **persistence**: about 0.5 m past the line for at least 10 observed frames, and longer on the far touchline | The tracking is 2-D: an airborne ball projects past the line from the camera; a ball on the line is still in play |
| **S10** | **Fixed frames:** 11185 (clock 00:00) is alive, the first-half kick-off; 110808–113741 (no clock) is `D`; 113742 (clock 45:00) is alive. Ungraded frames, half-time included → `X`; graded frames are never `X` | Published boundaries (FAQ.md), not labels learned from the ground truth |
| **S11** | The decoder may carry a hidden **contest** state (FIFA EFI "In Contest"), split into *contest after H* and *contest after A*. Each is **emitted as the last team** | Lets a loose ball last without flipping the label, and puts the turnover at the end of the contest, where control is gained |
| **S12** | **Control means co-movement, not proximity**: ball within about 1.5 m of a player **and** moving with them (relative speed under about 2 m/s). An opponent standing close to a dribbler or a shielding player is not in control | Proximity alone flips the label on every dribble and duel; that is why the raw baseline reports 131 possessions against 17 |
| **S13** | A **goalkeeper touch** is a change only if it becomes a catch or hold (ball speed collapses at the keeper and then moves with them), or if the keeper's team controls the ball afterwards | Law 12: a save or rebound is not goalkeeper control |
| **S14** | A **goal** is `D` from the crossing between the posts; the kick-off goes to the team that conceded. A goal is confirmed by a following kick-off formation or a `GOALS` tag, not by the crossing alone | The ball near goal is often lost in the tracking; a corner or goal kick that follows instead tells the decoder it was not a goal |

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
| Q5 | Does the ground truth flip ownership on a **goalkeeper parry** or a **block**, or only on control (S13)? | Any shot, save or block inside the public window (or a hand-labelled one): compare the true change frame with the keeper's touch and the next control onset |

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
| control = co-movement (S12) | `ball/events.py` control episodes, `config.py: control_*` | built |
| extrapolated line crossing for a ball lost on its way out | `ball/events.py: extrapolated_crossings` | built |
| kick-off after a goal → the team that conceded (S14) | `restarts.py: conceding_team, kickoff_restart_team` | built, unit-tested; not yet checked on the real goals (`make report`) |
| S4 guard: no restart while the set-up is still in progress | — | **to do**: restarts currently need only "leaves rest near a player" |
| restart team from who stands at the ball (throw-in, corner, goal kick, free kick, dropped ball) | `restarts.py: restart_taker`, `candidates.py` | built, unit-tested; no measurable gain on synthetic matches. The thrower standing outside the line is not used separately |
| corner vs goal kick from who stands at the ball (10.1) | `restarts.py` formation scores | partial: the formation is scored, the taker's team is not |
| offside / defensive free kick side | `restart_taker` (the team seen at the ball) | covered by the same rule; no separate offside logic |
| goalkeeper catch vs parry (S13) | `candidates.py: _keeper_touch_only`, `config.py: gk_*` | built, unit-tested; synthetic matches have no saves, so untested on a match |
| hidden contest state in the decoder (S11) | — | **not built**: the boundary penalty and S12 control already keep a loose ball on the last team; a 5-state decoder was judged not worth the risk without hand-labelled windows |
| audio whistle evidence (10.3) | `audio.py`, `candidates.py`, `run.py --audio` | built; thresholds and the 4-frame delay are untested on the real track. A whistle counts only next to a stoppage run |
| offsets Q1–Q3 | `config.py: stop_offset, restart_offset, turnover_offset` | placeholders = 0, set from measurements only |

---

## 10. Observable signals

How each rule above shows up in the data. The tracking is 2-D (no ball height), has no player
identities, and loses the ball on 41% of graded frames; every signal below is evidence for the
decoder, not a label on its own.

### 10.1 Signatures in the tracking

| situation | signature |
|---|---|
| **control** (dribble, carry, reception) | ball within about 1.5 m of a player, relative speed under about 2 m/s, for at least 4 frames (S12) |
| **touch** | sharp change of ball speed or direction next to a player |
| **pass** | speed jump at the passer (over about 8 m/s), distance from the passer grows, speed drops at the receiver, who then moves with it |
| **ball in flight, not seen** | a gap that starts and ends with a fast ball: keep the passer's team |
| **goalkeeper catch** | ball speed collapses within 1–3 frames at the deepest defender in their own box, then moves with them; later a speed spike on release |
| **parry / save** | sharp change of direction at the goalkeeper, the ball stays fast and moves away; no co-movement |
| **goal** | crossing between the posts (7.32 m centred on the goal line), then players walking, then a kick-off formation (ball on the centre spot, each team in its own half). `GOALS` tag |
| **throw-in** | ball at rest on a touchline, a player **outside** the touchline next to it: that player's team restarts |
| **corner** | ball at rest in a corner area, a taker next to it: the taker's team (attacking) restarts |
| **goal kick** | ball at rest in a goal area, the goalkeeper (deepest player) next to it: defending team |
| **free kick** | ball at rest, a wall of 3 or more players at about 9.15 m; the team that is *not* in the wall restarts |
| **penalty** | ball at rest on a penalty mark, all other players outside the area |
| **offside / defensive free kick** | restart deep in the half of the team that was defending |
| **stoppage without ball information** | collective deceleration: both teams' mean speed falls below about 1.2 m/s and stays there |
| **substitution** | track count changes near the halfway line during a stoppage |
| **injury** | players cluster round one stationary player; play stops |

**Identifying the goalkeeper without identities:** the player of a team who is deepest, and
nearest that team's own goal, over a window.

### 10.2 Using previous and next frames

The submission is produced offline, so a decision at frame *t* may use any frame of the same half.
The decoder already does (semi-Markov Viterbi over each half). Where a fixed look-back or
look-ahead window helps:

| decision | window | label written to |
|---|---|---|
| is an opponent's touch a change? | [t, t + k]: does that team gain control (S8, S12)? | the controlling touch t |
| is a goalkeeper touch a catch? | [t, t + 1–3 s]: does the ball move with the keeper (S13)? | the catch frame |
| when did `D` start? | [t − n, t] before players are seen walking: earliest line crossing or whistle | that crossing or whistle |
| is a moving ball a restart? | ball leaves rest at t **and** players accelerate in [t, t + n] (S4) | t |
| which team restarts? | [t − n, t]: who stands outside the line, at the corner, in the wall | the restart frame |
| was it a goal? | [t, t + 60 s]: kick-off formation, corner, goal kick, or play continuing (S14) | the crossing frame |

Choose *n* and *k* from football reasoning or from hand-labelled windows, never by tuning on the
public 5 minutes.

### 10.3 Audio: the referee's whistle

The whistle is the referee's own signal that play stops (and, for some restarts, that it may
resume). If the tactical video carries audio, it is direct evidence for `D` boundaries.

- **First check that there is audio at all**:
  `ffprobe -v error -show_streams -select_streams a data/tactical/BAR-ATH_tactical.mp4`.
  Tactical cameras often record none, or only ambient sound.
- **A whistle is a boundary, not a direction.** Referees whistle to **stop** play, and also to
  **start** some restarts: a kick-off, a penalty, a free kick after a wall, card or substitution
  (a "ceremonial" restart). The decoder decides which from context: deceleration before it means a
  stop, the ball leaving rest after it means a restart.
- **Crowd whistling is not the referee.** Spanish crowds whistle to jeer. A referee's whistle is a
  short narrow-band tone (roughly 2–4 kHz, 0.2–1.5 s); crowd whistling is broadband and sustained.
- **Sound arrives late.** Sound travels about 343 m/s, so a referee 50–70 m from the microphone
  is heard 4–5 frames after the whistle. There may also be an audio/video offset in the file.
  Measure both before using whistle times.
- **Method**: band-pass filter, short-time energy and spectral flatness, a duration check. Self-check
  the detections against the `SET PIECES` tag times and the collective deceleration. No labels needed.
- **Process the audio locally.** The match media must not go to a hosted model (FAQ.md, terms of use).

---

## 11. Standards this rulebook follows

| standard | used for |
|---|---|
| IFAB Laws of the Game 2026/27 (in force from 1 July 2026): Law 3 (substitutions, 10-second exit), 5 (advantage, injured players), 8 (kick-off, dropped ball), 9 (ball in and out of play), 10 (goal), 11 (offside), 12 (goalkeeper control, back-pass), 13–17 (restarts, throw-in and goal-kick countdowns, 8-second corner) | sections 3, 4, 5.2, 6 |
| IFAB Laws of the Game 2024/25, the edition refereed in this match | "Law edition" table only |
| IFAB VAR protocol | section 3.1 |
| Opta / Stats Perform sequences and possessions; Opta event definitions (tackle, interception, recovery, dispossessed) | section 5 |
| FIFA Enhanced Football Intelligence: possession states (team A, team B, In Contest, Out of Play) and ball recovery time | sections 1, 5, S11 |
| Organiser: CHALLENGE.md, FAQ.md, DATA.md, the public ground truth | sections 1, 2; overrides all the above |
