# PLAN — Frame-level ball possession, More than a Hack 2027 · Phase 1

FC Barcelona (`H`) vs Athletic Club (`A`) · design written 2026-10-21 · deadline 2026-11-17

Team choices: possession-first, with ball-state reconstruction as the backbone · CPU only ·
validation on open datasets plus a few hand-labelled windows of this match.

## Status — updated 2026-10-24

The first real submission exists and validates, `submissions/TotalFootball.csv`: tracking-only,
deterministic, no GPU. On the public 5 minutes it scores **spellF1 0.311** (bootstrap 5–95%:
0.16–0.48) and **macroF1 0.598**, with 19 possessions predicted against 17 true. The baselines on
the same window are 0.424 (`smooth125`) and 0.326 (`smooth50`). That is a correctness check, not
a ranking (Section 7).

What the public-window report shows:

| symptom | number | reading |
|---|---|---|
| dead ball missed | `D` recall 0.23, precision 0.66 | stoppages are mostly absorbed by a team, so possessions merge |
| boundaries not found | 3 of 20 true boundaries matched within ±1 s | the main timing problem. The "−13 frames" mean latency rests on those 3 alone and is not evidence of a constant offset |
| team bias toward Athletic | home share 0.47 predicted vs 0.57 true; `H` recall 0.59, `A` recall 0.84 | source not yet located: per-frame evidence, Smart Tagging, or the decoder |

### Done and not done

| area | item | status |
|---|---|---|
| environment | Python 3.13 venv; ROCm PyTorch on the RX 9060 XT (MIOpen off, `benchmarks/README.md`) | done |
| rules | [SITUATION.md](SITUATION.md): dead ball vs in play, restart sides, simplifications S1–S10, open questions Q1–Q4 | done |
| L0 | ATD parser, team mapping, cache, audit, alignment checks | done; ran on the bundle |
| L1–L3 | player cleaning, ball layer (clean, smooth, gaps, touches, control, out of play, at rest), features, restart formations, candidates | done (version 1); every threshold an untuned first guess |
| L4 | rules emission + Smart Tagging prior calibrated on silver frames | done (version 1) |
| L5–L6 | semi-Markov decoder, posterior samples, MBR on spellF1, output policy | done |
| tooling | harness + registry, `diagnose`, `semantics` (Q1–Q4), `clock`, `full_run` with console progress, `logs/` and `benchmarks/results/` | done |
| submission | `full_run --name TotalFootball` → validated CSV + SHA-256; video detector opt-in | done |
| video detector | crops, training on the GPU, gate, ROI search | built and trained once; gate not run; **off** until the weights can be committed |
| measurement | semantics study (Q1–Q4) and `diagnose` on the real match | **not run yet** |
| V3 | labelling tool `possession/labels.py` (change points → ground-truth CSV) | done |
| V3 | the labelled windows themselves; GitHub issue to the organisers about validation-only labels and committing weights | **not started** |
| dead ball | ball-free stoppage evidence (`calm.py`: two-state mixture on collective speed, no labels); extrapolated line crossing | done, 2026-10-24. Synthetic with the ball hidden in stoppages: `D` recall 0.39 → 0.98. **Not yet measured on the real match** |
| restarts | restart side by stoppage kind (ball out → opponent of the last toucher; whistle → team in control); kick-off team from tags + Law 8 | done, 2026-10-24 |
| restarts | S4 set-up guard; goal → the team that conceded kicks off | **not started** |
| team bias | Smart Tagging codes pooled by max, not summed; `diagnose` splits each source's team calls by the true team | done; **source not located yet** (needs a `diagnose` run) |
| tooling | `--set key=value` config overrides on `run`, `diagnose`, `semantics`, `full_run` | done |
| v2 on the real match | `full_run --name TotalFootball_v2` (2026-10-24): public spellF1 0.424, macroF1 0.739, 17 possessions; `D` recall 0.58; bias gone (wrong-team calls 20% / 13%). With tags off: 0.488 / 0.810. Full match: 267 possessions, 46.6% `D` | measured |
| measured conventions | Q3: true turnovers land about 13 frames before our first touch of the new team and about 24 before its control onset (8–9 of 12 found). Smart Tagging is about 70% right in both directions. Kick-off tags carry no team. Video detector: gate failed (coverage 0.02%) | measured |
| round 3 | turnovers anchored on the new team's first touch (SITUATION.md §5); tag weights calibrated per interval against control episodes (`tag_calib`); kick-off team from who stands on the centre spot + Law 8; calm onset / end in `semantics` | done, 2026-10-24; **not yet run on the real match** |
| V2 | DFL / Metrica loaders, simulator acceptance, LightGBM emission (version 2) | **not started** |
| freeze | `make repro-check` on two machines, TECHNICAL.md | **not started** |

### Next plan of action

Ordered. Nothing here is chosen by its public-window score.

1. **Make the fixes that follow from the rules, not from tuning.** Items 1–4 and 6 are done
   (2026-10-24, 72 tests pass); item 5's offsets exist and stay 0 until step 4:
   1. **Dead ball without a ball:** an unsupervised "play has stopped" signal from the players
      alone (a two-state mixture on collective speed, fitted to this match without labels). It
      counts as `D` evidence only when nobody controls the ball.
   2. **Extrapolated line crossing:** a ball last seen near a line, moving out, then lost, is out
      of play from the projected crossing (SITUATION.md S2, S9).
   3. **Restart sides by stoppage kind:** ball out → opponent of the last toucher; whistle inside
      the field → the team in control before it; kick-off team alternates by half (Law 8).
   4. **Smart Tagging:** overlapping phase codes describe one situation, so a team's tag evidence
      is the strongest of its tags, not their sum.
   5. **Per-type timing offsets** (turnover, stoppage, restart) as config values, set only from
      measurements (step 2).
   6. **Experiment switches:** `--set key=value` on every command, so an ablation (tags off,
      calm off) is one flag and is recorded in the config hash.
2. **Measure on the real match** (a person runs these locally; aggregates only): `semantics`,
   then `diagnose` with and without tags (`--set emit_tag_weight=0`). This answers Q1–Q4 and
   locates the team bias.
3. **Build V3:** a labelling tool that turns a list of change points into a ground-truth CSV.
   1. Each labeller first labels the public window blind and is scored against it.
   2. Then label 3–4 windows of 5 minutes outside it, into `labels_private/` (gitignored).
   3. File the GitHub issue (validation-only labels, committing weights) before relying on them.
4. **Decide on V3:** keep a change only if it beats the current version on V3 by more than the
   bootstrap band and does not break V1. Set the timing offsets from V3.
5. **Version 2 (M3):** DFL loader, simulator acceptance, LightGBM emission, validated on V2 and
   V3.
6. **Freeze by Nov 14:** repro-check, TECHNICAL.md, validate the exact upload file.

## 0. Summary

- **What wins.** spellF1 scores whole possessions, and smoothing-window baselines lose because they
  delay and fragment boundaries. We win by making boundaries **event-anchored**. A possession change
  snaps to a detected touch, ball-out or restart. A segment decoder with a football grammar and
  duration priors suppresses spurious flips.
- **Backbone.** A **ball-state layer** cleans the ball track and explains the 41% of graded frames
  with no ball. It emits touch, control, out-of-play and restart events with confidences. It also
  has its own ground-truth-free accuracy scorecard.
- **Evidence.** We estimate per-frame likelihoods of team-in-possession and dead ball from three
  sources:
  - ball–player geometry;
  - the collective shape of each team, which works without the ball;
  - Smart Tagging, as a soft prior calibrated on this match.

  Version 1 is rules. Version 2 is LightGBM trained on open tracking datasets, passed through an
  **ATD simulator** that reproduces this match's noise.
- **Decoding.** A **semi-Markov dynamic program** runs over candidate boundaries. A
  minimum-Bayes-risk pass then decides uncertain short spells to maximise expected spellF1.
- **Validation discipline.** The public 5 minutes check correctness only. Versions are chosen on
  external matches and on hand-labelled windows of this match.

## 1. What the score rewards — and therefore what we build

From `evaluation/possession_metrics.py`:
- A possession is a maximal run of `H` or `A` inside a half. `D` runs are not possessions, but they
  **split** them: `H D H` is two possessions.
- Predictions are matched one-to-one against same-team true possessions, greedily by IoU. F1 is
  averaged over IoU thresholds 0.3–0.7.
- `X` on a graded frame is forward-filled within the half before possessions are built. For spellF1,
  `X` means "repeat my previous label". For macroF1 it is a miss.
- Graded frames are 11185–90651 and 110808–190197. Frames 110808–113741 are graded and are all `D`;
  the second-half kick-off is at 113742 (FAQ).

**Error budget.** These figures come from perturbing the public ground truth with an exact
re-implementation of the scorer. The window has only 17 possessions, so treat them as indicative.

| perturbation of the true labels | spellF1 |
|---|---|
| every boundary jittered ±5 / ±12 / ±25 frames | 0.996 / 0.963 / 0.899 |
| every boundary 12 / 25 / 50 frames late | 0.906 / 0.777 / 0.588 |
| `D` never predicted (stoppages absorbed by a team) | 0.812 |
| possessions shorter than 2 s / 3 s / 4 s missed | 0.827 / 0.743 / 0.560 |
| one spurious 1 s opponent flip inside every possession > 10 s | **0.550** |

For comparison, EVALUATION.md reports full-match costs of 0.982 / 0.893 / 0.654 for 3 / 8 / 25
frames of lag.

IoU geometry: a possession of length L matches at IoU 0.7 only if it is shifted by at most
0.18·L. A 1 s possession therefore needs its ends within about 4 frames; a 10 s possession tolerates
about 1.8 s.

Consequences:
1. **Spurious flips and false `D` inside long possessions are the most expensive errors.** Every
   label change needs positive evidence. A missing ball is never evidence.
2. **Short possessions are common.** Six of the 17 public possessions are under 4 s, and the
   shortest is 19 frames. Missing them is the second most expensive error, so duration priors must
   allow them when the evidence is strong.
3. **Anchor boundaries to physical events,** never to the centre of a smoothing window. ±0.5 s is
   almost free; ±1 s is not.
4. **`D` matters twice:** its boundaries split possessions (spellF1), and it is one third of macroF1
   (the tie-break).

## 2. Design principles

1. **Segments, not frames.** Output is a list of (start, stop, label) per half, rendered to frames
   only at the end.
2. **Grammar.**
   - `H`↔`A` changes only at an opponent's controlled touch.
   - `H`/`A`→`D` only at a stoppage.
   - `D`→`H`/`A` only at a restart.
   - A loose or contested ball stays with the last team.
3. **Explicit uncertainty.** The ball layer outputs a posterior and a source flag per frame, not a
   blind interpolation.
4. **Decoupled training sources.**
   - Tracking evidence is learned from open datasets.
   - Smart Tagging reliability is calibrated on this match, without labels.
   - Decoder hyper-parameters are chosen on external and hand-labelled validation.
5. **The public window never ranks versions.**
6. **CPU, deterministic, one command, byte-identical output.**
7. **Measure before building** any stretch module (Section 11 gates).

## 3. Assets: how each is used

| asset | role | priority |
|---|---|---|
| tactical ATD (players + ball) | primary evidence for layers L1–L3 | core |
| Smart Tagging (638 intervals) | soft prior on team in possession, restarts, transitions and goals; boundary candidates | core |
| tactical homography (100% valid) | camera-aim prior for the ball; tells "out of view" gaps from occlusions; video overlays | core support |
| public GT (5 min) | semantics study, correctness and regression tests | checks only |
| open datasets (DFL/Sportec, Metrica samples) | training and validation through the ATD simulator | core |
| tactical video | review and labelling overlays on CPU; region-of-interest (ROI) ball recovery is a gated stretch | support |
| broadcast ATD + homography | dead-ball cue from replays; extra ball observations after sync | gated stretch |
| broadcast video | dropped: CPU budget, and its ATD carries what we need | dropped |

## 4. Architecture

```
                 tactical ATD    Smart Tagging    homography      (video: review only)
                       │               │               │
 L0 ingest + audit ────┴───────────────┴───────────────┘  → cache/*.npy (gitignored)
       │
 L1 players: dedupe · outliers · short-gap fill · goalkeepers · team shape
       │
 L2 BALL STATE: clean · smooth · touches/control · gap classes · gap posterior · out/stationary
       │                                         └──► ball scorecard (GT-free accuracy)
 L3 evidence: features F[t] + restart formations + candidate boundaries (turnover/stoppage/restart)
       │
 L4 emission p(H,A,D | F[t]) · boundary log-odds · Smart Tagging prior (calibrated on this match)
       │        ▲
       │        └── open datasets → ATD simulator → same L1–L3 code → training & validation
 L5 semi-Markov decoder: football grammar · duration priors · event-anchored boundaries
       │
 L6 MBR on spellF1 · X policy · half constraints → <team>.csv → validate_submission.py
```

**Layer outputs.** Arrays are indexed by video frame `t`, use float32 and metres, and are cached
under `cache/` keyed by a config hash.

| layer | output |
|---|---|
| L0 | `players_xy (N,46,2)`, `players_speed (N,46)`, `slot_team (46,)`, `ball_xy (N,2)`, `ball_obs (N,)`, tag intervals, `homography (N,3,3)` |
| L1 | cleaned per-team point sets, `team_shape (N,2,k)`, goalkeeper positions, attacking direction per half |
| L2 | `ball_est (N,2)`, `ball_sigma (N,)`, `ball_src (N,)` ∈ {observed, interpolated, carried, flight, unknown}, `ball_vel (N,2)`, `oob (N,)`, `stationary (N,)`, events table (frame, kind, team, score) |
| L3 | feature matrix `F (N,~120)`, candidate boundaries (frame, type, team, features) |
| L4 | `logp (N,3)` for H/A/D, log-odds per candidate boundary |
| L5 | segments per half, boundary marginals, posterior samples |
| L6 | `labels (N,)` → CSV |

## 5. Layer design

### 5.1 L0 — ingest and audit
- **Parser.** Write a vectorised streaming parser for the raw ATD. The line format is
  `<frame_id>:<slot>;…;<slot>:<ball>`; each group is `x,y,speed`, with `NaN` for a missing value
  (see `starter/baseline.py`).
  - Scale coordinates by `GAMES["BAR-ATH"]["pitch"]` (105.3 × 68.0 m).
  - Use `video_frame = frame_id − 1`. Do not use the per-half offsets in `games.py`; they belong to
    the ground truth's counter.
  - Do not load the full match through kloppy: it takes minutes and gigabytes.
- **Team mapping.** Use `starter/baseline.team_of_each_track()`: the first team in the XML is home
  (Barcelona). Smart Tagging team strings map `Barcelona`→H, `Atletic Club` (sic)→A, and `N/A`→
  unknown, matched case-insensitively.
- **Audit report.** Run it first; its numbers go into the technical document.
  1. Does the ATD XML declare ball channels beyond x/y/speed (height, ball state, owning team)? If
     so, they become first-class evidence.
  2. Ball-gap length histogram, and gap context: crowding, near a line, out of camera view.
  3. Slot lifetimes, objects per frame (mean 20.5) and duplicates.
  4. Does the tactical camera pan? If it does, how far is the camera-aim point (image centre
     projected to the pitch with `unproject`) from the visible ball?
  5. `check_alignment_raw()` and `check_time_base()` pass. `starter/baseline.py` rebuilds the four
     committed baselines **byte-identically**, which confirms the environment.

### 5.2 L1 — players and collective state
- No identities are needed: work with per-team point sets per frame.
- **Cleaning:**
  - merge same-team detections less than 0.6 m apart;
  - drop jumps that imply more than 12 m/s;
  - fill slot gaps of up to 10 frames when the slot reappears consistently.
- **Orientation.** Get the attacking direction per half from team centroids and goalkeepers at
  kick-off. The goalkeeper is the deepest player of each team, per frame.
- **Descriptors per team:** centroid, length and width, convex-hull area, mean speed, mean velocity
  toward each goal, height of the last line, and players per third. Across both teams: mean speed
  and fraction of players walking (< 2 m/s).

### 5.3 L2 — ball-state reconstruction (the backbone)
1. **Clean the observations.** Reject:
   - jumps faster than 40 m/s;
   - isolated detections that disagree with a robust local fit;
   - spare-ball swaps during stoppages (a jump of more than 10 m while the ball was stationary).
2. **Smooth within visible runs** (Savitzky–Golay or an RTS smoother). This gives velocity and
   acceleration with roughly 1 m/s noise.
3. **Detect touches and control.**
   - A touch is a velocity change above about 3 m/s, or a turn above 30° at speed, within 4 frames,
     with a player within 2 m. Record the frame, the toucher's team and the strength.
   - A control episode is the ball within about 1.5 m of a player and moving with them (relative
     speed below 2 m/s) for at least 3–5 frames.
   - Validate these thresholds on simulated external data, never on the public window.
4. **Classify each gap:**
   - length;
   - last and next observed state;
   - `out_of_view`: the projected position falls outside the image;
   - `crowded`: players within 3 m;
   - `in_flight`: the ball was fast before the gap;
   - `near_line`.
5. **Fill each gap as a posterior**, not just a point estimate.
   - Short gaps (≤ 8 frames, consistent endpoints): interpolate.
   - Flight: constant velocity with drag between the endpoints; the receiver is at the far end.
   - Carried (the ball is lost and found near the same team): attach it to the most likely carrier.
     Choose the carrier with a small dynamic program over nearby players, scoring distance to the
     bridging path plus a switch cost.
   - Long or unknown gaps: combine the camera-aim prior with motion voting (player velocities point
     at the ball) and keep σ wide. Never invent touches inside these gaps.
6. **Out of play and stoppages.**
   - `oob`: the ball estimate is more than 0.3 m past a line for at least 3 frames. Require extra
     persistence on the far side, because airborne balls and throw-ins project long.
   - `stationary`: speed below 0.5 m/s for at least 0.5 s.
   - `in_goal`.
   - Extrapolate line crossings for balls that were lost while moving out.
7. **Ball scorecard** (no ground truth needed).
   - Mask visible segments of this match. Draw their lengths from the real gap distribution and
     stratify them by context.
   - Report:
     - position error (median and p90, in metres) per gap length;
     - carrier-team accuracy;
     - touch recall inside the masked gaps.
   - Repeat against true positions on DFL data.

   This is the team's "how accurately do we track the ball" number, tracked per version.

### 5.4 L3 — evidence and candidate boundaries
**Features.** The same code runs on this match and on simulated external data.
- **Ball–player:**
  - nearest and second-nearest H and A distances, and the margin between them;
  - counts within 1, 2 and 5 m;
  - relative velocity of the nearest player;
  - time since the last touch, and that touch's team;
  - ball speed, acceleration and turn;
  - `ball_src`;
  - gap length, time since last seen and time until next seen. The run is offline, so looking ahead
    is allowed.
- **Location:**
  - ball position in each team's attacking frame;
  - distance to the lines and to restart spots;
  - box and third flags;
  - `oob` and `stationary`.
- **Collective, ball-free:** L1 descriptors and their 1–3 s trends. The team in possession stretches
  and advances; the other compacts and presses toward the ball.
- **Restart formations** (a score per frame):
  - kick-off: everyone in their own half, the ball on the centre spot;
  - corner;
  - goal kick;
  - throw-in: a player on the touchline;
  - free kick: a wall about 9 m from a stationary ball;
  - penalty.
- **Context windows:** aggregate features over ±0.2, ±1 and ±3 s.
- **Smart Tagging channels** (used in 5.5):
  - In-possession codes (BUILD UP, PROGRESSION, CREATING CHANCES, BALL IN FINAL THIRD, BALL IN THE
    BOX, PLAYERS IN THE BOX, LONG BALL, ATTACKING TRANSITION) give the tagged team.
  - Out-of-possession codes (DEFENDING IN * THIRD, DEFENSIVE TRANSITION) give the opponent.
  - SET PIECES mark a restart and its type. Throw-ins are tagged `N/A`, so infer the side.
  - GOALS mark a stoppage until the kick-off.
  - Tag edges are fuzzy: kick-off tags open about 3.5 s early. Use ramps, not hard edges.

**Candidate boundaries.** Generate for high recall, then let L4 score them.
- **Turnovers:**
  - control onsets and touches by the team not currently in possession;
  - gap-bridged changes (lost near one team, found near the other), placed inside the gap by the gap
    filler;
  - collective-reaction change points (team velocities reverse);
  - starts of transition tags.
- **Stoppages:**
  - `oob` onset, or an extrapolated line crossing;
  - collective deceleration while the ball is in play (a whistle);
  - a goal.
- **Restarts:**
  - the ball leaving rest (stationary for 1 s or more, then above 3 m/s) near a player, at a
    restart spot or in a restart formation;
  - SET PIECES windows.
- Apply non-maximum suppression within ±5 frames.

### 5.5 L4 — emission and boundary models
- **Version 1: rules, week 1.**
  - Soft control per team: `c_team = max_p exp(−d²/2σd²)·exp(−Δv²/2σv²)`.
  - A loose ball inherits the last team in control.
  - `D` comes from `oob` or a stationary ball, combined with walking players or a tag window.
- **Version 2: learned.**
  - **Per-frame LightGBM** over {H, A, D} on the L3 tracking features. Training data:
    - **DFL/Sportec open data:** 7 Bundesliga/2. Bundesliga matches at 25 Hz, with per-frame
      `BallPossession` (1/2) and `BallStatus` (0/1). CC-BY 4.0, figshare
      10.6084/m9.figshare.28196177, loadable with kloppy's `sportec` loader or floodlight.
    - **Metrica's public sample games** (github.com/metrica-sports/sample-data): the same provider,
      and game 3 uses the same FIFA/EPTS format. Derive possession from its events.

    Sample frames with a stride, because neighbours are near-duplicates. Over-sample near
    boundaries. Cross-validate leaving one match out.
  - **Boundary classifier:** for each candidate, P(a true boundary of this type within ±5 frames).
  - **ATD simulator.** This is what makes external data transferable. It:
    - strips identities into tracklets with this match's lifetime distribution;
    - adds identity swaps at close encounters, occlusion drop-outs, jitter and rare duplicates;
    - removes the ball using a missingness model fitted on this match (crowding, speed, lines,
      camera view), with bursty gaps matching the real histogram;
    - displaces airborne balls the way a ground projection would, using DFL's ball height and an
      assumed camera position.

    **Accept** the simulator when four statistics match this match: ball-visible rate (about 57%),
    gap histogram, objects per frame (about 20.5) and slot lifetimes. In addition, an adversarial
    classifier should barely separate simulated feature rows from real ones (AUC near 0.5).
  - **Smart Tagging prior, calibrated here without labels.** Estimate each code's reliability on
    "silver" frames with unambiguous control: ball observed, a player within 0.5 m, no opponent
    within 3 m. Add the result as a log-linear term to the LightGBM log-probabilities. External data
    has no Smart Tagging, so this calibration has to happen on this match.
  - **Label-definition shift.** DFL's possession definition is not Metrica's corrected-tracking
    owner. Calibrate timing offsets and decoder weights on the hand-labelled windows, never on
    external labels' edges.

### 5.6 L5 — event-anchored semi-Markov decoder
- **States** `H`, `A`, `D`, with the grammar from Section 2.
- **Restart side rules:**
  - corner → attacking team;
  - goal kick → defending team;
  - throw-in → opponent of the last toucher;
  - kick-off → the team that conceded, or the half-start rule;
  - free kick → usually the team in possession.
- **Hard constraints:**
  - the first half kicks off at 11185;
  - 110808–113741 is `D`, and the second half kicks off at 113742.

  The kick-off detector must reproduce both; this is a test.
- **Segment score** for `[s, e)` with label `k`:
  `α·Σ log p(k|t) + log p_dur(e−s | k) + β·boundary_logodds(s)`.
  - Prefix sums make each term O(1).
  - α < 1 discounts correlated frames.
- **Durations:**
  - Fit log-normal priors per label on external data. Possessions can be under 1 s; stoppages run
    from 2 s to several minutes.
  - Sanity-check against the public window (mean 14.2 s, minimum 0.76 s), but never fit to it.
- **Search:**
  - Run the dynamic program over candidate frames, plus a 1 s grid inside long ball gaps so that
    changes the camera never saw remain possible.
  - Cap segments at about 4 minutes.
  - Use vectorised numpy and deterministic tie-breaking.
- **Posterior:** a forward pass (log-sum-exp) gives boundary marginals and seeded posterior samples.

### 5.7 L6 — output policy
- **Minimum-Bayes-risk (MBR) on the ranking metric.**
  1. Draw 200 seeded posterior segmentations.
  2. Build candidate outputs: the most likely segmentation (MAP), plus variants that drop spells
     whose marginal probability is below θ.
  3. Keep the candidate with the highest mean spellF1 against the samples.

  Use a fast sweep-line spellF1, tested to equal `possession_metrics.spell_scores`.
- **Ungraded frames → `X`.** Graded frames get labels.
- **Optional `X`** on very-low-confidence frames *inside* a segment, never on its first frame. It
  cannot change spellF1 and can help macroF1. Off by default.
- **Write and check.** Write with `baseline.write_submission`, then run
  `evaluation/validate_submission.py`.

## 6. Possession rules implemented

This section doubles as the hand-labelling guide and a required section of the technical document.

| situation | label behaviour |
|---|---|
| controlled touch by the team not in possession (control ≥ k frames, or a touch followed by a team-mate's control) | change at control onset, minus the calibrated offset |
| deflection, block, uncontrolled touch, aerial duel without follow-up control | no change |
| loose or contested ball, ball in flight | stays with the last team |
| ball not visible | stays with the last team unless gap-bridged evidence says otherwise |
| ball wholly over a touchline or goal line | `D` from the crossing frame |
| whistle (foul, offside, injury, VAR, substitution) | `D` from the onset of collective deceleration, plus the calibrated offset |
| goal | `D` until the kick-off is struck |
| restart (kick-off, throw-in, corner, goal kick, free kick, penalty) | `D` until the ball is played, then the restarting team |
| goalkeeper holding the ball | alive, goalkeeper's team |
| frames 110808–113741 | `D` (published in the FAQ) |
| ungraded frames | `X` |

**Week-1 semantics study.** Measure the exact conventions on the 20 boundaries of the public
window: 12 turnovers, 4 stoppages and 4 restarts. For example: does the change land on the first
touch, or after k frames of control? Does `D` start at the line crossing or at the whistle? This is
understanding the definition, which CHALLENGE.md asks for; it is not selecting between versions.

## 7. Validation

| set | size | used for | never used for |
|---|---|---|---|
| V1 public 5 min | 17 possessions, 20 boundaries | alignment, format, semantics, regression tests | choosing versions, tuning |
| V2 open datasets through the simulator | ~10 matches, hundreds of possessions each | model selection, hyper-parameters, ablations (leave-one-match-out) | timing conventions |
| V3 hand-labelled windows of BAR-ATH | 3–4 windows of 5 min | final arbiter between versions; α, β, θ; timing offsets | training, the regenerate path, the repo |
| V4 ball scorecard (masked ball) | whole match | quality of the ball layer | — |
| V5 agreement with Smart Tagging | 638 tags | sanity checks, only for parts that do not consume tags | an optimisation target |

**Protocol:**
- Score with `possession_metrics.evaluate()` itself; never re-implement the ranking metric. Use
  `--bootstrap 1000`.
- Adopt a change only if it beats the current best on V3 by more than the bootstrap band and does
  not regress V2.
- **Hand labels:**
  1. First open a GitHub issue to confirm that validation-only hand labels are within RULES.md. It
     allows "small amounts of hand-labelled data used to train or validate".
  2. Each labeller first labels the public window blind and is scored against it. That trains them
     to the ground truth's conventions and measures label quality.
  3. Two labellers share one window to measure agreement, which is the ceiling for the model.
  4. Store labels outside the repository.
- Keep an **experiment registry** mapping config hash to V2, V3 and V4 scores. The technical
  document's ablation table comes straight from it.

## 8. Repository layout and reuse

```
possession/                       solution package (new; organiser files stay untouched)
  config.py                       every threshold & hyper-parameter, one dataclass, hashed
  ingest.py  audit.py             L0
  players.py                      L1
  ball/ clean.py smooth.py events.py gaps.py scorecard.py      L2
  features.py restarts.py tags.py                              L3
  emission_rules.py emission_gbm.py simulate_atd.py external.py   L4
  decode.py mbr.py policy.py                                   L5–L6
  harness.py                      V1–V5, experiment registry
  review.py                       local video/pitch overlays for error triage and labelling
  run.py                          python -m possession.run  (end to end)
tests/                            parser, alignment, grammar invariants, fast spellF1 == official
models/                           small LightGBM files trained on open data only
TECHNICAL.md                      technical document → exported to <team>.pdf
requirements-solution.txt         pinned versions, Python 3.11
Makefile                          audit · train · predict · validate · score · submission · repro-check
cache/  labels_private/           gitignored — frame-level derived data is never published
```

**Reuse; don't rewrite:**
- `games.py`: `asset_paths`, `require`, `half_video_blocks`, `GAMES[...]["pitch"]`.
- `starter/baseline.py`: `team_of_each_track`, `write_submission`, and the raw-format reference.
- `starter/load_atd.py`: `check_alignment_raw`, `video_frame_of`.
- `starter/load_smart_tagging.py`: `load`, `check_time_base`.
- `starter/load_homography.py`: `load`, `project`, `unproject`.
- `starter/utils.py`: `draw_pitch`, `plot_frame`, `animate_frames`.
- `evaluation/possession_metrics.py`: `evaluate`, `valid_runs`, `fill_abstentions`, `spell_scores`,
  `bootstrap_spell_f1`.
- `evaluation/run_evaluation.py --bootstrap 1000` and its `reports/<name>_transitions.csv`
  (matched, missed and spurious boundaries) drive error triage.
- Run `evaluation/validate_submission.py` before every upload.

## 9. Reproducibility, compliance and data hygiene

- **One command.** `make submission` runs ingest → train (only if `models/` is empty) → predict →
  validate.
  - CPU only; target under 20 minutes.
  - Use Python 3.11 (for example via uv): this machine has 3.14, and some wheels lag behind it.
  - Pin all versions.
- **Determinism.** Fix seeds and thread counts, set LightGBM `deterministic=True`, and use stable
  sorts and argmax. `make repro-check` builds twice, and on a second machine, then compares the
  CSV's SHA-256. The rules require *the same file*.
- **Nothing frame-level in git:** no cache, features, hand labels or submissions. Add `.gitignore`
  entries for `cache/`, `labels_private/` and `submissions/`, and check `git status` before every
  push.
- **No match data to hosted LLMs, including AI coding assistants.** The T&C forbid giving hosted
  LLMs the videos, datasets "or any extracts thereof".
  - Assistants write code against synthetic fixtures and the public repo files.
  - A person runs data-touching commands outside the assistant session and shares aggregate scores
    only.
  - Add deny rules for `data/**` and `cache/**` in the assistant's settings.
  - Check the T&C for the exact scope.
- **External data comes from other matches only.** Any other source of data about this fixture is
  forbidden (RULES.md §2). Credit DFL (CC-BY 4.0), and check the licence of Metrica's sample data.

## 10. Milestones and workstreams

| milestone | date | exit gate |
|---|---|---|
| M0 setup | Oct 23 | data in `data/`, Python 3.11 env, audit report, alignment and time-base checks pass, baselines rebuild byte-identically |
| M1 version 1 end to end | Oct 27 | rules pipeline produces a valid CSV; harness scores V1 and one external match; semantics note written; hand-label issue filed |
| M2 backbone | Nov 3 | ball layer v2 and scorecard; dead-ball and restart detectors; semi-Markov decoder; simulator accepted; ≥ 3 hand-labelled windows |
| M3 learned | Nov 10 | LightGBM emission and boundary classifier; tag prior calibrated; MBR; ablation table; stretch go/no-go decided |
| M4 freeze | Nov 14 | code freeze; repro-check on two machines; TECHNICAL.md complete |
| submit | Nov 16 | validate the exact files, then upload `<team>.csv`, `<team>.pdf` and the repo URL, a day ahead of the deadline |

**Workstreams (3–5 people):**
- **A — Ball and data:** L0–L2 and the scorecard.
- **B — Stoppages and restarts,** plus the Smart Tagging channel.
- **C — Learning:** external loaders, the ATD simulator, LightGBM.
- **D — Decoder and metric:** L5–L6, the harness, the registry, MBR.
- **E — fourth or fifth person:** labelling tool, hand labels, review clips, the technical document;
  then the stretch modules.

Always keep the best valid submission so far, re-validated at every milestone.

## 11. Stretch modules, built only once their gate is passed

| module | idea | gate (measure first) |
|---|---|---|
| broadcast sync | align broadcast ATD to tactical by matching ball and player positions in pitch coordinates, with piecewise offsets per half. Replays and close-ups (`valid=0`, 34% of broadcast frames) become dead-ball evidence, and the broadcast ball fills tactical gaps | P(broadcast ball visible \| tactical ball missing) ≥ 20%, or replay stretches overlap `D` on V1/V3 |
| ROI ball recovery from tactical video, on CPU | decode only frames inside graded gaps and search a small region around the L2 prior. Use a tiny detector trained on ATD-visible frames projected to pixels with the homography | precision ≥ 0.9 on masked visible frames at useful recall; under 1 hour of CPU; deterministic |
| learned ball trajectory | BallRadar-style Set Transformer + Bi-LSTM (Kim et al., KDD 2023), trained on simulated open data | beats the L2 gap filler on the scorecard |
| self-training | fine-tune LightGBM on high-confidence pseudo-labels from this match | V3 gain larger than the bootstrap band |

## 12. Risks and mitigations

| risk | mitigation |
|---|---|
| over-fitting the 17 public possessions | strict V1 role; selection on V2 and V3 with bootstrap intervals |
| definition mismatch (external labels vs Metrica's owner) | semantics study, hand-labelled calibration, few physically meaningful parameters |
| ball missing exactly at turnovers | ball-free collective evidence, gap-bridged candidates, metrics stratified by visibility |
| airborne ball and throw-in projection errors | persistence rules, far-side margins, projection displacement in the simulator |
| errors in Smart Tagging | soft prior with calibrated reliability; ablation with tags off |
| 6.5 GB RAM, CPU only | float32 memmaps, chunked features, LightGBM, no video in the main path |
| non-reproducible output | determinism rules, SHA-256 repro-check on two machines |
| leaking data | gitignore, pre-push check, no data to hosted LLMs |
| running out of time | staged milestones; a valid best-so-far file at all times |

## 13. Deliberately out of scope

- Rebuilding detection, tracking or re-identification; per-player identities.
- GPU models; decoding the broadcast video.
- Tuning anything on the public window; treating its possession counts as targets.
- Any data about this fixture from other sources.

## 14. Technical document outline (SUBMISSION.md)

SUBMISSION.md asks for six things; each comes from a section of this plan.

| SUBMISSION.md asks for | source in this plan |
|---|---|
| approach and why | Sections 1–2 and 4–5 |
| assets used and dropped | Section 3, plus the ablation table from the registry |
| how missing and noisy tracking is handled | L1 and L2, including the scorecard |
| possession rules | Section 6 |
| limitations | V3 error triage |
| reproduction | Section 9 |

## Appendix — reference numbers

- **Video:** 199,319 frames at 25 fps; 158,857 graded.
- **Ball visibility:** visible on 59.3% of graded frames; 46 track slots; 20.5 objects per graded
  frame on average.
- **Halves:** 11185–90651 and 110808–190197; the second-half kick-off is at 113742.
- **Public window** (11185–18684):
  - labels: `H` 3,454 · `A` 2,577 · `D` 1,469;
  - 17 possessions of 0.76–42.4 s (mean 14.2 s);
  - 4 stoppages of 5.5–20.4 s;
  - boundaries: 12 turnovers, 4 stoppages, 4 restarts.
- **Floor:** `baseline_smooth50` scores spellF1 0.2433 on the full match (0.3256 on the public
  window). The full-match noise band for spellF1 is about 0.04.
