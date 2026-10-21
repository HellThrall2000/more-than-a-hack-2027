"""A toy football match with known possession truth -- the development fixture (PLAN.md 9).

The match data is gated and must not be pasted into hosted assistants, so everything that can be
developed without it is developed against this generator: 22 players around a 4-4-2, a ball that is
dribbled, passed, intercepted and put out of play, restarts by the opponent of the last toucher.
It is a plumbing and regression fixture. It says nothing about how well the system does on the
real match, and nothing here was tuned to look like the real data.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import Config, DEFAULT

H, A, D = 0, 1, 2


@dataclasses.dataclass
class SyntheticMatch:
    players_xy: np.ndarray      # (N, 22, 2) clean, metres
    players_speed: np.ndarray   # (N, 22)
    ball_xy: np.ndarray         # (N, 2) clean
    slot_team: np.ndarray       # (22,)
    truth: np.ndarray           # (N,) int8: 0 home, 1 away, 2 dead
    cfg: Config


def make_match(n_frames=7500, seed=0, cfg: Config = DEFAULT, p_out=0.08, p_turn=0.12):
    rng = np.random.RandomState(seed)
    fps, length, width = cfg.fps, cfg.pitch_length, cfg.pitch_width
    slot_team = np.repeat([0, 1], 11).astype(np.int8)
    attack = np.where(slot_team == 0, 1.0, -1.0)

    fx = np.array([4, 18, 18, 18, 18, 40, 40, 40, 40, 62, 62], dtype=float)
    fy = np.array([34, 12, 26, 42, 56, 12, 26, 42, 56, 26, 42], dtype=float)
    base = np.zeros((22, 2))
    base[:11] = np.c_[fx, fy]
    base[11:] = np.c_[length - fx, width - fy]
    pos = base + rng.randn(22, 2) * 0.5
    wander = np.zeros((22, 2))

    team = int(rng.randint(2))
    holder = 9 if team == 0 else 20
    ball = pos[holder] + np.array([attack[holder] * 0.6, 0.0])
    state, hold_left = "hold", int(fps * 1.5)
    receiver, out_target, dead_left, restart_team, restarter = -1, None, 0, 0, -1

    players = np.zeros((n_frames, 22, 2), dtype=np.float32)
    ball_out = np.zeros((n_frames, 2), dtype=np.float32)
    truth = np.zeros(n_frames, dtype=np.int8)

    for t in range(n_frames):
        wander += rng.randn(22, 2) * 0.08
        wander = np.clip(wander, -3.0, 3.0)
        target = base + 0.35 * (ball - base) + wander
        vmax = np.full(22, 4.5)
        if state == "hold":
            target[holder] = pos[holder] + np.array([attack[holder] * 3.0, 0.0])
            vmax[holder] = 3.0
        elif state == "flight":
            target[receiver] = ball
        else:
            vmax[:] = 1.2
            target[restarter] = ball
            vmax[restarter] = 8.0
        delta = target - pos
        dist = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-6)
        pos = pos + delta / dist[:, None] * np.minimum(dist, vmax / fps)[:, None]
        pos[:, 0] = np.clip(pos[:, 0], -0.5, length + 0.5)
        pos[:, 1] = np.clip(pos[:, 1], -0.5, width + 0.5)

        if state == "hold":
            ball = pos[holder] + np.array([attack[holder] * 0.6, 0.0])
            hold_left -= 1
            if hold_left <= 0:
                r = rng.rand()
                if r < p_out:
                    side = -1.5 if ball[1] < width / 2 else width + 1.5
                    out_target = np.array([ball[0] + attack[holder] * 5.0, side])
                    state = "out"
                elif r < p_out + p_turn:
                    opp = np.flatnonzero(slot_team != team)
                    receiver = opp[np.argmin(np.hypot(*(pos[opp] - ball).T))]
                    state = "flight"
                else:
                    mates = np.flatnonzero((slot_team == team) & (np.arange(22) != holder))
                    d = np.hypot(*(pos[mates] - ball).T)
                    ok = mates[(d > 8) & (d < 30)]
                    pool = ok if len(ok) else mates
                    receiver = pool[rng.randint(len(pool))]
                    state = "flight"
        elif state in ("flight", "out"):
            goal = pos[receiver] if state == "flight" else out_target
            step = goal - ball
            dd = float(np.hypot(*step))
            if state == "flight" and dd < 0.7:
                if slot_team[receiver] != team:
                    team = int(slot_team[receiver])
                holder = int(receiver)
                state = "hold"
                hold_left = int(np.clip(rng.lognormal(np.log(fps * 1.6), 0.5), fps * 0.6, fps * 5))
                ball = pos[holder] + np.array([attack[holder] * 0.6, 0.0])
            else:
                ball = ball + step / max(dd, 1e-6) * min(dd, 14.0 / fps)
                gone = (ball[0] < -1.0 or ball[0] > length + 1.0
                        or ball[1] < -1.0 or ball[1] > width + 1.0)
                if state == "out" and gone:
                    state = "dead"
                    dead_left = int(rng.randint(fps * 4, fps * 12))
                    restart_team = 1 - team
                    cands = np.flatnonzero(slot_team == restart_team)
                    restarter = cands[np.argmin(np.hypot(*(pos[cands] - ball).T))]
        elif state == "dead":
            dead_left -= 1
            if dead_left <= 0:
                team, holder, state = restart_team, int(restarter), "hold"
                hold_left = int(fps * 1.2)
                ball = pos[holder] + np.array([attack[holder] * 0.6, 0.0])

        players[t] = pos
        ball_out[t] = ball
        crossed = (ball[0] < 0 or ball[0] > length or ball[1] < 0 or ball[1] > width)
        truth[t] = 2 if (state == "dead" or (state == "out" and crossed)) else team

    speed = np.zeros((n_frames, 22), dtype=np.float32)
    speed[1:] = np.hypot(*(players[1:] - players[:-1]).transpose(2, 0, 1)) * fps
    return SyntheticMatch(players, speed, ball_out, slot_team, truth, cfg)
