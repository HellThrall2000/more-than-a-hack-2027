"""L5 -- event-anchored semi-Markov decoder (PLAN.md 5.6).

A segmentation of one graded block is a chain of nodes (the block ends plus candidate boundary
frames) and a label per segment. A segment [i, j) with label k scores

    alpha * sum_t log p(k | t)  +  log p_dur(length | k)          (emission, duration prior)

and a boundary at node j between labels k' -> k scores a bonus of its kind (team change, stoppage,
restart) minus a fixed penalty. Same-label neighbours are not allowed, so two segments can only
meet at a real change. Everything is O(1) per (segment, label) through prefix sums, vectorised
over the start node. The same lattice is run with max (Viterbi) and with log-sum-exp (posterior
sampling for the minimum-Bayes-risk step).
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from .candidates import Candidates
from .config import Config

H, A, D = 0, 1, 2
NEG = -1e18
Segment = Tuple[int, int, int]      # (start, stop, label), stop exclusive


@dataclasses.dataclass
class BlockProblem:
    a: int
    b: int
    start_labels: Tuple[int, ...]
    nodes: np.ndarray               # (M,) absolute frames, nodes[0] == a, nodes[-1] == b
    turn: np.ndarray
    stop: np.ndarray
    restart: np.ndarray
    restart_pref: np.ndarray        # (M, 2)


def make_problem(cands: Candidates, a, b, start_labels, cfg: Config) -> BlockProblem:
    sel = (cands.frame > a) & (cands.frame < b)
    zeros = np.zeros(1)
    nodes = np.concatenate(([a], cands.frame[sel], [b])).astype(np.int64)
    pad = lambda x: np.concatenate((zeros, x[sel], zeros))      # noqa: E731
    return BlockProblem(a, b, tuple(start_labels), nodes, pad(cands.turn), pad(cands.stop),
                        pad(cands.restart),
                        np.vstack((np.zeros((1, 2)), cands.restart_pref[sel], np.zeros((1, 2)))))


def _logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze() if axis is None \
        else (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class Lattice:
    def __init__(self, logp, prob: BlockProblem, cfg: Config):
        self.p, self.cfg = prob, cfg
        self.nodes = prob.nodes
        self.M = len(self.nodes)
        seg = logp[prob.a:prob.b].astype(np.float64) * cfg.dec_alpha
        self.P = np.vstack((np.zeros((1, 3)), np.cumsum(seg, axis=0)))     # P[x] = sum_{f<x}
        self.rel = self.nodes - prob.a
        self.init = np.full(3, NEG)
        self.init[list(prob.start_labels)] = 0.0
        self.bmat = self._transition_bonuses()
        self.lo = np.searchsorted(self.nodes, self.nodes - cfg.dec_max_segment, side="left")

    def _transition_bonuses(self):
        """(M, 3, 3): bonus[j, k_prev, k] for a boundary at node j; NEG on the diagonal."""
        pr, cfg = self.p, self.cfg
        b = np.full((self.M, 3, 3), NEG)
        pen = cfg.dec_boundary_penalty
        for k_prev in range(3):
            for k in range(3):
                if k_prev == k:
                    continue
                if k_prev < 2 and k < 2:
                    b[:, k_prev, k] = pr.turn - pen
                elif k == D:
                    b[:, k_prev, k] = pr.stop - pen
                else:
                    b[:, k_prev, k] = pr.restart + pr.restart_pref[:, k] - pen
        return b

    def dur_logpdf(self, frames, k):
        cfg = self.cfg
        median, sigma = ((cfg.dec_dur_median_dead, cfg.dec_dur_sigma_dead) if k == D
                         else (cfg.dec_dur_median_alive, cfg.dec_dur_sigma_alive))
        sec = np.maximum(frames, 1) / float(cfg.fps)
        z = (np.log(sec) - np.log(median)) / sigma
        return -np.log(sec) - np.log(sigma * np.sqrt(2 * np.pi)) - 0.5 * z * z

    def seg(self, i, j, k):
        """Score of the segment from node(s) i to node j with label k (i may be an array)."""
        return (self.P[self.rel[j], k] - self.P[self.rel[i], k]
                + self.dur_logpdf(self.nodes[j] - self.nodes[i], k))

    def forward(self, mode="max"):
        red = (lambda x: np.max(x)) if mode == "max" else (lambda x: _logsumexp(x))  # noqa: E731
        M = self.M
        f = np.full((M, 3), NEG)
        h = np.full((M, 3), NEG)
        for j in range(1, M):
            lo = max(1, int(self.lo[j]))
            idx = np.arange(lo, j)
            for k in range(3):
                opts = []
                if self.init[k] > NEG / 2 and self.nodes[j] - self.nodes[0] <= self.cfg.dec_max_segment:
                    opts.append(np.array([self.init[k] + self.seg(0, j, k)]))
                if len(idx):
                    opts.append(h[idx, k] + self.seg(idx, j, k))
                if opts:
                    f[j, k] = red(np.concatenate(opts))
            if j < M - 1:
                for k in range(3):
                    h[j, k] = red(f[j, :] + self.bmat[j, :, k])
        self.f = f
        return f

    def trace(self, sample=False, rng: Optional[np.random.RandomState] = None) -> List[Segment]:
        """Backtrack through the lattice: argmax if sample is False, else draw from the posterior.

        Run forward("max") before argmax tracing and forward("sum") before sampling; the weights
        recomputed here are the same terms the forward pass reduced over.
        """
        f, nodes = self.f, self.nodes
        j = self.M - 1
        k = self._pick(f[j], sample, rng)
        out = []
        while True:
            lo = max(1, int(self.lo[j]))
            idx = np.arange(lo, j)
            weights, where = [], []
            if self.init[k] > NEG / 2 and nodes[j] - nodes[0] <= self.cfg.dec_max_segment:
                weights.append(np.array([self.init[k] + self.seg(0, j, k)]))
                where.append((np.array([0]), np.array([-1])))
            if len(idx):
                w = f[idx, :] + self.bmat[idx, :, k] + self.seg(idx, j, k)[:, None]     # (win, 3)
                weights.append(w.ravel())
                where.append((np.repeat(idx, 3), np.tile(np.arange(3), len(idx))))
            allw = np.concatenate(weights)
            choice = self._pick(allw, sample, rng)
            starts = np.concatenate([w[0] for w in where])
            prevs = np.concatenate([w[1] for w in where])
            i, k_prev = int(starts[choice]), int(prevs[choice])
            out.append((int(nodes[i]), int(nodes[j]), int(k)))
            if k_prev < 0:
                break
            j, k = i, k_prev
        return out[::-1]

    @staticmethod
    def _pick(w, sample, rng):
        if not sample:
            return int(np.argmax(w))
        p = np.exp(w - np.max(w))
        p /= p.sum()
        return int(np.searchsorted(np.cumsum(p), rng.rand(), side="right").clip(0, len(p) - 1))


def decode_block(logp, prob: BlockProblem, cfg: Config, n_samples=0, seed=0):
    """Most likely segmentation of one block and, optionally, posterior samples of it."""
    lat = Lattice(logp, prob, cfg)
    lat.forward("max")
    best = lat.trace(sample=False)
    samples = []
    if n_samples:
        lat.forward("sum")
        rng = np.random.RandomState(seed)
        samples = [lat.trace(sample=True, rng=rng) for _ in range(n_samples)]
    return best, samples


def decode_match(logp, cands: Candidates, blocks, kickoffs, cfg: Config, n_samples=0):
    """Decode every graded block. Returns [(half, best_segments, samples)].

    A block that is dead until a known kick-off (the second half: frames 110808-113741 are all D)
    has that prefix fixed to D and only the rest decoded, with an alive start label.
    """
    results = []
    for half, a, b in blocks:
        kick = (kickoffs or {}).get(half)
        prefix = []
        start_labels = (H, A, D)
        if kick is not None:
            if kick > a:
                prefix = [(a, kick, D)]
            a_dec, start_labels = max(a, kick), (H, A)
        else:
            a_dec = a
        prob = make_problem(cands, a_dec, b, start_labels, cfg)
        best, samples = decode_block(logp, prob, cfg, n_samples, seed=cfg.seed + half)
        results.append((half, prefix + best, [prefix + s for s in samples]))
    return results
