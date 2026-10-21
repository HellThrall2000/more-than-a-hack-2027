import numpy as np

from possession.candidates import Candidates
from possession.config import DEFAULT as cfg
from possession.decode import H, A, D, decode_match

H_, A_, D_ = 0, 1, 2


def cands(frames, turn=None, stop=None, restart=None):
    m = len(frames)
    z = np.zeros(m)
    return Candidates(np.array(frames, dtype=np.int64), z if turn is None else np.array(turn, float),
                      z if stop is None else np.array(stop, float),
                      z if restart is None else np.array(restart, float), np.zeros((m, 2)))


def logp_for(n, spans):
    """Strong per-frame evidence for the given [(a, b, label)]; uniform elsewhere."""
    lp = np.full((n, 3), np.log(1 / 3.0), dtype=np.float32)
    for a, b, k in spans:
        lp[a:b] = np.log(0.05)
        lp[a:b, k] = np.log(0.9)
    return lp


def check_chain(segs, a, b):
    assert segs[0][0] == a and segs[-1][1] == b
    for (s0, e0, k0), (s1, e1, k1) in zip(segs[:-1], segs[1:]):
        assert e0 == s1 and k0 != k1
    assert all(e > s for s, e, _ in segs)


def test_clear_evidence_is_segmented_at_the_event():
    lp = logp_for(900, [(0, 500, H_), (500, 900, A_)])
    c = cands([250, 500, 750], turn=[0, 8, 0])
    [(_, best, _)] = decode_match(lp, c, [(1, 0, 900)], {1: 0}, cfg)
    assert best == [(0, 500, H_), (500, 900, A_)]


def test_a_brief_weak_blip_does_not_split_a_possession():
    lp = logp_for(1200, [(0, 1200, H_)])
    lp[580:600] = np.log([0.3, 0.4, 0.3])          # 20 frames leaning slightly to A
    c = cands(list(range(50, 1200, 50)))
    [(_, best, _)] = decode_match(lp, c, [(1, 0, 1200)], {1: 0}, cfg)
    assert best == [(0, 1200, H_)]


def test_dead_ball_between_two_possessions_of_the_same_team_stays_two_possessions():
    lp = logp_for(1500, [(0, 500, H_), (500, 800, D_), (800, 1500, H_)])
    c = cands([500, 800], stop=[10, 0], restart=[0, 10])
    [(_, best, _)] = decode_match(lp, c, [(1, 0, 1500)], {1: 0}, cfg)
    assert best == [(0, 500, H_), (500, 800, D_), (800, 1500, H_)]


def test_second_half_is_dead_until_the_kickoff_and_never_starts_dead_after_it():
    lp = logp_for(3000, [(0, 3000, H_)])
    lp[1000:1600] = np.log(0.01)                   # even if the evidence says nothing is happening
    lp[1000:1600, D_] = np.log(0.98)
    [(_, best, _)] = decode_match(lp, cands([2000]), [(2, 1000, 3000)], {2: 1600}, cfg)
    assert best[0] == (1000, 1600, D_) and best[1][0] == 1600 and best[1][2] in (H_, A_)
    check_chain(best, 1000, 3000)


def test_random_inputs_always_give_a_valid_chain_and_valid_samples():
    rng = np.random.RandomState(3)
    for _ in range(5):
        n = 2500
        lp = np.log(rng.dirichlet(np.ones(3), size=n)).astype(np.float32)
        frames = sorted(rng.choice(np.arange(10, n - 10), 80, replace=False).tolist())
        c = cands(frames, turn=rng.rand(80) * 8, stop=rng.rand(80) * 8, restart=rng.rand(80) * 8)
        [(_, best, samples)] = decode_match(lp, c, [(1, 0, n)], {1: 0}, cfg.replace(seed=1),
                                            n_samples=10)
        check_chain(best, 0, n)
        assert best[0][2] != D_
        for s in samples:
            check_chain(s, 0, n)
