import numpy as np

from possession.mbr import drop_spells, spell_f1, spells_of
from possession.policy import metrics

pm = metrics()


def random_blocks(rng, n=4000):
    """Two graded blocks of random segments with distinct neighbouring labels."""
    blocks = []
    for a, b in ((100, n // 2), (n // 2 + 300, n - 100)):
        segs, t, last = [], a, -1
        while t < b:
            k = int(rng.choice([x for x in (0, 1, 2) if x != last]))
            e = min(b, t + int(rng.randint(5, 400)))
            segs.append((t, e, k))
            t, last = e, k
        blocks.append(segs)
    return blocks


def render(blocks, n=4000):
    codes = np.full(n, pm.X, dtype=np.int8)
    for segs in blocks:
        for a, b, k in segs:
            codes[a:b] = k
    return codes


def test_fast_spell_f1_equals_the_official_scorer():
    rng = np.random.RandomState(0)
    for _ in range(60):
        gt, pred = random_blocks(rng), random_blocks(rng)
        g, p = render(gt), render(pred)
        official = pm.spell_scores(g, p, pm.valid_runs(g))["mean_f1"]
        assert abs(spell_f1(spells_of(gt), spells_of(pred)) - official) < 1e-12


def test_dropping_a_spell_merges_it_into_its_predecessor():
    segs = [(0, 100, 0), (100, 130, 1), (130, 300, 0)]
    assert drop_spells(segs, [1]) == [(0, 300, 0)]
    assert drop_spells([(0, 50, 1), (50, 200, 0)], [0]) == [(0, 200, 0)]
