"""Block-chained MBAR with fixed anchor states (post-production ALS insertions).

Run from the repository root::

    pytest implicit_solvent_ddm/tests/test_block_anchors.py -v
"""
import numpy as np
import pandas as pd
import pytest

import implicit_solvent_ddm.pandasmbar as pdmbar
from implicit_solvent_ddm import block_mbar as bm
from implicit_solvent_ddm.matrix_order import CycleSteps


def _order(con=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0), charges=(0.0, 0.5, 1.0)):
    cs = CycleSteps(conformation_forces=list(con), orientational_forces=[c + 4.0 for c in con],
                    charges_windows=list(charges), external_dielectic=[], lj_windows=[0.2],
                    screening_windows=[0.5], in_solvent=True)
    cs.round(3)
    return cs.complex_order


def _blocks_as_states(order, blocks):
    return [[order[i] for i in block] for block in blocks]


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_anchored_partition_equals_position_partition(k):
    order = _order()
    bounds = bm.band_boundaries(order)
    anchors = bm.anchor_states(order, k, bounds)
    assert bm.leg_blocks(order, k, bounds, anchors) == bm.partition_blocks(len(order), k, bounds)


def test_insertion_grows_only_its_own_block():
    k = 4
    old = _order()
    anchors = bm.anchor_states(old, k, bm.band_boundaries(old))
    new = _order(con=(-8.0, -7.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0))
    old_blocks = _blocks_as_states(old, bm.leg_blocks(old, k, bm.band_boundaries(old), anchors))
    new_blocks = _blocks_as_states(new, bm.leg_blocks(new, k, bm.band_boundaries(new), anchors))
    inserted = ("lambda_window", "78.5", "1.0", "-7.0_-3.0")
    grown = [b for b in new_blocks if inserted in b]
    assert len(grown) == 1
    grown_before = [s for s in grown[0] if s != inserted]
    assert grown_before in old_blocks  # the new state joined an existing block...
    assert [b for b in new_blocks if inserted not in b] == [b for b in old_blocks if b != grown_before]  # ...and no other block moved
    old_pairs = bm.required_pairs(old, k, bm.band_boundaries(old), anchors)
    new_pairs = bm.required_pairs(new, k, bm.band_boundaries(new), anchors)
    assert old_pairs <= new_pairs  # every scored cell is still used
    m = len(grown[0]) - 1
    assert len(new_pairs - old_pairs) == 2 * m + 1


def test_position_partition_would_reshuffle_later_blocks():
    """Why anchors exist: without them one insertion shifts every later block."""
    k = 4
    # complex order runs strongest-first, so 1.0 sits near the start of the restraint run
    old, new = _order(), _order(con=(-8.0, -6.0, -4.0, -2.0, 0.0, 1.0, 2.0, 4.0))
    by_position = bm.required_pairs(new, k, bm.band_boundaries(new))
    anchored = bm.required_pairs(new, k, bm.band_boundaries(new), bm.anchor_states(old, k, bm.band_boundaries(old)))
    old_pairs = bm.required_pairs(old, k, bm.band_boundaries(old))
    assert len(by_position - old_pairs) > len(anchored - old_pairs)


def test_insertion_in_a_label_change_block():
    k = 4
    old = _order()
    anchors = bm.anchor_states(old, k, bm.band_boundaries(old))
    new = _order(con=(-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 3.0, 4.0))  # between q=1 (4.0) and 2.0
    blocks = _blocks_as_states(new, bm.leg_blocks(new, k, bm.band_boundaries(new), anchors))
    inserted = ("lambda_window", "78.5", "1.0", "3.0_7.0")
    grown = [b for b in blocks if inserted in b][0]
    assert grown[0] == ("electrostatics", "78.5", "1.0", "4.0_8.0") and len(grown) == 3


def test_oversized_span_splits_without_new_cells():
    order = _order()
    anchors = [order[0], order[-1]]  # one span over the whole leg
    whole = {(order[i], order[j]) for b in bm.leg_blocks(order, 2, anchors=anchors, max_states=len(order))
             for i in b for j in b}
    split_blocks = bm.leg_blocks(order, 2, anchors=anchors, max_states=4)
    assert all(len(b) <= 4 for b in split_blocks)
    assert all(a[-1] == b[0] for a, b in zip(split_blocks, split_blocks[1:]))  # still a chain
    split = {(order[i], order[j]) for b in split_blocks for i in b for j in b}
    assert split <= whole


def _harmonic(order, means, n=1500, seed=1):
    rng = np.random.default_rng(seed)
    xs, rows = [], []
    for state, mu in zip(order, means):
        xs.append(rng.normal(mu, 1.0, n))
        rows += [state] * n
    x = np.concatenate(xs)
    u = np.column_stack([0.5 * (x - mu) ** 2 for mu in means])
    names = ["traj_state", "traj_extdiel", "traj_charge", "traj_restraints"]
    return pd.DataFrame(u, index=pd.MultiIndex.from_tuples(rows, names=names),
                        columns=pd.Index(order, tupleize_cols=False))


def test_anchored_chain_matches_full_mbar():
    old = _order()
    anchors = bm.anchor_states(old, 4, bm.band_boundaries(old))
    new = _order(con=(-8.0, -7.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0))
    means = np.linspace(0.0, 0.6 * (len(new) - 1), len(new))  # all states shifted by 0.6 sigma
    df = _harmonic(new, means)
    chained, _, _ = bm.chained_mbar_result(df, new, 4, bm.band_boundaries(new), anchors=anchors)
    full, _, _ = pdmbar.mbar(df)
    assert chained.iloc[0, -1] == pytest.approx(full.iloc[0, -1], abs=0.05)
