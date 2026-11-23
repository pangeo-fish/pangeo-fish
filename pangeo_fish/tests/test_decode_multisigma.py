from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs  # noqa: F401

from pangeo_fish.hmm.decode import build_transitions, viterbi_healpix


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_multisigma_transitions_and_exact_viterbi(dtype):
    level = 2
    cells = np.arange(12 * 4**level, dtype=np.uint64)
    sigmas = [0.02, 0.15]
    neighbors, weights = build_transitions(cells, level, sigmas)
    assert neighbors[0].shape[1] != neighbors[1].shape[1]
    for s, sigma in enumerate(sigmas):
        single_neighbors, single_weights = build_transitions(cells, level, [sigma])
        np.testing.assert_array_equal(neighbors[s], single_neighbors[0])
        np.testing.assert_array_equal(weights[s], single_weights[0])

    rng = np.random.default_rng(3)
    pdf = rng.uniform(0.1, 1, (3, len(cells))).astype(dtype)
    initial = np.zeros(len(cells), dtype=dtype)
    initial[5] = 1
    mask = np.ones(len(cells), dtype=bool)
    mask[6] = False
    # The transition from step t to step t + 1 uses the index of step t (the
    # convention of the HMM filter), so the index of the last step is unused.
    indices = np.array([1, 0, 0], dtype=np.int32)
    # Favour a cell that the wide kernel reaches from the first cell and the
    # narrow one does not: the two possible conventions then give different paths.
    reach = [
        set(nbr[5, np.isfinite(w[5])].tolist()) for nbr, w in zip(neighbors, weights)
    ]
    far = min((reach[1] - reach[0]) & set(np.flatnonzero(mask).tolist()))
    pdf[1:, far] = 50
    pdf[:, ~mask] = np.nan

    # Enumerate all feasible paths independently of the decoder's dynamic
    # programming, using the exact same transitions as the single-sigma model.
    dense = []
    for nbr, w in zip(neighbors, weights):
        matrix = np.full((len(cells), len(cells)), -np.inf)
        for destination in range(len(cells)):
            valid = np.isfinite(w[destination])
            matrix[destination, nbr[destination, valid]] = w[destination, valid]
        dense.append(matrix)

    def best(transition_indices):
        into_second, into_last = transition_indices
        candidates = []
        for second, last in product(np.flatnonzero(mask), repeat=2):
            score = (
                dense[into_second][second, 5]
                + np.log(pdf[1, second])
                + dense[into_last][last, second]
                + np.log(pdf[2, last])
            )
            candidates.append((score, (5, second, last)))
        return max(candidates)

    best_score, best_path = best(indices[:-1])
    assert np.isfinite(best_score)
    # using the index of the step a transition ends at would give another path
    assert best(indices[1:])[1] != best_path
    emission = xr.Dataset(
        {
            "pdf": (("time", "cells"), pdf),
            "initial": ("cells", initial),
            "mask": ("cells", mask),
            "predictor_index": ("time", indices),
        },
        coords={
            "cell_ids": ("cells", cells),
            "time": pd.date_range("2025-01-01", periods=3),
        },
    ).dggs.decode({"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    actual = viterbi_healpix(emission, sigmas)
    np.testing.assert_array_equal(actual.cell_ids, cells[list(best_path)])
    xr.testing.assert_identical(actual.time, emission.time)
