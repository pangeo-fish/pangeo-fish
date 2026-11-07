from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
import xdggs  # noqa: F401

from pangeo_fish.hmm.decode import build_transitions, viterbi_healpix


def test_multisigma_transitions_and_exact_viterbi():
    level = 2
    cells = np.arange(12 * 4**level, dtype=np.uint64)
    sigmas = [0.02, 0.15]
    neighbors, weights = build_transitions(cells, level, sigmas)
    assert neighbors[0].shape[1] != neighbors[1].shape[1]
    for s, sigma in enumerate(sigmas):
        single_neighbors, single_weights = build_transitions(cells, level, [sigma])
        np.testing.assert_array_equal(neighbors[s], single_neighbors[0])
        np.testing.assert_array_equal(weights[s], single_weights[0])

    # Enumerate all feasible paths independently of the decoder's dynamic
    # programming, using the exact same transitions as the single-sigma model.
    rng = np.random.default_rng(3)
    pdf = rng.uniform(0.1, 1, (3, len(cells)))
    initial = np.zeros(len(cells))
    initial[5] = 1
    mask = np.ones(len(cells), dtype=bool)
    mask[6] = False
    pdf[:, ~mask] = np.nan
    indices = np.array([0, 0, 1], dtype=np.int32)
    dense = []
    for nbr, w in zip(neighbors, weights):
        matrix = np.full((len(cells), len(cells)), -np.inf)
        for destination in range(len(cells)):
            valid = np.isfinite(w[destination])
            matrix[destination, nbr[destination, valid]] = w[destination, valid]
        dense.append(matrix)
    candidates = []
    for second, last in product(np.flatnonzero(mask), repeat=2):
        score = (dense[0][second, 5] + np.log(pdf[1, second])
                 + dense[1][last, second] + np.log(pdf[2, last]))
        candidates.append((score, (5, second, last)))
    best_score, best_path = max(candidates)
    assert np.isfinite(best_score)
    emission = xr.Dataset(
        {"pdf": (("time", "cells"), pdf), "initial": ("cells", initial),
         "mask": ("cells", mask), "predictor_index": ("time", indices)},
        coords={"cell_ids": ("cells", cells),
                "time": pd.date_range("2025-01-01", periods=3)},
    ).dggs.decode({"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    actual = viterbi_healpix(emission, sigmas)
    np.testing.assert_array_equal(actual.cell_ids, cells[list(best_path)])
    xr.testing.assert_identical(actual.time, emission.time)
