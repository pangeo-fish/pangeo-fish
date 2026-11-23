"""The ``device`` argument of the helpers that build a predictor."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from healpix_geo import nested
from pint_xarray import unit_registry as ureg

from pangeo_fish import helpers

LEVEL = 8
CELL_RAD = np.sqrt(4 * np.pi / (12 * 4**LEVEL))

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
# "cpu" runs everywhere, "cuda" only where PyTorch sees a GPU
DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]
TORCH_CONV_METHODS = ["FFTConv", "LargeHealpixConv", "HealpixAnalyseLargeConv"]


@pytest.fixture(scope="module")
def emission():
    centre = nested.lonlat_to_healpix(
        np.array([10.0]), np.array([45.0]), LEVEL, ellipsoid="WGS84"
    )
    ring = np.asarray(
        nested.kth_neighbourhood(np.asarray(centre, dtype=np.uint64), LEVEL, 20)
    )[0]
    cells = np.unique(ring[ring >= 0]).astype("int64")
    lon, lat = nested.healpix_to_lonlat(
        cells.astype(np.uint64), LEVEL, ellipsoid="WGS84"
    )
    n_steps = 4

    rng = np.random.default_rng(seed=0)
    pdf = rng.random((n_steps, cells.size))
    pdf /= pdf.sum(axis=1, keepdims=True)
    initial = np.zeros(cells.size)
    initial[cells.size // 2] = 1.0
    final = np.zeros(cells.size)
    final[cells.size // 2 + 1] = 1.0

    ds = xr.Dataset(
        coords={
            "cell_ids": ("cells", cells),
            "time": ("time", pd.date_range("2022-06-13", periods=n_steps)),
            "latitude": ("cells", lat, {"standard_name": "latitude"}),
            "longitude": ("cells", lon, {"standard_name": "longitude"}),
        },
        data_vars={
            "pdf": (("time", "cells"), pdf),
            "initial": ("cells", initial),
            "final": ("cells", final),
            "mask": ("cells", np.ones(cells.size, dtype=bool)),
        },
    )
    ds["cell_ids"].attrs.update(
        {"level": LEVEL, "indexing_scheme": "nested", "ellipsoid": "WGS84"}
    )
    return ds


def _optimize(emission, conv_method, device, **kwargs):
    return helpers.optimize_pdf(
        ds=emission,
        adjustment_factor=1.0,
        truncate=None,
        maximum_speed=ureg.Quantity(3.0, "km / h"),
        tolerance=1e-3,
        conv_method=conv_method,
        device=device,
        **kwargs,
    )


class _Stop(Exception):
    pass


def _stop_at(monkeypatch, name):
    """Replace ``helpers.<name>`` by a function that records its kwargs and stops."""
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(helpers, name, fake)
    return seen


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
@pytest.mark.parametrize(
    "conv_method",
    ["HealpixConv", "HealpixAnalyseConv", *TORCH_CONV_METHODS, None],
)
def test_cpu_is_accepted_for_every_method(conv_method, device):
    helpers._check_device(device, conv_method)


def test_unknown_device_is_refused():
    with pytest.raises(ValueError, match="Unknown device 'gpu'"):
        helpers._check_device("gpu")


@pytest.mark.parametrize("conv_method", ["HealpixConv", "HealpixAnalyseConv"])
def test_cpu_only_methods_refuse_another_device(emission, conv_method):
    # refused before the emission is loaded or any CUDA check
    with pytest.raises(ValueError, match="runs on the CPU only"):
        _optimize(emission, conv_method, "cuda")
    with pytest.raises(ValueError, match="runs on the CPU only"):
        helpers._get_predictor_factory(
            helpers.to_healpix(emission),
            truncate=4.0,
            dims=["cells"],
            conv_method=conv_method,
            device="cuda",
        )


@pytest.mark.parametrize("device", ["meta", torch.device("mps")])
def test_other_torch_device_types_are_refused_before_loading(
    monkeypatch, emission, device
):
    def unexpected_compute(*args, **kwargs):
        pytest.fail("the emission was loaded before device validation")

    monkeypatch.setattr(xr.Dataset, "compute", unexpected_compute)
    monkeypatch.setattr(helpers, "_load_emission", unexpected_compute)
    with pytest.raises(ValueError, match="Unsupported device"):
        _optimize(emission, "FFTConv", device)
    with pytest.raises(ValueError, match="Unsupported device"):
        helpers.compute_states(chunks={}, save=False, device=device)


@pytest.mark.skipif(torch.cuda.is_available(), reason="PyTorch sees a CUDA device")
def test_cuda_without_a_cuda_device_is_a_clear_error(emission):
    with pytest.raises(RuntimeError, match="sees no CUDA device"):
        _optimize(emission, "FFTConv", "cuda")
    with pytest.raises(RuntimeError, match="sees no CUDA device"):
        helpers.compute_states(chunks={}, save=False, device="cuda")


@requires_cuda
def test_missing_cuda_index_is_a_clear_error():
    with pytest.raises(RuntimeError, match="CUDA device"):
        helpers._check_device(f"cuda:{torch.cuda.device_count()}", "FFTConv")


def test_optimize_pdf_passes_the_device(monkeypatch, emission):
    device = torch.device("cpu")  # an object, to check that this one arrives
    seen = _stop_at(monkeypatch, "_get_predictor_factory")
    with pytest.raises(_Stop):
        _optimize(emission, "FFTConv", device)
    assert seen["device"] is device


def test_compute_states_passes_the_device(monkeypatch, emission):
    device = torch.device("cpu")
    seen = _stop_at(monkeypatch, "_build_estimator")
    with pytest.raises(_Stop):
        helpers.compute_states(
            ds=emission, target_root="unused", chunks={}, save=False, device=device
        )
    assert seen["device"] is device


@pytest.mark.filterwarnings("ignore:A .predictor_index. entry:RuntimeWarning")
@pytest.mark.parametrize("conv_method", TORCH_CONV_METHODS)
def test_build_estimator_passes_the_device(emission, tmp_path, conv_method):
    kwargs = {"max_sigma": 0.1} if conv_method == "FFTConv" else {}
    factory = helpers._get_predictor_factory(
        helpers.to_healpix(emission),
        truncate=None,
        dims=["cells"],
        conv_method=conv_method,
        **kwargs,
    )
    params = helpers._update_params_dict(factory, {"sigma": [6 * CELL_RAD]})
    (tmp_path / "parameters.json").write_text(json.dumps(params))

    device = torch.device("cpu")
    estimator, _, _ = helpers._build_estimator(
        emission, str(tmp_path), None, device=device
    )
    assert estimator.predictor_factory.keywords["device"] is device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("conv_method", TORCH_CONV_METHODS)
def test_torch_predictors_run_on_the_device_and_return_numpy(
    emission, conv_method, device
):
    ds = helpers.to_healpix(emission)
    kwargs = {"max_sigma": 0.1} if conv_method == "FFTConv" else {}

    def build(**device_kwargs):
        factory = helpers._get_predictor_factory(
            ds,
            truncate=None,
            dims=["cells"],
            conv_method=conv_method,
            **kwargs,
            **device_kwargs,
        )
        return factory(sigma=6 * CELL_RAD)

    predictor = build(device=device)
    tensors = [*predictor.layer.parameters(), *predictor.layer.buffers()]
    assert tensors
    assert {t.device.type for t in tensors} == {torch.device(device).type}

    x = emission["pdf"].values[0]
    out = predictor.predict(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape

    expected = build().predict(x)  # default device: the CPU
    np.testing.assert_allclose(
        out, expected, rtol=1e-4, atol=1e-6 * np.abs(expected).max()
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("conv_method", ["FFTConv", "LargeHealpixConv"])
def test_optimize_and_compute_states_on_the_device(
    emission, tmp_path, conv_method, device
):
    def run(root, device):
        params, _ = _optimize(
            emission,
            conv_method,
            device,
            save_parameters=True,
            target_root=str(root),
        )
        states = helpers.compute_states(
            ds=emission,
            target_root=str(root),
            chunks={"time": -1},
            save=False,
            device=device,
        )
        return params, states

    params, states = run(tmp_path / "device", device)
    assert params["predictor_factory"]["kwargs"]["device"] == device
    saved = json.loads((tmp_path / "device" / "parameters.json").read_text())
    assert saved["predictor_factory"]["kwargs"]["device"] == device

    values = states["states"].compute().data
    assert isinstance(values, np.ndarray)
    assert values.shape == emission["pdf"].shape

    params_cpu, states_cpu = run(tmp_path / "cpu", "cpu")
    np.testing.assert_allclose(params["sigma"], params_cpu["sigma"], rtol=1e-3)

    # The estimator encodes exact zeros as NaN. CPU and CUDA rounding can leave
    # a tiny positive probability in the same cell, so compare probability mass.
    expected = states_cpu["states"].fillna(0).to_numpy()
    actual = states["states"].fillna(0).to_numpy()
    np.testing.assert_allclose(
        actual, expected, rtol=1e-3, atol=1e-6 * np.abs(expected).max()
    )
