from types import SimpleNamespace as NS
from unittest.mock import Mock

import pandas as pd
import pytest

from pangeo_fish import helpers

DAY_MS = 86_400_000.0
BBOX = {"latitude": (50.7, 53.0), "longitude": (1.5, 5.08), "max_depth": 50}
PERIOD = slice("2023-05-03", "2023-06-08")


def coordinate(name, minimum, maximum, step, unit="degrees"):
    return NS(
        coordinate_id=name,
        minimum_value=minimum,
        maximum_value=maximum,
        step=step,
        values=None,
        coordinate_unit=unit,
    )


def variable(name, lon=(-19.08, 5.08, 0.0278), lat=(26.17, 56.08, 0.0278), time=None):
    time = time or ("1993-01-01", "2026-07-21")
    t0, t1 = (pd.Timestamp(t).value // 1_000_000 for t in time)
    return NS(
        short_name=name,
        coordinates=[
            coordinate("time", t0, t1, DAY_MS, "milliseconds since 1970-01-01"),
            coordinate("depth", None, None, None, "m"),
            coordinate("latitude", *lat),
            coordinate("longitude", *lon),
        ],
    )


def dataset(dataset_id, variables, parts="default"):
    if isinstance(variables, dict):  # static: {part name: [variable names]}
        part_list = [
            NS(
                name=name,
                services=[
                    NS(
                        service_name="static-arco",
                        variables=[NS(short_name=v, coordinates=[]) for v in names],
                    )
                ],
            )
            for name, names in variables.items()
        ]
    else:
        part_list = [
            NS(
                name=parts,
                services=[
                    NS(service_name="original-files", variables=[]),
                    NS(service_name="arco-geo-series", variables=variables),
                ],
            )
        ]
    return NS(
        dataset_id=dataset_id,
        versions=[NS(label="202012", parts=[]), NS(label="202511", parts=part_list)],
    )


STATIC_PARTS = {"bathy": ["deptho", "deptho_lev", "mask"], "coords": ["e1t"]}

PRODUCTS = {
    "IBI_MY": [
        dataset("cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m", [variable("thetao")]),
        dataset("cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m", [variable("zos")]),
        dataset("cmems_mod_ibi_phy_my_0.027deg-3D_static", STATIC_PARTS),
    ],
    "NWS_MY": [
        dataset(
            "cmems_mod_nws_phy-t_my_7km-3D_P1D-m",
            [
                variable(
                    "thetao",
                    lon=(-19.89, 13.0, 0.1111),
                    lat=(40.07, 65.0, 0.0667),
                    time=("1993-01-01", "2026-08-31"),
                )
            ],
        ),
        dataset("cmems_mod_nws_phy_my_7km-3D_static", STATIC_PARTS),
    ],
    "GLO_MY": [
        dataset(
            "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            [
                variable(
                    name,
                    lon=(-180.0, 179.92, 0.0833),
                    lat=(-80.0, 90.0, 0.0833),
                    time=("1993-01-01", "2021-06-30"),
                )
                for name in ("thetao", "zos")
            ],
        ),
        dataset("cmems_mod_glo_phy_my_0.083deg_static", STATIC_PARTS),
    ],
}
CANDIDATES = {
    "cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m": "IBI_MY",
    "cmems_mod_nws_phy-t_my_7km-3D_P1D-m": "NWS_MY",
    "cmems_mod_glo_phy_my_0.083deg_P1D-m": "GLO_MY",
}


@pytest.fixture
def describe(monkeypatch):
    def fake(product_id, **kwargs):
        if product_id not in PRODUCTS:
            raise ValueError(f"unknown product {product_id}")
        return NS(products=[NS(datasets=PRODUCTS[product_id])])

    mock = Mock(side_effect=fake)
    monkeypatch.setattr(helpers.copernicusmarine, "describe", mock)
    return mock


def test_table(describe):
    table = helpers.list_reference_models(
        bbox=BBOX, time_slice=PERIOD, candidates=CANDIDATES
    ).set_index("dataset_id")

    assert describe.call_count == 3
    ibi = table.loc["cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m"]
    assert ibi["version"] == "202511"
    assert ibi["time_start"] == pd.Timestamp("1993-01-01")
    assert ibi["time_end"] == pd.Timestamp("2026-07-21")
    assert ibi["covers_bbox"] and ibi["covers_period"] and ibi["load_model"]
    assert ibi["bbox_share"] == pytest.approx(1.0)
    assert ibi["dy_km"] == pytest.approx(0.0278 * 111.32)
    assert ibi["dx_km"] < ibi["dy_km"]

    nws = table.loc["cmems_mod_nws_phy-t_my_7km-3D_P1D-m"]
    assert nws["covers_bbox"] and nws["covers_period"]
    assert not nws["load_model"]
    assert "zos" in nws["note"]

    glo = table.loc["cmems_mod_glo_phy_my_0.083deg_P1D-m"]
    assert glo["covers_bbox"] and glo["load_model"]
    assert not glo["covers_period"]

    # full coverage of bbox and period first, finest grid first
    assert list(table.index) == list(CANDIDATES)


def test_partial_bbox_share(describe):
    bbox = {"latitude": (50.7, 53.0), "longitude": (4.0, 6.16)}
    table = helpers.list_reference_models(
        bbox=bbox, time_slice=PERIOD, candidates=CANDIDATES
    ).set_index("dataset_id")
    ibi = table.loc["cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m"]
    assert ibi["bbox_share"] == pytest.approx(0.5)
    assert not ibi["covers_bbox"]
    assert table.loc["cmems_mod_nws_phy-t_my_7km-3D_P1D-m", "covers_bbox"]
    assert table.index[0] == "cmems_mod_nws_phy-t_my_7km-3D_P1D-m"


def test_bbox_outside(describe):
    bbox = {"latitude": (60.0, 61.0), "longitude": (20.0, 21.0)}
    table = helpers.list_reference_models(
        bbox=bbox, time_slice=PERIOD, candidates=CANDIDATES
    ).set_index("dataset_id")
    assert table.loc["cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m", "bbox_share"] == 0
    assert table.loc["cmems_mod_glo_phy_my_0.083deg_P1D-m", "bbox_share"] == 1


def test_missing_companion_and_errors(describe):
    candidates = {
        "cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m": "IBI_MY",
        "does_not_exist": "IBI_MY",
        "other": "UNKNOWN_PRODUCT",
    }
    original = PRODUCTS["IBI_MY"]
    PRODUCTS["IBI_MY"] = original[:1] + original[2:]  # drop the SSH dataset
    try:
        table = helpers.list_reference_models(
            bbox=BBOX, time_slice=PERIOD, candidates=candidates
        ).set_index("dataset_id")
    finally:
        PRODUCTS["IBI_MY"] = original

    assert describe.call_count == 2  # one call per product
    ibi = table.loc["cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m"]
    assert not ibi["load_model"]
    assert "cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m" in ibi["note"]
    assert table.loc["does_not_exist", "note"] == "dataset not found in product"
    assert table.loc["other", "note"].startswith("describe failed")
    assert not table.loc["other", "covers_period"]


def test_static_part_required(describe):
    original = PRODUCTS["IBI_MY"]
    PRODUCTS["IBI_MY"] = original[:2] + [
        dataset("cmems_mod_ibi_phy_my_0.027deg-3D_static", {"coords": ["e1t"]})
    ]
    try:
        table = helpers.list_reference_models(
            bbox=BBOX,
            time_slice=PERIOD,
            candidates={"cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m": "IBI_MY"},
        )
    finally:
        PRODUCTS["IBI_MY"] = original
    assert not table.loc[0, "load_model"]
    assert "bathy" in table.loc[0, "note"]
