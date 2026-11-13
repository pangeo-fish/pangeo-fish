from types import SimpleNamespace as NS
from unittest.mock import Mock

import pandas as pd
import pytest

from pangeo_fish import helpers

IBI = "cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m"
IBI_SSH = "cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m"
IBI_STATIC = "cmems_mod_ibi_phy_my_0.027deg-3D_static"
NWS = "cmems_mod_nws_phy-t_my_7km-3D_P1D-m"
NWS_STATIC = "cmems_mod_nws_phy_my_7km-3D_static"
GLO = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
GLO_STATIC = "cmems_mod_glo_phy_my_0.083deg_static"

BBOX = {"latitude": (50.7, 53.0), "longitude": (1.5, 5.08), "max_depth": 50}
PERIOD = slice("2023-05-03", "2023-06-08")
DAY_MS = 86_400_000.0


def coordinate(
    name, minimum=None, maximum=None, step=None, unit="degrees", values=None
):
    return NS(
        coordinate_id=name,
        minimum_value=minimum,
        maximum_value=maximum,
        step=step,
        values=values,
        coordinate_unit=unit,
    )


def variable(name, lon=(-19.08, 5.08, 0.0278), lat=(26.17, 56.08, 0.0278), time=None):
    t0, t1 = (
        pd.Timestamp(t).value // 1_000_000 for t in time or ("1993-01-01", "2026-07-21")
    )
    return NS(
        short_name=name,
        coordinates=[
            coordinate("time", t0, t1, DAY_MS, "milliseconds since 1970-01-01"),
            coordinate("latitude", *lat),
            coordinate("longitude", *lon),
        ],
    )


def dataset(dataset_id, parts):
    """Described dataset; ``parts`` maps part names to lists of variables.

    The default version comes first, as ``copernicusmarine.describe`` sorts it.
    """
    part_list = [
        NS(
            name=name,
            services=[
                NS(service_name="original-files", variables=[]),
                NS(service_name="arco-geo-series", variables=variables),
            ],
        )
        for name, variables in parts.items()
    ]
    return NS(
        dataset_id=dataset_id,
        versions=[NS(label="202511", parts=part_list), NS(label="202012", parts=[])],
    )


def static(dataset_id, part="bathy"):
    names = ["deptho", "deptho_lev", "mask"]
    return dataset(
        dataset_id,
        {part: [NS(short_name=n, coordinates=[]) for n in names], "coords": []},
    )


def catalogue(*datasets):
    return {d.dataset_id: d for d in datasets}


# --- pure helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    "bbox, lon_range, lat_range, expected",
    [
        pytest.param(BBOX, (-19.08, 5.08), (26.17, 56.08), 1.0, id="inside"),
        pytest.param(
            {"longitude": (4.0, 6.16), "latitude": (50.7, 53.0)},
            (-19.08, 5.08),
            (26.17, 56.08),
            0.5,
            id="half-longitude",
        ),
        pytest.param(
            {"longitude": (0, 1), "latitude": (0, 60)},
            (-10, 10),
            (0, 30),
            0.5 / 0.866025,  # sin(30) / sin(60): area-weighted, not 0.5
            id="area-weighted-latitude",
        ),
        pytest.param(
            {"longitude": (20.0, 21.0), "latitude": (60.0, 61.0)},
            (-19.08, 5.08),
            (26.17, 56.08),
            0.0,
            id="outside",
        ),
        pytest.param(
            {"longitude": (170.0, 175.0), "latitude": (60.0, 61.0)},
            (-180.0, 179.92),
            (-80.0, 90.0),
            1.0,
            id="global-grid",
        ),
    ],
)
def test_bbox_share(bbox, lon_range, lat_range, expected):
    assert helpers._bbox_share(bbox, lon_range, lat_range) == pytest.approx(
        expected, rel=1e-5
    )


@pytest.mark.parametrize(
    "coord, expected",
    [
        pytest.param(coordinate("x", -1, 2, 0.5), (-1.0, 2.0, 0.5), id="min-max"),
        pytest.param(
            coordinate("x", values=[3.0, 1.0, 2.0, 2.5]),
            (1.0, 3.0, 0.5),
            id="values-median-step",
        ),
        pytest.param(coordinate("x", values=[4.0]), (4.0, 4.0, None), id="one-value"),
        pytest.param(coordinate("x"), (None, None, None), id="empty"),
    ],
)
def test_coordinate_extent(coord, expected):
    assert helpers._coordinate_extent(coord) == expected


@pytest.mark.parametrize(
    "value, unit, expected",
    [
        (DAY_MS, "milliseconds since 1970-01-01", "1970-01-02"),
        (DAY_MS, None, "1970-01-02"),
        (3600, "seconds since 1950-01-01T00:00:00Z", "1950-01-01T01:00"),
        (24, "hours since 2000-01-01", "2000-01-02"),
        (90, "minutes since 2000-01-01 00:00:00", "2000-01-01T01:30"),
    ],
)
def test_describe_time(value, unit, expected):
    assert helpers._describe_time(value, unit) == pd.Timestamp(expected)


def test_describe_version_and_part():
    # The toolbox puts the default version first; a larger label later (e.g.
    # not released yet) must not be picked.
    first = NS(label="202411", parts=[NS(name="default"), NS(name="bathy")])
    described = NS(versions=[first, NS(label="202511", parts=[])])
    assert helpers._describe_version(described) is first
    assert helpers._describe_part(first).name == "default"
    assert helpers._describe_part(first, "bathy").name == "bathy"
    assert helpers._describe_part(first, "missing") is None


@pytest.mark.parametrize(
    "dataset_id, expected",
    [
        (
            IBI,
            [
                (IBI, None, ["thetao"]),
                (IBI_SSH, None, ["zos"]),
                (IBI_STATIC, "bathy", ["deptho", "mask"]),
            ],
        ),
        (GLO, [(GLO, None, ["thetao", "zos"]), (GLO_STATIC, None, ["deptho", "mask"])]),
    ],
)
def test_copernicus_requests(dataset_id, expected):
    assert helpers._copernicus_requests(dataset_id) == expected


@pytest.mark.parametrize(
    "datasets, supported, note",
    [
        pytest.param(
            [
                dataset(IBI, {"default": [variable("thetao")]}),
                dataset(IBI_SSH, {"default": [variable("zos")]}),
                static(IBI_STATIC),
            ],
            True,
            "",
            id="ibi-split",
        ),
        pytest.param(
            [dataset(IBI, {"default": [variable("thetao")]}), static(IBI_STATIC)],
            False,
            f"companion dataset {IBI_SSH!r} not found",
            id="missing-companion",
        ),
        pytest.param(
            [
                dataset(IBI, {"default": [variable("thetao")]}),
                dataset(IBI_SSH, {"default": [variable("zos")]}),
                static(IBI_STATIC, part="other"),
            ],
            False,
            f"{IBI_STATIC!r} has no part 'bathy'",
            id="missing-static-part",
        ),
        pytest.param(
            [dataset(NWS, {"default": [variable("thetao")]}), static(NWS_STATIC)],
            False,
            "no zos in this dataset (load_model maps a separate SSH dataset "
            "only for the IBI reanalysis)",
            id="no-zos",
        ),
    ],
)
def test_load_model_support(datasets, supported, note):
    dataset_id = datasets[0].dataset_id
    assert helpers._load_model_support(dataset_id, catalogue(*datasets)) == (
        supported,
        note,
    )


# --- list_reference_models -----------------------------------------------


@pytest.fixture
def fake_describe(monkeypatch):
    """Patch ``copernicusmarine.describe`` with products built by the test."""

    def install(products):
        def fake(product_id, **kwargs):
            if product_id not in products:
                raise helpers.copernicusmarine.ProductNotFound(product_id)
            return NS(products=[NS(datasets=products[product_id])])

        mock = Mock(side_effect=fake)
        monkeypatch.setattr(helpers.copernicusmarine, "describe", mock)
        return mock

    return install


def test_list_reference_models(fake_describe):
    describe = fake_describe(
        {
            "IBI_MY": [
                dataset(IBI, {"default": [variable("thetao")]}),
                dataset(IBI_SSH, {"default": [variable("zos")]}),
                static(IBI_STATIC),
            ],
            "NWS_MY": [
                dataset(
                    NWS,
                    {
                        "default": [
                            variable(
                                "thetao",
                                lon=(-19.89, 13.0, 0.1111),
                                lat=(40.07, 65.0, 0.0667),
                                time=("1993-01-01", "2026-08-31"),
                            )
                        ]
                    },
                ),
                static(NWS_STATIC),
            ],
            "GLO_MY": [
                dataset(
                    GLO,
                    {
                        "default": [
                            variable(
                                name,
                                lon=(-180.0, 179.92, 0.0833),
                                lat=(-80.0, 90.0, 0.0833),
                                time=("1993-01-01", "2021-06-30"),
                            )
                            for name in ("thetao", "zos")
                        ]
                    },
                ),
                static(GLO_STATIC, part="default"),
            ],
        }
    )
    candidates = {
        GLO: "GLO_MY",
        NWS: "NWS_MY",
        IBI: "IBI_MY",
        "does_not_exist": "IBI_MY",
        "other": "UNKNOWN_PRODUCT",
    }
    table = helpers.list_reference_models(
        bbox=BBOX, time_slice=PERIOD, candidates=candidates
    )

    assert describe.call_count == 4  # one call per product
    # full coverage of bbox and period first, finest grid first
    assert list(table["dataset_id"]) == [IBI, NWS, GLO, "does_not_exist", "other"]
    table = table.set_index("dataset_id")

    ibi = table.loc[IBI]
    assert ibi["version"] == "202511"
    assert ibi["time_start"] == pd.Timestamp("1993-01-01")
    assert ibi["time_end"] == pd.Timestamp("2026-07-21")
    assert ibi["covers_bbox"] and ibi["covers_period"] and ibi["load_model"]
    assert ibi["dy_km"] == pytest.approx(0.0278 * 111.32)
    assert ibi["dx_km"] < ibi["dy_km"]

    nws = table.loc[NWS]
    assert nws["covers_bbox"] and nws["covers_period"]
    assert not nws["load_model"]
    assert nws["note"].startswith("no zos in this dataset")

    glo = table.loc[GLO]
    assert glo["covers_bbox"] and glo["load_model"]
    assert not glo["covers_period"]

    assert table.loc["does_not_exist", "note"] == "dataset not found in product"
    assert table.loc["other", "note"].startswith("describe failed")
    assert not table.loc["other", ["covers_bbox", "covers_period", "load_model"]].any()


def test_unexpected_describe_response_is_raised(fake_describe):
    # Only a missing product and network errors become a row note.
    fake_describe({"IBI_MY": [NS(dataset_id=IBI)]})  # no ``versions``
    with pytest.raises(AttributeError):
        helpers.list_reference_models(
            bbox=BBOX, time_slice=PERIOD, candidates={IBI: "IBI_MY"}
        )
