import matplotlib
import pandas as pd

from pangeo_fish import track_bounds, visualization
from pangeo_fish.tests.test_track_bounds import RELEASE, synthetic_tag

matplotlib.use("Agg")


def test_plot_track_bounds():
    ds = synthetic_tag()
    start = track_bounds.detect_start_of_track(ds, activity_threshold=0.2)
    end = track_bounds.detect_end_of_track(
        ds, start=start["start"], activity_threshold=0.2
    )
    fig = visualization.plot_track_bounds(
        ds,
        start,
        end,
        events={"release": RELEASE, "recapture": pd.Timestamp("2023-05-09 11:00")},
        activity_threshold=0.2,
    )

    assert len(fig.axes) == 2
    matplotlib.pyplot.close(fig)
