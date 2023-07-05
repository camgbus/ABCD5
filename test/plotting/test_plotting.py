"""Tests that ensure that plots CAN BE CREATED, so that they can be manually inspected in a
'test_objects' subdirectory within the outputs path.
"""
import os
import pandas as pd
from abcd.local.paths import output_path
from abcd.plotting.pygal.pygal_plots import pyramid_histogram
from abcd.plotting.pygal.rendering import save
from abcd.data.var_tailoring.discretization import discretize_var

def test_plot_pyramid_histogram():
    values = [0.3, 0.1, 0.3, 8, 0.5, 0.2, 5, 0.3, 4, 5, 9, 2, 1.6, 1.2, 8]
    df = pd.DataFrame({"Y": values})
    df = discretize_var(df, "Y", "Y_dr", nr_bins=3, by_freq=False)
    plot = pyramid_histogram(df, y="Y", hue="Y_dr", title = None, nr_bins=3)
    save(plot, path=os.path.join(output_path, "test_objects"), file_name="test_plot_pyramid_histogram")