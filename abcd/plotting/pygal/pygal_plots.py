"""Different plot examples generated with pygal
"""
import pygal
from abcd.data.var_tailoring.discretization import boundaries_by_frequency

def pyramid_histogram(df, y, hue, nr_bins = 5, title = None):
    '''Generate a pyramid plot with the distribution of a variable, grouped according to another.
    
    Parameters:
        df (pandas.DataFrame): A Pandas df
        y (str): A continuous variable in the df
        hue (str) : A categorical variable in the df
        nr_bins (int): Number of histogram bins
        title (str): Optional plot title
    Returns:
        plot (pygal.graph.pyramid.Pyramid): A pygal pyramid plot
    '''
    # Calculate boundaries dividing the y values
    boundaries = boundaries_by_frequency(list(df[y]), nr_bins)
    boundaries[-1] += 1
    
    # Divide df based on the categorical variable
    dfs = {hue_x: df.loc[df[hue] == hue_x] for hue_x in set(df[hue])}
        
    # Make plot
    plot = pygal.Pyramid(human_readable=True)
    if title:
        plot.title = title
    plot.x_labels = ["<= {0:.2f}".format(x) for x in boundaries[:-1]]
    plot.x_title = "Frequency"
    plot.y_title = y
    sorted_df_keys = sorted(list(dfs.keys()))
    for group_key in sorted_df_keys:
        frequancies = [len(dfs[group_key].loc[(dfs[group_key][y] >= boundaries[ix]) & 
                        (dfs[group_key][y] < boundaries[ix+1])]) for ix in range(len(boundaries)-1)]
        plot.add(group_key, frequancies)
    return plot   