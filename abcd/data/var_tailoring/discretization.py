"""Discretize a continuous variable into a range of classes.
"""

import numpy as np

def discretize_var(df, y, y_new_name, nr_bins=5, by_freq=False):
    '''
    Generate a new, discretized version of the variable, based on variable boundaries.
    
    Parameters:
        df (pandas.DataFrame): A Pandas df
        y (str): A continuous variable in the df
        y_new_name (str): The name assigned to the new (categorical) variable
        nr_bins (int): The number of new values the variable can take
        by_freq (bool): If True, the range is divided based on the frequency in the df. If False,
            the it is divided based on the numerical range.
    Returns:
        new_df (pandas.DataFrame): A Pandas df with the new variable. Note that the old one is kept.
    '''
    # Ensure no missing values
    assert df[y].isna().sum() == 0
    # Calculate boundaries
    if by_freq:
        boundaries = boundaries_by_frequency(list(df[y]), nr_bins)
    else:
        boundaries = boundaries_by_range(list(df[y]), nr_bins)
    boundaries = [round(x, 2) for x in boundaries[1:]]
    assert len(boundaries) == len(set(boundaries))  # Ensure that nr_bins is not too large
    # Discretize
    def dicretize(v, boundaries):
        for bound in boundaries:
            if v <= bound:
                return "<= {0:.2f}".format(bound)    
    new_col = [dicretize(x, boundaries) for x in list(df[y])]
    df[y_new_name] = new_col
    return df
    
def boundaries_by_range(values, nr_bins):
     '''Generate boundaries that divide values based on the range, including min and max, so
     nr_bins == len(boundaries) - 1'''
     min_val, max_val = min(values), max(values)
     bound_mul = (max_val-min_val)/nr_bins
     boundaries = [min_val + ix*bound_mul for ix in range(nr_bins+1)]
     return boundaries

def boundaries_by_frequency(values, nr_bins):
     '''Generate boundaries that divide values into similarly large groups, including min and max, 
     so nr_bins == len(boundaries) - 1'''
     bound_mul = 100/nr_bins
     percentile_boundaries = [ix*bound_mul for ix in range(nr_bins+1)]
     boundaries = [np.percentile(values, x) for x in percentile_boundaries]
     return boundaries