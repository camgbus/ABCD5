"""Normalize a variable in some range.
"""

def normalize_var(df, y, y_new_name, norm=False, y_min=None, y_max=None):
    '''
    Generate a new, discretized version of the variable, based on variable boundaries.
    
    Parameters:
        df (pandas.DataFrame): A Pandas df
        y (str): A continuous variable in the df
        y_new_name (str): The name assigned to the new (normalized) variable
        norm (True): If True, mean normalization is performed. If False, min_max normalization.
        y_min, y_max: provided minimum and maximum values to perform min max normalization with. If None, the min and max of the variable are used.
    Returns:
        new_df (pandas.DataFrame): A Pandas df with the new variable. Note that the old one is kept.
    '''
    # Ensure no missing values
    assert df[y].isna().sum() == 0
    if norm == True: # mean normalization
        df[y_new_name] = (df[y]-df[y].mean())/df[y].std()
    else: # min max normalization
        if y_min and y_max: # provided min and max
            df[y_new_name] = (df[y]-y_min)/(y_max-y_min)
        else: # use min and max y
            df[y_new_name] = (df[y]-df[y].min())/(df[y].max()-df[y].min())
    return df
