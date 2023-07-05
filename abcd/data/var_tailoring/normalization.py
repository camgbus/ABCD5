"""Normalize a variable in some range.
"""

def normalize_var(df, y, y_new_name, norm=False):
    '''
    Generate a new, discretized version of the variable, based on variable boundaries.
    
    Parameters:
        df (pandas.DataFrame): A Pandas df
        y (str): A continuous variable in the df
        y_new_name (str): The name assigned to the new (normalized) variable
        norm (True): If True, mean normalization is performed. If False, min_max normalization.
    Returns:
        new_df (pandas.DataFrame): A Pandas df with the new variable. Note that the old one is kept.
    '''
    # Ensure no missing values
    assert df[y].isna().sum() == 0
    if norm == True:
        df[y_new_name] = (df-df.mean())/df.std()
    else:
        df[y_new_name] = (df-df.min())/(df.max()-df.min())
    return df
