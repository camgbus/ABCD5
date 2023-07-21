
import pandas as pd
import numpy as np
from abcd.data.assert_significance import is_different_from_mean

def test_col_significance():
    df = pd.DataFrame({'a': [100, 100, 50, 50, 1, 1, 100, 100, 50, 50, 1, 1, 100, 100, 50, 50, 1, 1],
                   'b': [0.1, 0.01, 0.5, 0.5, 0.1, 1, 1, 1, 0.5, 0.5, 0.01, 0.1, 1, 1, 0.5, 0.5, 0.1, 0.1]})
    assert is_different_from_mean(df['a'])
    assert is_different_from_mean(df['b'])
    assert not is_different_from_mean(df['a'], population_mean=np.mean(df['a']))