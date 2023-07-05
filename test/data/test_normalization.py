import pandas as pd
from abcd.data.var_tailoring.normalization import normalize_var

def test_normalization():
    values = [0.3, 0.1, 0.3, 8, 0.5, 0.2, 5, 0.3, 4, 5, 9, 2, 1.6, 1.2, 8]
    df = pd.DataFrame({"Y": values})
    new_df = normalize_var(df, "Y", "Y_n", norm=False)
    new_col_values = list(new_df["Y_n"])
    assert new_col_values[1] == 0
    assert new_col_values[10] == 1
    assert all(0 <= x <= 1 for x in new_col_values)