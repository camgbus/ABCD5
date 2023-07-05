import pandas as pd
from abcd.data.var_tailoring.discretization import discretize_var

def test_discretization():
    values = [0.3, 0.1, 0.3, 8, 0.5, 0.2, 5, 0.3, 4, 5, 9, 2, 1.6, 1.2, 8]
    df = pd.DataFrame({"Y": values})
    new_df = discretize_var(df, "Y", "Y_dr", nr_bins=3, by_freq=False)
    new_col_values = list(new_df["Y_dr"])
    new_possible_values = sorted(list(set(new_col_values)))
    freqs  = [new_col_values.count(x) for x in new_possible_values]
    assert new_col_values == ['<= 3.07', '<= 3.07', '<= 3.07', '<= 9.00', '<= 3.07', '<= 3.07', '<= 6.03', '<= 3.07', '<= 6.03', '<= 6.03', '<= 9.00', '<= 3.07', '<= 3.07', '<= 3.07', '<= 9.00']
    assert new_possible_values == ['<= 3.07', '<= 6.03', '<= 9.00']
    assert freqs == [9, 3, 3]

    new_df = discretize_var(df, "Y", "Y_df", nr_bins=3, by_freq=True)
    new_col_values = list(new_df["Y_df"])
    new_possible_values = sorted(list(set(new_col_values)))
    freqs  = [new_col_values.count(x) for x in new_possible_values]
    assert new_col_values == ['<= 0.43', '<= 0.43', '<= 0.43', '<= 9.00', '<= 4.33', '<= 0.43', '<= 9.00', '<= 0.43', '<= 4.33', '<= 9.00', '<= 9.00', '<= 4.33', '<= 4.33', '<= 4.33', '<= 9.00']
    assert new_possible_values == ['<= 0.43', '<= 4.33', '<= 9.00']
    assert freqs == [5, 5, 5]