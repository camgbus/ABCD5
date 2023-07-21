"""Test whether a variable in a data frame is significantly different from the expected population 
mean with a one sample t-test.
"""
import scipy.stats as stats
import numpy as np

def is_different_from_mean(values, population_mean=0, p_threshold=0.05):
    ''' Perform a one sample t-test to ensure a significant difference from an expected mean '''
    t_statistic, p_value = stats.ttest_1samp(a=values, popmean=population_mean)
    return p_value < p_threshold

def k_greatest_stat(df, vars, population_mean=0, k=5):
    ''' Select the k vars with highest t_statistic, defined as (np.mean(x) - popmean)/se '''
    vars = sorted(vars, key=lambda v: stats.ttest_1samp(a=df[v], popmean=population_mean)[0], 
                  reverse=True)
    return vars[:k]
    
def k_greatest_std(df, vars, k=5):
    ''' Select the k vars with highest standard deviation '''
    vars = sorted(vars, key=lambda v: np.std(df[v]), reverse=True)
    return vars[:k]