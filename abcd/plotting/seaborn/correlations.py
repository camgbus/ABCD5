"""Plot the correlation of certain columns in a data frame.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlations(df, columns, method='pearson'):
    '''Plot symmetric correlations as a Seaborn Heatmap.'''
    
    df = df[columns].copy()

    # Get correlations between variables with Pandas method
    corr_matrix = df.corr(method=method) 

    # Get lower triangular correlation matrix (due to symmetri)
    corr_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(bool))
    
    # Plot heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, cmap="mako", annot=True, fmt='.1g')
    plt.tight_layout()
    return plt