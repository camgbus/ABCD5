"""Plot the correlation of certain columns in a data frame.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlations(df, columns, method='pearson', figsize=(10,8), annot=True):
    '''Plot symmetric correlations as a Seaborn Heatmap.'''
    
    df = df[columns].copy()

    # Get correlations between variables with Pandas method
    corr_matrix = df.corr(method=method) 

    # Get lower triangular correlation matrix (due to symmetry)
    corr_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(bool))
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap="vlag", fmt='.1g', annot=annot)
    plt.tight_layout()
    return plt
