"Scatter plot of true y vs. regression predictions"

import matplotlib.pyplot as plt

def plot_regression_scatter(y_true, y_pred, title=''):
    """
    Args: y_true, y_pred: (n, 1) numpy arrays 
    """
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True y')
    plt.ylabel('Predicted y')
    plt.title(title)
    return plt