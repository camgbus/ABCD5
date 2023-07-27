import pygal

def plot_progress(df, y_column):
    plot = pygal.Line()
    for ds in set(df['Dataset']):
        plot.add(ds, df.loc[df['Dataset']==ds][y_column])
    plot.x_title = "Epoch"
    plot.x_labels = sorted(list(set(df['Epoch'])))
    plot.y_title = y_column
    return plot