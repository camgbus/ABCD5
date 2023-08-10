import pygal

def plot_progress(df, y_column):
    plot = pygal.Line()
    ds_names = list(set(df['Dataset']))
    # So standard names are in the right order in the legend
    for ds_name in ['Test', 'Val', 'Train']:
        if ds_name in ds_names:
            ds_names.remove(ds_name)
            ds_names.insert(0, ds_name)
    for ds_name in ds_names:
        plot.add(ds_name, df.loc[df['Dataset']==ds_name][y_column])
    plot.x_title = "Epoch"
    plot.x_labels = sorted(list(set(df['Epoch'])))
    plot.y_title = y_column
    return plot