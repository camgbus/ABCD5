import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.9)

def plot_scatter(df, x_col, y_col, hue_col, title, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x_col,
                    y=y_col, 
                    hue=hue_col, palette=sns.color_palette("mako_r", as_cmap=True),
                    data=df)
    plt.title(title, weight='bold')
    plt.legend(title=hue_col, loc='lower right')
    return plt