import os

def save(plot, path, file_name):
    '''Save plot as an svg file.'''
    if ".svg" not in file_name:
        file_name = file_name+".svg"
        
    plot.savefig(os.path.join(path, file_name), facecolor='w', bbox_inches="tight", dpi = 300)

def display_svg(plot):
    '''Display plot, e.g. in an IPhython notebook.'''
    plot.show()