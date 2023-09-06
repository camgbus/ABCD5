import os

def save(plot, path, file_name, ending='.svg'):
    '''Save plot as an svg file.'''
    if ending not in file_name:
        file_name = file_name+ending
        
    plot.savefig(os.path.join(path, file_name), transparent=True, bbox_inches="tight", dpi = 300)

def display_svg(plot):
    '''Display plot, e.g. in an IPhython notebook.'''
    plot.show()