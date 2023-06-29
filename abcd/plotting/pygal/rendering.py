import os
from IPython.display import SVG, HTML, display

def save(plot, path, file_name):
    '''Save plot as an svg file.'''
    if ".svg" not in file_name:
        file_name = file_name+".svg"
    plot.render_to_file(os.path.join(path, file_name), disable_xml_declaration=True)
    
def display_svg(plot):
    '''Display plot, e.g. in an IPhython notebook.'''
    display(SVG(plot.render(disable_xml_declaration=True)))
    
def display_html(plot):
    '''Allows rendering interactive plots in IPython.'''
    # From https://feststelltaste.github.io/software-analytics/notebooks/vis/experimental/pandas-pygal/effective_charting.html
    base_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>
    <script type="text/javascript" src="https://kozea.github.io/pygal.js/2.0.x/pygal-tooltips.min.js""></script>
    </head>
    <body>
        <figure>
        {rendered_chart}
        </figure>
    </body>
    </html>
    """
    display(HTML(base_html.format(rendered_chart=plot.render(is_unicode=True))))