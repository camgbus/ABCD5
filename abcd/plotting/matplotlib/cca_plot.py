"""Plot for Canonical Correlation Analysis performed with Sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_cca_rotations(cca_model, x_vars, y_vars, x_component=0, y_component=1,
             x_color='#6AB8E8', y_color='#A99EEC', x_label='Features', y_label='Targets'):
  # Obtain the rotation matrices
  xrot = cca_model.x_rotations_
  yrot = cca_model.y_rotations_

  # Put them together in a numpy matrix
  xyrot = np.vstack((xrot,yrot))

  nr_variables = xyrot.shape[0]
  all_vars = x_vars + y_vars

  plt.figure(figsize=(10, 10))
  plt.xlim((-1,1))
  plt.ylim((-1,1))
  ax = plt.gca()
  for grid_circle_rad in [1, 0.75, 0.5, 0.25]:
    circle = plt.Circle((0, 0), grid_circle_rad, facecolor='white', edgecolor='gray')
    ax.add_patch(circle)

  # Plot an arrow and a text label for each variable
  for var_i in range(nr_variables):
    x = xyrot[var_i, x_component]
    y = xyrot[var_i, y_component]
    
    plt.arrow(0,0,x,y, shape='full', color='gray', length_includes_head=True, head_width=0)
    plt.scatter(x, y, color=x_color if var_i >= len(x_vars) else y_color)
    plt.text(x+0.01,y+0.01,all_vars[var_i], weight='bold', color=x_color if var_i >= len(x_vars) else y_color)
    
  # Costum legend for features and targets
  legend_elements = [Patch(facecolor=x_color, edgecolor=x_color, label=x_label),
                    Patch(facecolor=y_color, edgecolor=y_color, label=y_label)]
  ax.legend(handles=legend_elements, loc='upper right')

  # Axis labels
  plt.xlabel("Dimension {}".format(x_component+1))
  plt.ylabel("Dimension {}".format(y_component+1))

  return plt