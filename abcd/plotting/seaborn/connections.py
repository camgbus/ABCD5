
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abcd.local.paths import data_path, output_path
from abcd.data.read_connections import networks
from collections import OrderedDict

figure_size=(20, 10)
annot=True
vmin=None
vmax=None

networks = OrderedDict([("A","Network A"),
            ("B","Network B"),
            ("C","Network C")])

df = pd.DataFrame({'id': [1, 2, 3], 'rsfmri_c_ngd_A_ngd_A': [1, 1, 1], 'rsfmri_c_ngd_B_ngd_B': [2, 2, 2], 'rsfmri_c_ngd_C_ngd_C': [3, 3, 3], 'rsfmri_c_ngd_A_ngd_B': [1, 1, 2], 'rsfmri_c_ngd_B_ngd_A': [5, 10, 15], 'rsfmri_c_ngd_A_ngd_C': [1, 2, 3], 'rsfmri_c_ngd_C_ngd_A': [50, 100, 150], 'rsfmri_c_ngd_B_ngd_C': [10, 10, 10], 'rsfmri_c_ngd_C_ngd_B': [50, 40, 45]})

network_names = list(networks.values())

connections_a_to_b = []
for network_a_id, network_a in networks.items():
    connections_from_a = []
    for network_b_id, network_b in networks.items():
        connections = df["rsfmri_c_ngd_{}_ngd_{}".format(network_a_id, network_b_id)]
        connections_from_a.append(np.mean(connections))
    connections_a_to_b.append(connections_from_a)
        
df = pd.DataFrame(connections_a_to_b)
  
plt.figure()
sns.set(rc={'figure.figsize':figure_size})
ax = sns.heatmap(df, annot=annot, vmin=vmin, vmax=vmax, cmap=None, xticklabels=network_names, yticklabels=network_names)
ax.set(xlabel='To network', ylabel='From network')
plt.savefig(os.path.join(output_path, "connections.png"), facecolor='w', bbox_inches="tight", dpi = 300)