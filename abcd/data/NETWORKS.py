from collections import OrderedDict
from itertools import product

NETWORKS = OrderedDict([("ad","auditory"),
            ("cgc","cingulo-opercular"),
            ("ca","cingulo-parietal"),
            ("dt","default"),
            ("dla","dorsal attention"),
            ("fo","fronto-parietal"),
            ("n","none"),
            ("rspltp","retrosplenial temporal"),
            ("sa","salience"),
            ("smh","sensorimotor hand"),
            ("smm","sensorimotor mouth"),
            ("vta","ventral attention"),
            ("vs","visual")])

# All connection columns have the shape rsfmri_c_ngd_<Network A>_ngd_<Network B>
CONNECTIONS = ["rsfmri_c_ngd_{}_ngd_{}".format(n1, n2) for (n1, n2) in 
               product(NETWORKS.keys(), NETWORKS.keys())]