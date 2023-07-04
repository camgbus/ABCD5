# abcd

## Data Pre-processing and Partitioning

### Imports
```
from abcd.utils.io import *
from abcd.local.paths import output_path
from abcd.local.paths import core_path
from abcd.data.read_data import read_events_general_info, add_subject_sex, add_event_connectivity_scores, get_subjects_events
from abcd.data.define_splits import inter_site_splits, save_restore_sex_fmri_splits
from abcd.data.divide_with_splits import divide_events_by_splits, divide_subjects
from abcd.data.read_labels import get_mh_labels, read_mh_scores
from abcd.data.load_data_local import load_data_local
```

### Filter subjects
Only fetch subject and event data (which contains the functional connectivity scores) for subjects which 1\) have resting state fMRI connectivity scores, 2\) have info on the sex assigned at birth, and 3\) have been scanned in one site.
```
subjects_df, events_df = get_subjects_events()
```

### Partition subjects by research site
```
site_splits = save_restore_sex_fmri_splits()
```

### Fetch mental health CBCL t-scores
```
mh_scores_df = get_mh_labels()
```

### Load the data
- Creates a local folder called model_data within the output path defined in paths.py. 
- Creates 21 folders within it called model01, model01..., model 21.
- Populates each of these 21 folders with the train/id test/ood test data.
  - model01 corresponds to the data partitioning which uses site01 as the ood test data.
- models_data is a dictionary which maps each model to its data: {model description (string) --> {data partition (string) --> pandas df}}
    - model descriptions are 'model01', 'model02',..., 'model21'
    - data partitions are 'events_train', 'events_id_test', 'events_ood_test', 'mh_scores_train', 'mh_scores_id_test', 'mh_scores_ood_test'
```
models_data = load_data_local(site_splits, mh_scores_df, events_df)
```