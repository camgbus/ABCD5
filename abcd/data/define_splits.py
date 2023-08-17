"""Division of subjects for cross-validation.

All subjects for which there are resting state fMRI connectivity scores
are divided into the site of acquisition. Each of the 22 sites is further 
divided into five splits, keeping siblings together.
"""

import random
from tqdm import tqdm
import abcd.utils.io as io
from abcd.data.read_data import get_subjects_events, get_subjects_events_visits
from abcd.local.paths import output_path

SITES = ["site{:02d}".format(nr_id) for nr_id in range(1, 22)]

def save_restore_visit_splits(target_visits = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1'], k=5, seed=0):
    name = "splits_{}_{}_{}".format('-'.join([v[0] for v in target_visits]), k, seed)
    try:
        splits = io.load_json(output_path, name)
    except:
        subjects_df, _ = get_subjects_events_visits(target_visits=target_visits)
        splits = inter_site_splits(subjects_df, k=k, seed=seed)
        io.dump_json(splits, output_path, name)
    return splits
    
def save_restore_sex_fmri_splits(k=5, seed=0):
    name = "splits_sex_fmri_{}_{}".format(k, seed)
    try:
        splits = io.load_json(output_path, name)
    except:
        subjects_df, _ = get_subjects_events()
        splits = inter_site_splits(subjects_df, k=k, seed=seed)
        io.dump_json(splits, output_path, name)
    return splits

def inter_site_splits(subjects_df, k=3, seed=0):
    '''Divides subjects first by site, then randomly into k folds (keeping family members together).

    Parameters:
        subjects_df (pandas.DataFrame): Subjects dataframe
    Returns:
        site_splits ({str -> {str -> [str]}}): A dictionary linking each site ID to a k-item long 
            dict linking a split ID to a subject ID list
    '''
    random.seed(seed)
    site_splits = dict()
    for site_id in tqdm(SITES):
        family_groups = []
        site_df = subjects_df.loc[subjects_df["site_id_l"] == site_id]
        family_ids = set(site_df["rel_family_id"])
        for family_id in family_ids:
            family_subjects = list(site_df.loc[site_df["rel_family_id"] == family_id]["src_subject_id"])
            family_groups.append(family_subjects)
        splits = {str(split_ix) : [] for split_ix in range(k)}
        # Shuffle items
        random.shuffle(family_groups)
        # After shuffling, order with the largest groups first
        family_groups.sort(key=lambda x: len(x), reverse=True)
        # Assign each item, one after another, to the group with less items
        for item in family_groups:
            smallest_group_ix = min(splits.keys(), key=lambda x: len(splits[x]))
            splits[smallest_group_ix] += item
        site_splits[site_id] = splits
    return site_splits

def assert_one_site_per_family(subjects_df):
    '''Currently, there are no cases of different family members being scanned at different sites.
    Make sure this keeps being the case.
    
    Parameters:
        subjects_df (pandas.DataFrame): Subjects dataframe
    '''
    family_ids = set(subjects_df["rel_family_id"])
    for family_id in tqdm(family_ids):
        sites = subjects_df.loc[subjects_df["rel_family_id"] == family_id]["site_id_l"]
        assert len(set(sites)) == 1