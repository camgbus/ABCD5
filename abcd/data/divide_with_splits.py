"""Use previously defined split values to divide a dataframe of events into training, ID validation
and OOD testing.
"""

import pandas as pd

def divide_events_by_splits(events_df, site_splits, site_id, mh_scores_df):
    '''
    1) Divide a df of events into three for training, id id testing and OOD testing, based on the 
    src_subject_id
    2) Divide a df of raw cbcl mental health scores into three for training, id id testing, and OOD
    testing such that they correspond to the event dataframe partitions.
    '''
    subject_division = divide_subjects(site_splits, site_id)

    #step 1
    events_train = events_df[events_df['src_subject_id'].isin(subject_division["train"])]
    events_id_test = events_df[events_df['src_subject_id'].isin(subject_division["id_test"])]
    events_ood_test = events_df[events_df['src_subject_id'].isin(subject_division["ood_test"])]
    assert len(events_train) + len(events_id_test) + len(events_ood_test) == len(events_df)
    
    #step 2
    events_train, mh_scores_train = partition_mh_scores(events_train, mh_scores_df)
    events_id_test, mh_scores_id_test = partition_mh_scores(events_id_test, mh_scores_df)
    events_ood_test, mh_scores_ood_test = partition_mh_scores(events_ood_test, mh_scores_df)
    
    return events_train, events_id_test, events_ood_test, mh_scores_train, mh_scores_id_test, mh_scores_ood_test


def partition_mh_scores(events_partition, mh_scores_df):
    '''Given events_partition, which is one of the train/id_test/ood_test partitions of events_df, and
    mh_scores_df, which is the unfiltered df of raw mental health scores, create mh_scores_filtered and
    update events_partition, such that their rows correspond to the same data points.

    Parameters:
        events_df: pd dataframe
        mh_scores_df: pd dataframe
    Returns:
        events_df: pd dataframe
        mh_scores_filtered: pd dataframe
    '''
        
    #join dataframes on shared primary key ('src_subject_id', 'eventname')
    common_df = pd.merge(events_partition[['src_subject_id', 'eventname']], 
                        mh_scores_df[['src_subject_id', 'eventname']], 
                        how='inner', on=['src_subject_id', 'eventname'])

    #filter out the rows in events_partition which do not exist in common_df
    events_partition = pd.merge(events_partition, common_df, how='inner', on=['src_subject_id', 'eventname'])

    #filter out the rows in mh_scores_df which do not exist in common_df
    mh_scores_filtered = pd.merge(mh_scores_df, common_df, how='inner', on=['src_subject_id', 'eventname'])

    #sort dfs (they should already be in the correct order, but better to be safe)
    events_partition.sort_values(['src_subject_id', 'eventname'], inplace=True)
    mh_scores_filtered.sort_values(['src_subject_id', 'eventname'], inplace=True)

    assert(events_partition.shape[0] == mh_scores_filtered.shape[0]) #must have same # rows
    bool_df = events_partition[['src_subject_id', 'eventname']] == mh_scores_filtered[['src_subject_id', 'eventname']]
    assert(bool_df.all(axis=0).all(axis=0)) #each row must have same primary key

    return events_partition, mh_scores_filtered


def divide_subjects(site_splits, site_id):
    '''From a splits dictionary with the form {str -> {str -> [str]}}, where the first key is the 
    site id and the second, the split ix, return a dictionary {"train": [str], "id_test": [str], 
    "ood_test": [str]} with three lists of subject ids. 
    
    Parameters:
        site_splits ({str -> {int -> [str]}}): A dictionary linking each site ID to dict linking a 
            split ID to a subject ID list
        site_id (str): The index of this fold. site_id == site_1 means that site 1 is leaft for OOD 
            testing. The last fold is always used for ID testing (except for the site respective of 
            site_id, of course, which belongs to the OOD testing set).
    Returns:
        subject_division ({str -> [str]}): Three lists of subjects with the form 
            {"train": [str], "id_test": [str], "ood_test": [str]}
    '''
    subject_division = {"train": [], "id_test": [], "ood_test": []}
    for splits_site_id, site_folds in site_splits.items():
        k = len(site_folds)
        for fold_id, fold_subjects in site_folds.items():
            if splits_site_id == site_id:
                subject_division["ood_test"] += fold_subjects
            elif fold_id == str(k-1):
                subject_division["id_test"] += fold_subjects
            else:
                subject_division["train"] += fold_subjects
    return subject_division

