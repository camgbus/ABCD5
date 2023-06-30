"""Use previously defined split values to divide a dataframe of events into training, ID validation
and OOD testing.
"""

def divide_events_by_splits(events_df, site_splits, site_id):
    '''Divide a df of events into three for training, id id testing and OOD testing, based on the 
    src_subject_id'''
    subject_division = divide_subjects(site_splits, site_id)
    events_train = events_df[events_df['src_subject_id'].isin(subject_division["train"])]
    events_id_test = events_df[events_df['src_subject_id'].isin(subject_division["id_test"])]
    events_ood_test = events_df[events_df['src_subject_id'].isin(subject_division["ood_test"])]
    assert len(events_train) + len(events_id_test) + len(events_ood_test) == len(events_df)
    return events_train, events_id_test, events_ood_test
    
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

