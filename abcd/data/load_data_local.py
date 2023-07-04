from abcd.utils.io import *
from abcd.local.paths import output_path
from abcd.data.divide_with_splits import divide_events_by_splits, divide_subjects


def load_data_local(site_splits, mh_scores_df, events_df):
    '''
    Create a local folder called model_data within the output path defined in paths.py. 
    Create 21 folders within it called model01, model01..., model 21.
    Populate each of these 21 folders with the train/id test/ood test data. model01 corresponds to the 
    data partitioning which uses site01 as the ood test data.

    If model_data already exists, the function returns without modifying or creating any folders/files.

    Parameters:
        site_splits: #{key='site#' --> value={key=fold# --> value= list of subejct IDs}}
        mh_scores_df: pd dataframe
        events_df: pd dataframe
    Returns:
        all_data: {model description (string) --> {data partition (string) --> pandas df}}
            - model descriptions are 'model01', 'model02',..., 'model21'
            - data partitions are 'events_train', 'events_id_test', 'events_ood_test', 'mh_scores_train', 
            'mh_scores_id_test', 'mh_scores_ood_test'
    '''

    #create model_data folder
    all_data = {}
    folder_path = os.path.join(output_path, "model_data")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Created model_data directory")
    else :
        print("model_data directory already exists. Did not perform data population.")
        return

    #create model01 ... model21 folders, populate with train/test data
    for site_name in site_splits.keys():
        model_desc = "model" + site_name[-2:]
        model_folder_path = os.path.join(folder_path, model_desc)

        #create folder
        if os.path.exists(model_folder_path):
            continue
        os.makedirs(model_folder_path)

        #get train and test data (site corresponding to site_name is left for ood testing)
        events_train, events_id_test, events_ood_test, mh_scores_train, mh_scores_id_test, mh_scores_ood_test =  divide_events_by_splits(events_df, site_splits, site_name, mh_scores_df) #all pandas dataframes
        data_dict = {
            'events_train': events_train, 
            'events_id_test': events_id_test, 
            'events_ood_test':events_ood_test, 
            'mh_scores_train': mh_scores_train,
            'mh_scores_id_test': mh_scores_id_test,
            'mh_scores_ood_test': mh_scores_ood_test
        }
        all_data[model_desc] = data_dict

        #put train and test data in model folder
        for key in data_dict.keys():
            file_path = os.path.join(model_folder_path, key + '_' + site_name[-2:] + '.csv')

            dump_df(data_dict[key], file_path)

    return all_data