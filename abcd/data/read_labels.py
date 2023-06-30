"""Read mental health raw scores data, combining several variables from the ABCD 5.0 release.
"""

import os
import pandas as pd
from tqdm import tqdm
import abcd.utils.io as io
from abcd.local.paths import core_path, output_path
from abcd.data.NETWORKS import CONNECTIONS

def get_mh_labels():
    '''Fetches mental health raw scores (no filtering by sex or # sites the subject has been to).

    Returns:
        mh_scores_df (pandas.DataFrame): mental health scores dataframe with src_subject_id, eventname, and raw 
        mental health cbcl scores.
    '''

    try: #data has already been read
        mh_raw_scores_df = io.load_df(output_path, "mh_raw_scores")
    except: #read data for first time
        mh_raw_scores_df = read_mh_scores()
    return mh_raw_scores_df


def read_mh_scores(output_path=output_path):
    '''Reads mental health raw scores from ABCD 5.0 database.

    Returns:
        raw_scores_df (pandas.DataFrame): mental health scores dataframe with src_subject_id, eventname, and raw 
        mental health cbcl scores.
    '''

    mh_labels_file = os.path.join(core_path, "mental-health", "mh_p_cbcl.csv")
    scores_df = io.load_df(mh_labels_file, sep =',')

    # Select only desired columns: subject id, event name, and the raw scores
    selected_columns = ['src_subject_id', 'eventname'] + [col for col in scores_df.columns if col.endswith('_r')]
    raw_scores_df = scores_df[selected_columns]


    #create csv file
    if output_path:
        io.dump_df(raw_scores_df, output_path, "mh_raw_scores")

    return raw_scores_df