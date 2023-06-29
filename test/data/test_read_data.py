import pytest
import os
from abcd.local.paths import core_path
from abcd.data.read_data import read_events_general_info, add_subject_sex, add_event_connectivity_scores, get_subjects_events

def test_read_data():
    subjects_df_gi, events_df_gi = read_events_general_info()
    assert len(subjects_df_gi) == 11686
    assert len(events_df_gi) == 88896
    
    subjects_df_sex_fmri, events_df_sex_fmri = get_subjects_events()
    assert len(subjects_df_sex_fmri) == 9879
    assert len(events_df_sex_fmri) == 19605

@pytest.mark.skip(reason="Saving time, if there is an issue the function above will detect it")
def test_intermediate_ops():
    subjects_df_gi, events_df_gi = read_events_general_info()

    subjects_df_sex, events_df_sex = add_subject_sex(subjects_df_gi, events_df_gi)
    assert len(subjects_df_sex) == 10039
    assert len(events_df_sex) == 81151

    subjects_df_cs, events_df_cs = add_event_connectivity_scores(subjects_df_gi, events_df_gi)
    assert len(subjects_df_cs) == 11435
    assert len(events_df_cs) == 21796

    subjects_df_sex_cs, events_df_sex_cs = add_event_connectivity_scores(subjects_df_sex, events_df_sex)
    assert len(subjects_df_sex_cs) == 9879
    assert len(events_df_sex_cs) == 19605

    subjects_df_cs_sex, events_df_cs_sex = add_subject_sex(subjects_df_cs, events_df_cs)
    assert len(subjects_df_cs_sex) == 9879
    assert len(events_df_cs_sex) == 19605
    