from abcd.data.read_data import get_subjects_events
from abcd.data.define_splits import inter_site_splits, save_restore_sex_fmri_splits

def test_splits():
    subjects_df, _ = get_subjects_events()
    splits = inter_site_splits(subjects_df, k=5)

    # Ensure that no subject is repeated
    all_subjects = []
    for site_splits in splits.values():
        for split in site_splits.values():
            all_subjects += split
    assert len(all_subjects) == len(set(all_subjects)) == 9879 

    # Ensure that no two subjects from the same family are in the same split
    family_splits = dict()
    for site_id, site_splits in splits.items():
        for split_id, split in site_splits.items():
            site_split_id = site_id + str(split_id)
            for subject in split:
                family = subjects_df.loc[subjects_df["src_subject_id"] == subject]["rel_family_id"]
                family = list(family)[0]
                if family not in family_splits:
                    family_splits[family] = site_split_id
                else:
                    assert family_splits[family] == site_split_id
    assert len(family_splits) == 8233
    
def test_sex_fmri_splits():
    splits = save_restore_sex_fmri_splits(k=5)
    assert splits["site01"]["0"][:3] == ['NDAR_INVHPVL425U', 'NDAR_INVMYUP1YJ7', 'NDAR_INVE0KZKF5V']