"""Different variables in the ABCD 5.0 release.
"""

import os
from collections import OrderedDict
from itertools import product
from abcd.local.paths import core_path

### --- NIH Toolbox neurocognition --- ###
NIH_PATH = os.path.join(core_path, "neurocognition", "nc_y_nihtb.csv")

NIH_TESTS = {"nihtbx_cardsort": "Dimensional Change Card Sort",
             "nihtbx_flanker": "Flanker Inhibitory Control and Attention",
             "nihtbx_picture": "Picture Sequence Memory",
             "nihtbx_list": "List Sorting Working Memory",
             "nihtbx_pattern": "Pattern Comparison Processing Speed",
             "nihtbx_picvocab": "Picture Vocabulary",
             "nihtbx_reading": "Oral Reading Recognition"}
    
# Cognition Fluid Composite: Includes Dimensional Change Card Sort, Flanker Inhibitory Control and 
# Attention, Picture Sequence Memory, List Sorting Working Memory, and Pattern Comparison tests
# Crystallized Composite: Includes Picture Vocabulary and Oral Reading Recognition tests
NIH_COMBINED_SCORES = {"nihtbx_fluidcomp": "Cognition Fluid Composite",
                       "nihtbx_cryst": "Crystallized Composite",
                       "nihtbx_totalcomp": "Cognition Total Composite"}

# Only the raw scores have a different solumn name sometimes ('_rs' for picvocab and fluidcomp or 
# '`_rawscore' otherwise), but tese have more often missing values than the 'uncorrected' ones
NIH_VARIATIONS = {"_uncorrected": "Uncorrected Standard Score",
                  "_agecorrected": "Age-Corrected Standard Score", 
                  "_rawscore": "Raw Score", 
                  "_theta": "Theta", 
                  "_itmcnt": "ItmCnt", 
                  "_cs": "Computed Score",
                  "_fc": "Fully-Corrected T-score"}

NIH_TESTS_uncorrected = {key+"_uncorrected": val for key, val in NIH_TESTS.items()}

NIH_COMBINED_SCORES_uncorrected = {key+"_uncorrected": val for key, val in NIH_COMBINED_SCORES.items()}


### --- CBCL behavior questionnaire --- ###

CBCL_PATH = os.path.join(core_path, "mental-health", "mh_p_cbcl.csv")

CBCL_SCORES_t = {"cbcl_scr_syn_anxdep_t": "Anxious/Dep.",
             "cbcl_scr_syn_withdep_t": "Depression",
             "cbcl_scr_syn_somatic_t": "Somatic",
             "cbcl_scr_syn_social_t": "Social",
             "cbcl_scr_syn_attention_t": "Attention",
             "cbcl_scr_syn_rulebreak_t": "Rule-breaking",
             "cbcl_scr_syn_aggressive_t": "Aggressive",
             "cbcl_scr_syn_internal_t": "Internalizing",
             "cbcl_scr_syn_external_t": "Externalizing"}   

CBCL_SCORES_raw = {"cbcl_scr_syn_anxdep_r": "Anxious/Dep.",
             "cbcl_scr_syn_withdep_r": "Depression",
             "cbcl_scr_syn_somatic_r": "Somatic",
             "cbcl_scr_syn_social_r": "Social",
             "cbcl_scr_syn_attention_r": "Attention",
             "cbcl_scr_syn_rulebreak_r": "Rule-breaking",
             "cbcl_scr_syn_aggressive_r": "Aggressive",
             "cbcl_scr_syn_internal_r": "Internalizing",
             "cbcl_scr_syn_external_r": "Externalizing"}   


### --- Resting state fMRI connectivity scores --- ###

fMRI_PATH = os.path.join(core_path, "imaging", "mri_y_rsfmr_cor_gp_gp.csv")

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

NAMED_CONNECTIONS = {"rsfmri_c_ngd_{}_ngd_{}".format(n1, n2): "{}-{}".format(n1, n2) for (n1, n2) in 
               product(NETWORKS.keys(), NETWORKS.keys())}

### --- Environment variables --- ###

DEMO_PATH = os.path.join(core_path, "abcd-general", "abcd_p_demo.csv")

# Race/Ethn.: 1 = White; 2 = Black; 3 = Hispanic; 4 = Asian; 5 = Other
# Combined family income (very similar variables)
# 1 = Less than $5,000 ; 2 = $5,000 through $11,999 ; 3 = $12,000 through $15,999 ; 4 = $16,000 through $24,999 ; 5 = $25,000 through $34,999 ; 6 = $35,000 through $49,999 ; 7 = $50,000 through $74,999 ; 8 = $75,000 through $99,999 ; 9 = $100,000 through $199,999 ; 10 = $200,000 and greater ; 999, Don't know ; 777, Refuse to answer	
# Highest level of education: 0 (never) to 21 (PhD)
DEMO_VARS = {"race_ethnicity": "Race/Ethn.",
            "demo_comb_income_v2": "Combined family income 1"#,
            # "demo_comb_income_v2_l": "Combined family income 2", # Reduced dataframe to about half
            # "demo_prnt_ed_v2_2yr_l": "Education parent 1", # Reduced dataframe significantly
            # "demo_prtnr_ed_v2_2yr_l": "Education parent 2" # Reduced dataframe significantly
            }

### --- Structural MRI features --- ###

STRUCT_FEATURES = {"smri_thick_cdk_mean": "Mean cortical thickness",
                "smri_sulc_cdk_mean": "Mean cortical sulcal depth",
                "smri_area_cdk_total": "Total hemisphere area",
                "smri_vol_cdk_total": "Whole brain cortical volume",
                "smri_t1wgray02_cdk_mean": "Average T1 intensity of gray matter",
                "smri_t1ww02_cdk_mean": "Average T1 intensity of white matter",
                "smri_t2wg02_cdk_mean": "Average T2 intensity of gray matter",
                "smri_t2ww02_cdk_mean": "Average T2 intensity of white matter"}

STRUCT_FILES = {"smri_thick_cdk_mean": "mri_y_smr_thk_dsk.csv",
                "smri_sulc_cdk_mean": "mri_y_smr_sulc_dsk.csv",
                "smri_area_cdk_total": "mri_y_smr_area_dsk.csv",
                "smri_vol_cdk_total": "mri_y_smr_vol_dsk.csv",
                "smri_t1wgray02_cdk_mean": "mri_y_smr_t1_gray_dsk.csv",
                "smri_t1ww02_cdk_mean": "mri_y_smr_t1_white_dsk.csv",
                "smri_t2wg02_cdk_mean": "mri_y_smr_t2_gray_dsk.csv",
                "smri_t2ww02_cdk_mean": "mri_y_smr_t2_white_dsk.csv"}