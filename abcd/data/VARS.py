"""Different variables in the ABCD 5.0 release.
"""

import os
from collections import OrderedDict
from itertools import product
from abcd.local.paths import core_path

### --- VAR VALUES --- ###
VALUES = {"kbi_sex_assigned_at_birth": {1: "Male", 2: "Female"}}

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
NIH_COMBINED_uncorrected = {key+"_uncorrected": val for key, val in NIH_COMBINED_SCORES.items()}


### --- CBCL behavior questionnaire --- ###

CBCL_PATH = os.path.join(core_path, "mental-health", "mh_p_cbcl.csv")

CBCL_SCORES_t = {"cbcl_scr_syn_anxdep_t": "Anxious/Dep.",
             "cbcl_scr_syn_withdep_t": "Depression",
             "cbcl_scr_syn_somatic_t": "Somatic",
             "cbcl_scr_syn_social_t": "Social",
             "cbcl_scr_syn_thought_t": "Thought",
             "cbcl_scr_syn_attention_t": "Attention",
             "cbcl_scr_syn_rulebreak_t": "Rule-breaking",
             "cbcl_scr_syn_aggressive_t": "Aggressive",
             "cbcl_scr_syn_internal_t": "Internalizing",
             "cbcl_scr_syn_external_t": "Externalizing"}   

CBCL_SCORES_raw = {"cbcl_scr_syn_anxdep_r": "Anxious/Dep.",
             "cbcl_scr_syn_withdep_r": "Depression",
             "cbcl_scr_syn_somatic_r": "Somatic",
             "cbcl_scr_syn_social_r": "Social",
             "cbcl_scr_syn_thought_r": "Thought",
             "cbcl_scr_syn_attention_r": "Attention",
             "cbcl_scr_syn_rulebreak_r": "Rule-breaking",
             "cbcl_scr_syn_aggressive_r": "Aggressive",
             "cbcl_scr_syn_internal_r": "Internalizing",
             "cbcl_scr_syn_external_r": "Externalizing"}   
CBCL_colors = ('#5D6BBF', '#2E9E99', '#469867', '#CFA38B', '#969696', '#7D5A98', '#A80532', '#AC4436')


### --- Resting state fMRI connectivity scores --- ###

fMRI_PATH = os.path.join(core_path, "imaging", "mri_y_rsfmr_cor_gp_gp.csv")

NETWORKS = OrderedDict([("ad","auditory"),
            ("cgc","cing. opercular"),
            ("ca","cing. parietal"),
            ("dt","default"),
            ("dla","dorsal att."),
            ("fo","fronto parietal"),
            ("n","none"),
            ("rspltp","retrospl. temp."),
            ("sa","salience"),
            ("smh","sensorimr. hand"),
            ("smm","sensorimr. mouth"),
            ("vta","ventral att."),
            ("vs","visual")])

# All connection columns have the shape rsfmri_c_ngd_<Network A>_ngd_<Network B>
CONNECTIONS = ["rsfmri_c_ngd_{}_ngd_{}".format(n1, n2) for (n1, n2) in 
               product(NETWORKS.keys(), NETWORKS.keys())]

NAMED_CONNECTIONS = {"rsfmri_c_ngd_{}_ngd_{}".format(n1, n2): "{}-{}".format(n1, n2) for (n1, n2) in 
               product(NETWORKS.keys(), NETWORKS.keys())}

fMRI_to_subcortical_PATH = os.path.join(core_path, "imaging", "mri_y_rsfmr_cor_gp_aseg.csv")

NETWORK_NAMES_ASAG = OrderedDict([("au","auditory"),
            ("cerc","cingulo-opercular"),
            ("copa","cingulo-parietal"),
            ("df","default"),
            ("dsa","dorsal attention"),
            ("fopa","fronto parietal"),
            ("none","none"),
            ("rst","retrosplenial temporal"),
            ("sa","salience"),
            ("smh","sensorimotor hand"),
            ("smm","sensorimotor mouth"),
            ("vta","ventral attention"),
            ("vs","visual")])

SUBCORTICAL = OrderedDict([("aalh","left accumbens area"),
            ("aarh","right accumbens area"),
            ("aglh"," left amygdala"),
            ("agrh","right amygdala"),
            ("bs","brain stem"),
            ("cdelh","left caudate"),
            ("cderh","right caudate"),
            ("crcxlh"," left cerebellum cortex"),
            ("crcxrh"," right cerebellum cortex"),
            ("hplh","left hippocampus"),
            ("hprh","right hippocampus"),
            ("pllh","left pallidum"),
            ("plrh","right pallidum"),
            ("ptlh","left putamen"),
            ("ptrh","right putamen"),
            ("thplh","left thalamus proper"),
            ("thprh","right thalamus proper"),
            ("vtdclh","left ventraldc"),
            ("vtdcrh","right ventraldc")])

CONNECTIONS_C_SC = {"rsfmri_cor_ngd_{}_scs_{}".format(n1, n2): "{}-{}".format(n1, n2) for (n1, n2) in 
               product(NETWORK_NAMES_ASAG.keys(), SUBCORTICAL.keys())}

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

VALUES["race_ethnicity"] = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
VALUES["demo_comb_income_v2"] = {1: "< $5,000", 2: "$5,000 - $11,999", 3: "$12,000 - $15,999", 4: "$16,000 through $24,999", 5: "$25,000 through $34,999", 6: "$35,000 through $49,999", 7: "$50,000 through $74,999", 8: "$75,000 through $99,999", 9: "$100,000 through $199,999", 10: ">= $200,000"}

### --- Structural MRI features --- ###

DESIKAN_STRUCT_FEATURES = {"smri_thick_cdk": "Cortical thickness",
                "smri_sulc_cdk": "Sulcal depth",
                "smri_area_cdk": "Surface area",
                "smri_vol_cdk": "Volume"#,
                #"smri_t1wgray02_cdk": "T1 intensity - gray matter",
                #"smri_t1ww02_cdk": "T1 intensity - white matter",
                #"smri_t2wg02_cdk": "T2 intensity - gray matter",
                #"smri_t2ww02_cdk": "T2 intensity - white matter"
                }

DESIKAN_STRUCT_FILES = {"smri_thick_cdk": "mri_y_smr_thk_dsk.csv",
                "smri_sulc_cdk": "mri_y_smr_sulc_dsk.csv",
                "smri_area_cdk": "mri_y_smr_area_dsk.csv",
                "smri_vol_cdk": "mri_y_smr_vol_dsk.csv"#,
                #"smri_t1wgray02_cdk": "mri_y_smr_t1_gray_dsk.csv",
                #"smri_t1ww02_cdk": "mri_y_smr_t1_white_dsk.csv",
                #"smri_t2wg02_cdk": "mri_y_smr_t2_gray_dsk.csv",
                #"smri_t2ww02_cdk": "mri_y_smr_t2_white_dsk.csv"
                }

# 68 Desikan (dsk) parcels
# Variable name = name + _parcel   e.g. smri_thick_cdk_banksstslh
DESIKAN_PARCELS_1 = ['banksstslh', 'cdacatelh', 'cdmdfrlh', 'cuneuslh', 'ehinallh', 'fusiformlh', 'ifpllh', 'iftmlh', 'ihcatelh', 'locclh', 'lobfrlh', 'linguallh', 'mobfrlh', 'mdtmlh', 'parahpallh', 'paracnlh', 'parsopclh', 'parsobislh', 'parstgrislh', 'pericclh', 'postcnlh', 'ptcatelh', 'precnlh', 'pclh', 'rracatelh', 'rrmdfrlh', 'sufrlh', 'supllh', 'sutmlh', 'smlh', 'frpolelh', 'tmpolelh', 'trvtmlh', 'insulalh', 'banksstsrh', 'cdacaterh', 'cdmdfrrh', 'cuneusrh', 'ehinalrh', 'fusiformrh', 'ifplrh', 'iftmrh', 'ihcaterh', 'loccrh', 'lobfrrh', 'lingualrh', 'mobfrrh', 'mdtmrh', 'parahpalrh', 'paracnrh', 'parsopcrh', 'parsobisrh', 'parstgrisrh', 'periccrh', 'postcnrh', 'ptcaterh', 'precnrh', 'pcrh', 'rracaterh', 'rrmdfrrh', 'sufrrh', 'suplrh', 'sutmrh', 'smrh', 'frpolerh', 'tmpolerh', 'trvtmrh', 'insularh']
DESIKAN_PARCELS_2 = ['banksstslh', 'cdatcgatelh', 'cdmdflh', 'cuneuslh', 'ehinallh', 'fusiformlh', 'ifpllh', 'iftmlh', 'ihcgatelh', 'ltocclh', 'ltoboflh', 'linguallh', 'moboflh', 'mdtmlh', 'parahpallh', 'paractlh', 'popclh', 'pobalislh', 'parstgslh', 'pericclh', 'postctlh', 'pscgatelh', 'prectlh', 'pnlh', 'rtatcgatelh', 'rtmdflh', 'suflh', 'supllh', 'sutmlh', 'smlh', 'fpolelh', 'tmpolelh', 'tvtmlh', 'insulalh', 'banksstsrh', 'cdatcgaterh', 'cdmdfrh', 'cuneusrh', 'ehinalrh', 'fusiformrh', 'ifplrh', 'iftmrh', 'ihcgaterh', 'ltoccrh', 'ltobofrh', 'lingualrh', 'mobofrh', 'mdtmrh', 'parahpalrh', 'paractrh', 'popcrh', 'pobalisrh', 'parstgsrh', 'periccrh', 'postctrh', 'pscgaterh', 'prectrh', 'pnrh', 'rtatcgaterh', 'rtmdfrh', 'sufrh', 'suplrh', 'sutmrh', 'smrh', 'fpolerh', 'tmpolerh', 'tvtmrh', 'insularh']

DESIKAN_PARCELS = {"smri_thick_cdk": DESIKAN_PARCELS_1,
                "smri_sulc_cdk": DESIKAN_PARCELS_1,
                "smri_area_cdk": DESIKAN_PARCELS_1,
                "smri_vol_cdk": DESIKAN_PARCELS_1,
                "smri_t1wgray02_cdk": DESIKAN_PARCELS_1,
                "smri_t1ww02_cdk": DESIKAN_PARCELS_1,
                "smri_t2wg02_cdk": DESIKAN_PARCELS_2,
                "smri_t2ww02_cdk": DESIKAN_PARCELS_2}

DESIKAN_MEANS = ['meanlh', 'meanrh', 'mean']
DESIKAN_TOTALS = ['totallh', 'totalrh', 'total']


### --- Sleep --- ###

SLEEP = {'ksads_sleepprob_raw_814_p': 'Trouble falling asleep in the past 2 weeks (P)',
               'ksads_sleepprob_raw_816_p': 'Ever had trouble falling asleep > 2 weeks (P)',
               'ksads_sleepprob_raw_814_t': 'Trouble falling asleep in the past 2 weeks (C)',
               'ksads_sleepprob_raw_816_t': 'Ever had trouble falling asleep > 2 weeks (C)',
               'fit_ss_sleepperiod_minutes': 'Fitbit total sleep minutes (D)',
               'fit_ss_sleep_period_minutes': 'Fitbit total sleep minutes (W)',
               'sleepdisturb1_p': 'Hours of sleep most nights past 6 m (P)',
               'sleepdisturb2_p': 'Time until falling asleep past 6 m (P)',
               'mctq_sd_min_to_sleep': 'Minutes needed to fall asleep (C)'
               }

VALUES['ksads_sleepprob_raw_814_p'] = VALUES['ksads_sleepprob_raw_814_t'] = {0: "Not at all", 1: "Rarely", 2: "Several days", 3: "More than half the days", 4: "Nearly every day"}
VALUES['ksads_sleepprob_raw_816_p'] = VALUES['ksads_sleepprob_raw_816_t'] = {0: "No", 1: "Yes"}
VALUES['sleepdisturb1_p'] = {1: "9-11 hs", 2: "8-9 hs", 3: "7-8 hs", 4: "5-7 hs", 5: "< 5 hs"}
VALUES['sleepdisturb2_p'] = {1: "< 15", 2: "15-30", 3: "30-45", 4: "45-60", 5: "> 60"}                         
VALUES['mctq_sd_min_to_sleep'] = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "15", 12: "20", 13: "25", 14: "30", 15: "40", 16: "50", 17: "60", 18: "75", 19: "90", 20: "105", 21: "120", 22: "180", 23: "240"}                     
                             
SLEEP_PATHS = {'ksads_sleepprob_raw_814_p': os.path.join(core_path, 'mental-health', 'mh_p_ksads_slp.csv'),
                     'ksads_sleepprob_raw_816_p': os.path.join(core_path, 'mental-health', 'mh_p_ksads_slp.csv'),
                     'ksads_sleepprob_raw_814_t': os.path.join(core_path, 'mental-health', 'mh_y_ksads_slp.csv'),
                     'ksads_sleepprob_raw_816_t': os.path.join(core_path, 'mental-health', 'mh_y_ksads_slp.csv'),
                     'fit_ss_sleepperiod_minutes': os.path.join(core_path, 'novel-technologies', 'nt_y_fitb_slp_d.csv'),
                     'fit_ss_sleep_period_minutes': os.path.join(core_path, 'novel-technologies', 'nt_y_fitb_slp_w.csv'),
                     'sleepdisturb1_p': os.path.join(core_path, 'physical-health', 'ph_p_sds.csv'),
                     'sleepdisturb2_p': os.path.join(core_path, 'physical-health', 'ph_p_sds.csv'),
                     'mctq_sd_min_to_sleep': os.path.join(core_path, 'physical-health', 'ph_y_mctq.csv')
                     }