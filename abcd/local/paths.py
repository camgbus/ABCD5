"""See a description of all variables and where to find them on https://data-dict.abcdstudy.org/.
"""

import os

# TODO define paths for the ABCD 5.0 release data and output files
data_path = r""
output_path = r""

core_path = os.path.join(data_path, 'core')
core_dirs = ["abcd-general", 
            "culture-environment", 
            "gender-identity-sexual-health", 
            "genetics", 
            "imaging", 
            "linked-external-data", 
            "mental-health", 
            "neurocognition", 
            "novel-technologies",
            "physical-health",
            "substance-use"]

abcd_y_lt = os.path.join(core_path, "abcd-general", "abcd_y_lt")

