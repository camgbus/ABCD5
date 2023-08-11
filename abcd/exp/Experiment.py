"""Experiment class that tracks experiments. A directory is created for the experiment. There,
a batch script and configuration can be stored, along with intermediate and results files.
Once the Experiment is finished, metadata such as duration and error stack traces are stored. The
output printed to the console is likewise stored.
"""

import os
import sys
import io
import traceback
from abcd.utils.helper_functions import get_time, get_time_string
from abcd.utils.io import load_json, dump_json
from abcd.local.paths import output_path

class Experiment:
    '''An Experiment that creates a directory with its name and stores a summary.json files.
    Parameters:
        name (str): experiment name. If empty, a datestring is set as name.
        config (dict): a dictionary with parameters
        notes (str): optional notes about the experiment
        reload_exp (bool): Reload or throw error when experiment name exists?
    '''
    def __init__(self, name=None, config={}, notes='', reload_exp=False, debugging=False):
        self.start_time = get_time()
        if not debugging:
            self.old_stdout = sys.stdout
            sys.stdout = self.stdout = io.StringIO()
        # Store or build name
        if name is None:
            self.name = get_time_string(self.start_time)
        self.name = name
        self.path = os.path.join(output_path, "exp", self.name)
        if reload_exp:
            assert os.path.exists(self.path), "Experiment {} not found".format(self.path)
            self.config = load_json(path=self.path, file_name='config')
            self.summary = load_json(path=self.path, file_name='summary')
            # When reloading, original notes are saved
            notes = self.summary['notes']
        else:
            # Make sure experiment does not already exist
            assert not os.path.exists(self.path), "Experiment {} already exists".format(self.path)
            os.makedirs(self.path)
            self.config = config
            dump_json(self.config, path=self.path, file_name='config')
        self.summary = {'start_time': get_time_string(self.start_time), 'notes': notes}

    def finish(self, failed=False, notes=''):
        elapsed_time = get_time() - self.start_time
        if failed:
            tb = traceback.format_exc()
            self.summary['state'] = 'FAILURE'
            with open(os.path.join(self.path, 'traceback.txt'), 'w') as f:
                f.write(tb)
        else:
            self.summary['state'] = 'SUCCESS'
        self.summary['elapsed_time'] = '{0:.2f} min'.format(elapsed_time.total_seconds()/60)
        self.summary['notes'] = notes
        sys.stdout = self.old_stdout
        with open(os.path.join(self.path, "stdout.txt"), 'w') as f:
            f.write(self.stdout.getvalue())
        dump_json(self.summary, path=self.path, file_name='summary')