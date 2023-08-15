import os
import argparse
from abcd.exp.Experiment import Experiment
from abcd.utils.communication.TelegramBot import TelegramBot
import abcd.exp.config_utils as cu
from abcd.utils.io import load_json
from abcd.local.paths import output_path

# TODO Adapt
from abcd.analysis.classification import classification
configs_path = os.path.join(output_path, 'configs', 'classification', 'eval')
eval_config = {"states": ["best"], "shap": True}

def main():
    # Parse device and gpu to communicate to bot
    parser = argparse.ArgumentParser()
    parser.add_argument("-hn", "--hardware_name", action='store', type=str, default='')
    parser.add_argument("-sc", "--starting_config", action='store', type=str, default='')
    parser.add_argument("-d", "--debugging", action='store_true')
    args = parser.parse_args()
    hardware_name = args.hardware_name
    starting_config = args.starting_config
    debugging = args.debugging
    state = "FAILED" if debugging else "DONE"

    # Initialize bot
    bot = TelegramBot()

    # While there are available configurations
    if args.starting_config == '':
        config_name = cu.choose_next_config(configs_path, state)
    else:
        config_name = '[{}]{}.json'.format(state, starting_config)
    
    failures = 0
    while config_name is not None:  
        config = load_json(path=configs_path, file_name=config_name)
        if debugging:
            exp = Experiment(name=config['exp_name'], reload_exp=True, debugging=True)
            classification(exp, eval_config=eval_config)
        else:
            # Initialize experiment
            config_name = cu.rename_config(configs_path, config_name, state, 'EVALUATING')      
            exp = Experiment(name=config['exp_name'], reload_exp=True)
            bot.send_msg('Starting eval {} in {}'.format(exp.name, hardware_name))
            try:    
                # Actually run. TODO adapt
                classification(exp, eval_config=eval_config)
                exp.finish()
            except:
                exp.finish(failed=True)
                failures += 1
                cu.rename_config(configs_path, config_name, 'EVALUATING', 'FAILED')
        
        bot.send_msg('Finished eval {} in {} with {}'.format(exp.name, hardware_name, exp.summary['state']))

        # Break to prevent chain failures
        if failures > 0:
            config_name = None
        else:
            # Free experiments that were dependent on the current one
            cu.free_dependencies(configs_path, config.get('frees'))
            cu.rename_config(configs_path, config_name, 'EVALUATING', 'EVAL_DONE')
            config_name = cu.choose_next_config(configs_path, state=state)

    if failures > 0:
        print('Finishing eval in {} due to failures.'.format(hardware_name))
        bot.send_msg('Finishing eval in {} due to failures.'.format(hardware_name))
    else:
        print('Finishing eval in {} (no more configs).'.format(hardware_name))
        bot.send_msg('Finishing eval in {} (no more configs).'.format(hardware_name)) 

if __name__ == "__main__":
    main()