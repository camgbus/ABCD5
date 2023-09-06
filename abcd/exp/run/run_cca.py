import os
import argparse
from abcd.exp.Experiment import Experiment
from abcd.utils.communication.TelegramBot import TelegramBot
import abcd.exp.config_utils as cu
from abcd.utils.io import load_json
from abcd.local.paths import output_path

# TODO Adapt
from abcd.analysis.cca import cca
configs_path = os.path.join(output_path, 'configs', 'cca')

def main():
    '''Classifier training or evaluation.'''
    # Parse device and gpu to communicate to bot
    parser = argparse.ArgumentParser()
    parser.add_argument("-hn", "--hardware_name", action='store', type=str, default='')
    parser.add_argument("-sc", "--starting_config", action='store', type=str, default='')
    parser.add_argument("-d", "--debugging", action='store_true')
    args = parser.parse_args()
    hardware_name = args.hardware_name
    starting_config = args.starting_config
    debugging = args.debugging
    state = "FAILED" if debugging else "READY"

    # Initialize bot
    bot = TelegramBot()

    # While there are available configurations
    if args.starting_config == '':
        config_name = cu.choose_next_config(configs_path, state)
    else:
        config_name = f'[{state}]{starting_config}.json'
    
    failures = 0
    while config_name is not None:  
        config = load_json(path=configs_path, file_name=config_name)
        if debugging:
            exp = Experiment(name=config['exp_name'], config=config, reload_exp=True, debugging=True)
            cca(exp)
            config_name = cu.rename_config(configs_path, config_name, 'FAILED', 'TRAINING')
        else:
            # Initialize experiment
            config_name = cu.rename_config(configs_path, config_name, state, 'TRAINING')      
            exp = Experiment(name=config['exp_name'], config=config)
            bot.send_msg(f'Starting {exp.name} in {hardware_name}')
            try:    
                # Actually run. TODO adapt
                cca(exp)
                exp.finish()
            except Exception:
                exp.finish(failed=True)
                failures += 1
                cu.rename_config(configs_path, config_name, 'TRAINING', 'FAILED')
        
            bot.send_msg(f'Finished {exp.name} in {hardware_name} with {exp.summary["state"]}')

        # Break to prevent chain failures
        if failures > 0:
            config_name = None
        else:
            # Free experiments that were dependent on the current one
            cu.free_dependencies(configs_path, config.get('frees'))
            cu.rename_config(configs_path, config_name, 'TRAINING', 'DONE')
            config_name = cu.choose_next_config(configs_path)

    if failures > 0:
        print(f'Finishing in {hardware_name} due to failures.')
        bot.send_msg(f'Finishing in {hardware_name} due to failures.')
    else:
        print(f'Finishing in {hardware_name} (no more configs).')
        bot.send_msg(f'Finishing in {hardware_name} (no more configs).') 

if __name__ == "__main__":
    main()