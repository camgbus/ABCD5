import os
import argparse
from abcd.exp.Experiment import Experiment
from abcd.utils.communication.TelegramBot import TelegramBot
import abcd.exp.config_utils as cu
from abcd.utils.io import load_json
from abcd.local.paths import output_path

# TODO Adapt
from abcd.analysis.classification import train_classifier
configs_path = os.path.join(output_path, 'configs', 'classification')

def main():
    # Parse device and gpu to communicate to bot
    parser = argparse.ArgumentParser()
    parser.add_argument("-hn", "--hardware_name", action='store', type=str, default='')
    parser.add_argument("-sc", "--starting_config", action='store', type=str, default='')
    args = parser.parse_args()
    hardware_name = args.hardware_name
    starting_config = args.starting_config

    # Initialize bot
    bot = TelegramBot()

    # While there are available configurations
    if args.starting_config == '':
        config_name = cu.choose_next_config(configs_path)
    else:
        config_name = '[READY]'+ starting_config + '.json'
    config_name = cu.rename_config(configs_path, config_name, 'READY', 'TRAINING')

    failures = 0
    while config_name is not None:        
        config = load_json(path=configs_path, file_name=config_name)

        # TODO Adapt
        exp_name = config['exp_name']

        # Initialize experiment
        exp = Experiment(name=exp_name, config=config)
        bot.send_msg('Starting {}'.format(exp.name))
        try:    # Actually run
            train_classifier(exp)
            exp.finish()
        except:
            exp.finish(failed=True)
            failures += 1
            cu.rename_config(configs_path, config_name, 'TRAINING', 'FAILED')
        
        bot.send_msg('Finished {} with {}'.format(exp.name, exp.summary['state']))

        # Break to prevent chain failures
        if failures > 0:
            config_name = None
        else:
            # Free experiments that were dependent on the current one
            cu.free_dependencies(configs_path, config.get('frees'))
            cu.rename_config(configs_path, config_name, 'TRAINING', 'DONE')
            config_name = cu.choose_next_config(configs_path)

    if failures > 0:
        print('Finishing in {} due to failures.'.format(hardware_name))
        bot.send_msg('Finishing in {} due to failures.'.format(hardware_name))
    else:
        print('Finishing in {} (no more configs).'.format(hardware_name))
        bot.send_msg('Finishing in {} (no more configs).'.format(hardware_name)) 

if __name__ == "__main__":
    main()