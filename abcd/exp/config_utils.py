import os
import abcd.utils.io as io

def rename_config(configs_path, config_name, prev_state, new_state):
    new_name = config_name.replace('['+prev_state+']', '['+new_state+']')
    os.rename(os.path.join(configs_path, config_name), 
        os.path.join(configs_path, new_name))
    return new_name

def choose_next_config(configs_path, state="READY"):
    '''Return next config that is available, staring with shortest
    '''
    available = sorted([f for f in os.listdir(configs_path) if '[{}]'.format(state) in f], key=lambda x: len(x))
    if len(available)>0:
        config_name = available[0]
        return config_name
    return None

def free_dependencies(configs_path, freed_models=None):
    if freed_models:
        for model in freed_models:
            config_name = '[BLOCKED]'+'_'.join(model) + '.json'
            if os.path.exists(os.path.join(configs_path, config_name)):
                rename_config(configs_path, config_name, 'BLOCKED', 'READY')

def create_config(config, configs_path, config_name, args_positional=[], 
    args_dict=dict(), frees=[], ready=True):
    if 'args_positional' not in config:
        config['args_positional'] = args_positional
    if 'args_dict' not in config:
        config['args_dict'] = args_dict
    if 'frees' not in config:
        config['frees'] = frees
    if ready:
        config_name = '[READY]' + config_name
    else:
        config_name = '[BLOCKED]' + config_name
    io.save_json(config, path=configs_path, name=config_name)