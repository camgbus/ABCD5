"""Verious IO operations.
"""

import os
import json
import pandas as pd
import collections

def _get_full_path(path, file_name=None, ending='.txt'):
    if file_name:
        if ending not in file_name:
            file_name = file_name+ending
        path = os.path.join(path, file_name)
    return path

def dump_df(df, path, file_name=None, sep=','):
    ''''Stores a Pandas df as csv file'''
    path = _get_full_path(path, file_name, ending='.csv')
    df.to_csv(path, sep=sep, index=False)
    
def load_df(path, file_name=None, sep=',', cols=None):
    ''''Loads a df into a csv file'''
    path = _get_full_path(path, file_name, ending='.csv')
    df = pd.read_csv(path, sep=sep, usecols=cols)
    assert df is not None
    return df

def dump_json(obj, path, file_name=None):
    '''Stores an object as json file'''
    path = _get_full_path(path, file_name, ending='.json')
    # If keys are a tuple, divide with |||
    if isinstance(obj, dict) and isinstance(next(iter(obj.keys())), tuple):
        obj = collections.OrderedDict([("|||".join(key), value) for key, value in obj.items()])
    with open(path, 'w') as f:
        json.dump(obj, f)
        
def load_json(path, file_name=None):
    '''Loads a json file as an OrderedDict'''
    path = _get_full_path(path, file_name, ending='.json') 
    with open(path, 'r') as f:
        obj = json.load(f, object_pairs_hook=collections.OrderedDict)
    if isinstance(obj, dict) and "|||" in next(iter(obj.keys())):
        another_obj = collections.OrderedDict()
        obj = collections.OrderedDict([(tuple(key.split("|||")), value) for key, value in obj.items()])
        obj = another_obj
    return obj
        
def dump_json_obj_dict(obj, path, file_name=None):
    '''Stores a dictionary where the values are objects of a certain class as a json'''
    parsed_objs = collections.OrderedDict()
    for key, value in obj.items():
        parsed_objs[key] = json.dumps(value.__dict__)
    dump_json(parsed_objs, path, file_name)
        
def load_json_obj_dict(class_def, path, file_name=None):
    '''Loads a disctionary where the values are objects of a certain class back from a json'''  
    obj = load_json(path, file_name)
    parsed_obj = collections.OrderedDict()
    for key, value in obj.items():
        obj_str = json.loads(value)
        parsed_obj[key] = class_def(**obj_str)
    return parsed_obj
    