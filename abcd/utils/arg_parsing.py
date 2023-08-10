"""
Simulates console arguments
"""

import sys

def set_console_args(args_positional=[], args_dict={}):
    args = [sys.argv[0]] + [str(x) for x in args_positional]
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                args.append('--{}'.format(key))
        elif isinstance(value, list):
            args += ['-{}'.format(key)] + [str(x) for x in value]
        else:
            args += ['-{}'.format(key), str(value)]
    sys.argv = args