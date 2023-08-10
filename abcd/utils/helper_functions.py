"""Miscellaneous helper functions.
"""

import datetime

def f_optional_args(f, args, x):
    '''If there are arguments, these are passed to the function.'''
    if args:
        return f(x, **args)
    else:
        return f(x)

def get_time():
    return datetime.datetime.now()

def get_time_string(cover=False, time=None):
    '''
    Returns the current time in the format YYYY-MM-DD_HH-MM, or
    [YYYY-MM-DD_HH-MM] if 'cover' is set to 'True'.
    '''
    if time is None:
        time = get_time()
    date = str(time).replace(' ', '_').replace(':', '-').split('.')[0]
    if cover:
        return '['+date+']'
    else:
        return date
