

def assert_same_state_dicts(model_1, model_2, max_error = 0):
    '''Check that 2 models have the same state dictionary'''
    for (param_1, param_2) in zip(model_1.parameters(), model_2.parameters()):
        if param_1.data.ne(param_2.data).sum() > max_error:
            return False
    return True