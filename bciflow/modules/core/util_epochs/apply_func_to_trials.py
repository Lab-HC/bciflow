"""Apply a function to each trial in the data."""

import inspect
import numpy as np

from .get_trials import get_trials

def apply_func_to_trials(data, func, func_param, new_data_values):
    """Apply a function to each trial in the data.

    Parameters
    ----------
    data : dict
        A dictionary containing the data to be processed. It must have
        the keys 'X' and 'y'.
    func : function or object with a transform method
        The function to be applied to each trial. If it is an object
        with a transform method, the transform method will be used.
    func_param : dict
        A dictionary containing the parameters to be passed to the
        function. The keys of the dictionary are the names of the
        parameters, and the values are the values of the parameters.
    new_data_values : dict
        A dictionary containing the new values to be added to the
        data. The keys of the dictionary are the names of the new
        keys to be added to the data, and the values are the values
        of the new keys. The new keys will be added to the data
        after the function is applied to each trial.

    Returns
    -------
    dict
        A dictionary containing the processed data. It has the same
        keys as the input data, plus the new keys specified in
        new_data_values.
    """

    new_x = []
    for _trial in range(data['X'].shape[0]):
        if inspect.isfunction(func):

            data_copy = get_trials(data, [_trial])
            new_x.append(func(data_copy, **func_param)['X'])
        else:
            print(func.__name__)
            new_x.append(func.transform(data_copy, **func_param)['X'])
    x_copy = np.array(new_x)

    if x_copy.shape[1] == 1:
        x_copy = x_copy[:, 0, ...]

    data['X'] = x_copy

    for key, value in new_data_values.items():
        data[key] = value

    return data
