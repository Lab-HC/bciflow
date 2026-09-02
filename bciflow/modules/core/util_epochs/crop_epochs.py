"""Function to crop epochs to a specific time window."""

import numpy as np

def crop_epochs(data, tmin, window_size):
    """Crop epochs to a specific time window.
    
    Parameters
    ----------
    
    data : dict
        A dictionary containing the data to be cropped. It must have
        the keys 'X', 'tmin', and 'sfreq'.
    tmin : float
        The time (in seconds) of the start of the cropped window. It
        must be greater than or equal to the tmin of the original data.
    window_size : float
        The size (in seconds) of the cropped window.
        
    Returns
    -------
    dict
        A dictionary containing the cropped data. It has the keys 'X' and 'tmin'.    
    """

    x = data['X'].copy()
    x = x.reshape((np.prod(x.shape[:-1]), x.shape[-1]))

    indice = int((tmin - data["tmin"]) * data["sfreq"])
    max_indice = indice + int(window_size * data["sfreq"])
    if np.any(indice + int(window_size * data["sfreq"]) > x.shape[-1]):
        raise ValueError("tmin + window_size must be less than or equal " \
        "to the tmax of the original data")

    x = x[:, indice:max_indice]
    x = x.reshape((*data['X'].shape[:-1], max_indice - indice))

    data["X"] = x
    data['tmin'] = tmin

    return data
