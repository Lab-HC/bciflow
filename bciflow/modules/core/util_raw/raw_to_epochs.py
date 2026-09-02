"""Function to convert raw data to epochs."""

import numpy as np

def raw_to_epochs(
        raw_data,
        window_size=2,
        step_size=0.5
    ):
    """Convert raw data to epochs.
    
    Parameters
    ----------
    raw_data : dict
        Raw data in the format of bciflow.
    window_size : float, optional
        Size of the window in seconds, by default 2.
    step_size : float, optional
        Step size in seconds, by default 0.5.
        
    Returns
    -------
    dict
        Data in the format of bciflow, but with epochs instead of raw data.
    """

    window_size = int(window_size * raw_data['sfreq'])
    step_size = int(step_size * raw_data['sfreq'])

    x_epochs, y_epochs = [], []

    for trial in range(0, raw_data['X'].shape[1]-window_size, step_size):
        x_epochs.append(raw_data['X'][:, trial:trial+window_size])
        y_epochs.append(raw_data['y'][trial+window_size-1])

    data_epochs = raw_data.copy()
    data_epochs['X'] = np.array(x_epochs)
    data_epochs['y'] = np.array(y_epochs)
    data_epochs['data_type'] = "epochs"

    return data_epochs
