import numpy as np
from scipy.io import loadmat
from typing import Optional, Dict, Any, List


def attention_raw(
    subject: int = 1,
    path: str = 'data/attention/',
) -> Dict[str, Any]:
    """
    Description
    -----------
    This function loads EEG data for a specific subject and session from the attention dataset.
    It processes the data to fit the structure of the `eegdata` dictionary, which is used
    for further processing and analysis.

    The dataset can be found at:
     - https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection

    Parameters
    ----------
    subject : int
        index of the subject to retrieve the data from
    path : str
        Path to the .mat file.

    Returns
    -------
    dict
        Dictionary with:
            X: EEG data as [1, 1, channels, samples].
            y: Labels per sample.
            sfreq: Sampling frequency.
            y_dict: Label mapping dictionary.
            events: Event segments dictionary.
            ch_names: Channel names.
            tmin: Start time (0.0).
            data_type: Type of the data ('raw').

    Examples
    --------
    Load EEG data for subject 1, all sessions, and default labels:

    >>> from bciflow.datasets import attention_raw
    >>> eeg_data = attention_raw(subject=1)
    >>> print(eeg_data['X'].shape)  # Shape of the EEG data
    >>> print(eeg_data['y'])  # Labels
    '''
    """

    # Check if the subject input is valid
    if type(subject) != int:
        raise ValueError("subject has to be an int type value")
    if subject > 34 or subject < 1:
        raise ValueError("subject has to be between 1 and 34")
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'

    mat = loadmat("EEG Data/eeg_record14.mat")
    o = mat['o'][0][0]

    sfreq = int(o[3][0][0])
    labels_raw = o[4].flatten().astype(int)
    meta = o[6]


    eeg_continuous = meta[:, 2:16].T 
    X = np.expand_dims(eeg_continuous, axis=0)  
    y = labels_raw  
    unique_labels = np.unique(y)
    y_dict = {int(label): int(label) for label in unique_labels}

    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    dataset = {
        'X': X,
        'y': y,
        'sfreq': sfreq,
        'y_dict': y_dict,
        'ch_names': ch_names,
        'tmin': 0.0,
        'data_type': "raw"
    }

    return dataset