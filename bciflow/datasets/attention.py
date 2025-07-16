import numpy as np
from scipy.io import loadmat
from typing import Optional, Dict, Any, List


def attention(
    subject: int = 1,
    path: str = 'data/attention/',
    labels: List[str] = ['focused', 'unfocused'],
) -> Dict[str, Any]:
    """
    Description
    -----------
    This function loads EEG data for a specific subject and session from the attention dataset.
    It processes the data to fit the structure of the `eegdata` dictionary, which is used
    for further processing and analysis.

    The dataset can be found at:
     - 

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

    >>> from bciflow.datasets import attention
    >>> eeg_data = attention(subject=1)
    >>> print(eeg_data['X'].shape)  # Shape of the EEG data
    >>> print(eeg_data['y'])  # Labels
    '''
    """

    # Check if the subject input is valid
    if type(subject) != int:
        raise ValueError("subject has to be an int type value")
    if subject > 34 or subject < 1:
        raise ValueError("subject has to be between 1 and 34")
    if type(labels) != list:
        raise ValueError("labels has to be a list type value")
    for i in labels:
        if i not in ['focused', 'unfocused']:
            raise ValueError("labels has to be a sublist of ['focused', 'unfocused']")
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'


    mat = loadmat(path+"eeg_record%d.mat" % subject)
    o = mat['o'][0][0]

    sfreq = o[3][0][0]                  # e.g., 128.0 Hz
    labels = o[4].flatten()             # shape (n_samples,)
    timestamps = o[5]                   # shape (n_samples, 6)
    meta = o[6]                         # shape (n_samples, 25)

    eeg_continuous = meta[:, 2:16].T    # shape (14, n_samples)
    n_channels, n_samples = eeg_continuous.shape

    X = np.expand_dims(np.expand_dims(eeg_continuous, axis=0), axis=0)  # [1, 1, channels, samples]
    y = labels  # raw labels per sample
    events = {
        "focused": [0, 600],
        "unfocused": [600, 1200],
        "drowsy": [1200, 2100],
    }

    y_dict = {"focused": 0, "unfocused": 1}

    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    labels_dict = {1: 'left-hand', 2: 'right-hand',3:"both-feet",4:"tongue"}
    y = np.array([labels_dict[i] for i in y])
    selected_labels = np.isin(y, labels)
    X, y = X[selected_labels], y[selected_labels]
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y])

    dataset = {
        'X': X,
        'y': y,
        'sfreq': sfreq,
        'y_dict': y_dict,
        'events': events,
        'ch_names': ch_names,
        'tmin': 0.0,
        'data_type': "raw"
    }
    return dataset
