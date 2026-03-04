import numpy as np
from scipy.io import loadmat
from typing import List, Optional, Dict, Any

def cbcic_raw(subject: int = 1, 
          session_list: Optional[List[str]] = None,
          path: str = 'D:/Arquivos/BCI/BCI/tarefas/datasets/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow') -> Dict[str, Any]:
    '''
    Description
    -----------

    This function loads EEG data for a specific subject and session from the cbcic dataset.
    It processes the data to fit the structure of the `eegdata` dictionary, which is used
    for further processing and analysis.

    The dataset can be found at: 
     - https://github.com/5anirban9/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow

    Parameters
    ----------
    subject : int, optional
        Index of the subject to retrieve the data from. Must be between 1 and 10.
        Default is 1.
    session_list : list, optional
        List of session codes to load. Valid options are 'T' (training) and 'E' (evaluation).
        If None, all sessions are loaded. Default is None.
    labels : list, optional
        List of labels to include in the dataset. Valid options are 'left-hand' and 'right-hand'.
        Default is ['left-hand', 'right-hand'].
    path : str, optional
        Path to the folder containing the dataset files. Default is 'data/cbcic/'.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        - X: EEG data as a numpy array [trials, 1, channels, time].
        - y: Labels corresponding to the EEG data.
        - sfreq: Sampling frequency of the EEG data.
        - y_dict: Mapping of labels to integers.
        - events: Dictionary describing event markers.
        - ch_names: List of channel names.
        - tmin: Start time of the EEG data.
        - data_type: Type of the data ('epochs').

        
    Raises
    ------
    ValueError
        If any of the input parameters are invalid or if the specified file does not exist.

    Examples
    --------
    Load EEG data for subject 1, all sessions, and default labels:

    >>> from bciflow.datasets import cbcic.raw
    >>> eeg_data = cbcic.raw(subject=1)
    >>> print(eeg_data['X'].shape)  # Shape of the EEG data
    >>> print(eeg_data['y'])  # Labels
    '''

    # Check if the subject input is valid
    if type(subject) != int:
        raise ValueError("subject has to be an int type value")
    if subject > 10 or subject < 1:
        raise ValueError("subject has to be between 1 and 10")

    # Check if the session_list input is valid
    if type(session_list) != list and session_list is not None:
        raise ValueError("session_list has to be a list or None")
    if session_list is not None:
        for i in session_list:
            if i not in ['T', 'E']:
                raise ValueError("session_list has to be a sublist of ['T', 'E']")
    else:
        session_list = ['T', 'E']
    # Check if the path input is valid
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'

    # Set basic parameters of the clinical BCI challenge dataset
    sfreq = 512.
    ch_names = np.array(["F3", "FC3", "C3", "CP3", "P3",
                         "FCz", "CPz",
                         "F4", "FC4", "C4", "CP4", "P4"])

    continuous_data = []
    events = []
    offset = 0

    for sec in session_list:

        file_name = 'parsed_P%02d%s.mat' % (subject, sec)
        raw = loadmat(path + file_name)

        eeg = raw['RawEEGData']        
        labels_raw = np.reshape(raw['Labels'], -1)
        if eeg.shape[1] == 12:  
            eeg = np.transpose(eeg, (0, 2, 1))
        n_trials, n_samples, n_channels = eeg.shape
        for i in range(n_trials):

            trial = eeg[i]               
            continuous_data.append(trial)
            if labels_raw[i] in [1, 2]:
                events.append([offset, 0, labels_raw[i]])

            offset += n_samples

    X = np.concatenate(continuous_data, axis=0)
    events = np.array(events)
    y_dict = {1: 0, 2: 1}
    events[:, 2] = np.vectorize(y_dict.get)(events[:, 2])

    return {
        'X': X,
        'events': events,
        'sfreq': sfreq,
        'ch_names': ch_names,
        'data_type': 'raw'
    }
