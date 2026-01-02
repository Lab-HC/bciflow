import numpy as np
import pandas as pd
import scipy
import mne
from typing import List, Dict, Any

import matplotlib.pyplot as plt

def bciciv2a_raw(subject: int=1,
                 session_list: List[str] = ['T', 'E'],
                 EOG: bool = False,
                 path: str = 'data/BCICIV2a/',
                 verbose='ERROR') -> Dict[str, Any]:
    """
    Description
    -----------
        
    This function loads EEG data for a specific subject and session from the bciciv2a dataset.
    It processes the data to fit the structure of the `eegdata` dictionary, which is used
    for further processing and analysis.

    The dataset can be found at:
     - https://www.bbci.de/competition/iv/#download
     - https://www.bbci.de/competition/iv/results/index.html#labels
    
    Parameters
    ----------
        subject : int
            index of the subject to retrieve the data from
        session_list : list
            list of session identifiers to be considered. 
            It should be a sublist of ['T','E'].
        EOG : bool
            whether to include EOG channels in the data.
        path :
            path to the directory tha contains the datasets files.
        verbose : str
            verbosity level for logging. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        - X: EEG data array of shape (n_trials, n_channels, n_times).
        - y: Labels array of shape (n_trials,).
        - sfreq: Sampling frequency of the EEG data.
        - y_dict: Mapping of labels to integers.
        - ch_names: List of channel names.
        - tmin: Start time of the EEG data.

    Examples
    --------
    Load EEG data for subject 1, all sessions, and default labels:

    >>> from bciflow.datasets import bciciv2a.raw
    >>> eeg_data = bciciv2a.raw(subject=1)
    >>> print(eeg_data['X'].shape)  # Shape of the EEG data
    >>> print(eeg_data['y'])  # Labels
    '''
    """
    
    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9 or subject < 1:
        raise ValueError("Has to be an existing subject")

    if type(session_list) != list:
        raise ValueError("session_list has to be a list type value")
    for i in session_list:
        if i not in ['T','E']:
            raise ValueError("session_list has to be a sublist of ['T','E']")
    
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'
    
    if type(verbose) != str:
        raise ValueError("verbose has to be a str type value")
    if verbose not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError("verbose has to be one of the following: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

    sfreq = 250.
    tmin = 0.

    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    if EOG:
        ch_names += ['EOG-left', 'EOG-central', 'EOG-right']
    ch_names = np.array(ch_names)

    raw_data, raw_labels = [], []
    for sec in session_list:
        raw=mne.io.read_raw_gdf(path+'/A%02d%s.gdf'%(subject, sec), preload=True, verbose=verbose)
        raw_labels_ = np.array(scipy.io.loadmat(path+'/A%02d%s.mat'%(subject, sec))['classlabel']).reshape(-1)
        raw_data_ = raw.get_data()[:len(ch_names)]
        annotations = raw.annotations.to_data_frame()
        first_timestamp = pd.to_datetime(annotations['onset'].iloc[0])
        annotations['onset'] = (pd.to_datetime(annotations['onset']) - first_timestamp).dt.total_seconds()
        annotations['description'] = annotations['description'].astype(int)

        times_ = np.array(raw.times)
        y_labels = np.zeros(len(times_))

        # idling eyes open
        new_trial_time = np.array(annotations[annotations['description']==276]['onset'])
        for i in range(len(new_trial_time)):
            start_trial = new_trial_time[i]
            y_labels[np.searchsorted(times_, start_trial):] = 11

        # idling eyes closed
        new_trial_time = np.array(annotations[annotations['description']==277]['onset'])
        for i in range(len(new_trial_time)):
            start_trial = new_trial_time[i]
            y_labels[np.searchsorted(times_, start_trial):] = 12

        # trials
        new_trial_time = np.array(annotations[annotations['description']==768]['onset'])
        for i in range(len(new_trial_time)):
            start_trial = new_trial_time[i]
            start_cue = start_trial + 2
            start_imagery = start_trial + 3
            start_break = start_trial + 6
            end_break = start_trial + 7.5
            start_trial_idx = np.searchsorted(times_, start_trial)
            start_cue_idx = np.searchsorted(times_, start_cue)
            start_imagery_idx = np.searchsorted(times_, start_imagery)
            start_break_idx = np.searchsorted(times_, start_break)

            y_labels[start_trial_idx:start_cue_idx] = 1
            y_labels[start_cue_idx:start_imagery_idx] = raw_labels_[i] + 1
            y_labels[start_imagery_idx:start_break_idx] = raw_labels_[i] + 5
            y_labels[start_break_idx:] = 10

        raw_data.append(raw_data_)
        raw_labels.append(y_labels)

    X, y = np.concatenate(raw_data, axis=1), np.concatenate(raw_labels)

    y_dict = {1:"fixation-cross", 2:"left-cue", 3:"right-cue", 4:"both-feet-cue", 5:"tongue-cue", 
              6:"left-imagery", 7:"right-imagery", 8:"both-feet-imagery", 9:"tongue-imagery", 10:"break",
             11:"idling-eyes-open", 12:"idling-eyes-closed"}

    return {'data_type': "raw",
            'X': X, 
            'y': y, 
            'sfreq': sfreq, 
            'y_dict': y_dict, 
            'ch_names': ch_names,
            'tmin': tmin}

if __name__ == "__main__":
    data = bciciv2a_raw(subject=1, session_list=['T', 'E'], EOG=False, path='../data/BCICIV2a/')
    print(data['data_type'])
    print(data['X'].shape)
    print(data['y'].shape)
    print(data['sfreq'])
    print(data['y_dict'])
    print(data['ch_names'])
    print(data['tmin'])
