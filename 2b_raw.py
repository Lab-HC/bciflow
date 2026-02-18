import numpy as np
import pandas as pd
import scipy
import mne
from typing import List, Optional, Dict, Any

def bciciv2b_raw(subject: int=1, 
             session_list: Optional[List[str]] = None, 
             labels: List[str] = ['left-hand', 'right-hand'],
             path: str = 'data/BCICIV2b/') -> Dict[str, Any]:
    """
    Description
    -----------
    
    This function loads EEG data for a specific subject and session from the bciciv2b dataset.
    It processes the data to fit the structure of the `eegdata` dictionary, which is used
    for further processing and analysis.


    The dataset can be found at:
     - https://www.bbci.de/competition/iv/#download
     - https://www.bbci.de/competition/iv/results/index.html#labels

    Parameters
    ----------
        subject : int
            index of the subject to retrieve the data from
        session_list : list, optional
            list of session codes
        labels : dict
            dictionary mapping event names to event codes
        path :
            path to the directory tha contains the datasets files.


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
        
    Examples
    --------
    Load EEG data for subject 1, all sessions, and default labels:

    >>> from bciflow.datasets import bciciv2b
    >>> eeg_data = bciciv2b(subject=1)
    >>> print(eeg_data['X'].shape)  # Shape of the EEG data
    >>> print(eeg_data['y'])  # Labels
    '''
    """

    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9 or subject < 1:
        raise ValueError("Has to be an existing subject")
    if type(labels) != list:
        raise ValueError("labels has to be a list type value")
    for i in labels:
        if i not in ['left-hand', 'right-hand']:
            raise ValueError("labels has to be a sublist of ['left-hand', 'right-hand']")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if path[-1] != '/':
        path += '/'
        
    sfreq = 250.
    events = {'get_start': [0, 3],
                'beep_sound': [2],
                'cue': [3, 4],
                'task_exec': [4, 7],
                'break': [7, 8.5]}
    ch_names = ['C3', 'Cz', 'C4']
    ch_names = np.array(ch_names)
    tmin = 0.

    if session_list is None:
        session_list = ['01T', '02T', '03T', '04E', '05E']

    raw_data, raw_labels = [], []
    for sec in session_list:
        raw=mne.io.read_raw_gdf(path+'B%02d%s.gdf'%(subject, sec), preload=True, verbose='ERROR')
        raw_data_ = raw.get_data()[:3]
        raw_labels_ = np.array(scipy.io.loadmat(path+'/A%02d%s.mat'%(subject, sec))['classlabel']).reshape(-1)
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
            end_break_idx = np.searchsorted(times_, start_trial + 7.5)
            y_labels[start_break_idx:end_break_idx] = 10

        raw_data.append(raw_data_)
        raw_labels.append(y_labels)    

    X = np.concatenate(raw_data, axis=1)
    y = np.concatenate(raw_labels)

    y_dict = {
        1: "fixation-cross",
        2: "left-cue",
        3: "right-cue",
        6: "left-imagery",
        7: "right-imagery",
        10: "break",
        11: "idling-eyes-open",
        12: "idling-eyes-closed"
    }

    return {'data_type': 'raw',
        'X': X,             
        'y': y,             
        'sfreq': sfreq,
        'y_dict': y_dict,
        'ch_names': ch_names,
        'tmin': tmin}
