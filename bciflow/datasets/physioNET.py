import pyedflib as plib
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import mne
import pandas as pd
import numpy as np


def string_to_number(label : str, run) -> int:

    first_case = [3, 4, 7, 8, 11, 12] # Left hand and Right hand
    #second_case = [5, 6, 9, 10, 13, 14] # Both hands and Both feet

    mapping1 = {
        'T0': 0, # descanso
        'T1': 1, # Left hand
        'T2': 2 # Right hand 
    }
    mapping = {
        'T0': 0, # descanso
        'T1': 3, # Both hands
        'T2': 4, # Both feet
    }
    if run in first_case:
        mapping = mapping1
    
    return mapping.get(label, -1)  # Retorna -1 se o rótulo não for encontrado


def physio_net(subject : int = 1,
                session_list : Optional[List[str]] = None,
                labels : List[str] = ['rest', 'left-hand', 'right-hand', 'both-hands', 'both-feet'],
                path : str = 'data/PhysioNET/') -> Dict[str, Any]: 
    """
    '''
    Description
    -----------

    This function loads EEG data for a specific subject and session from the PhysioNet dataset.
    It processes the data to fit the structure of the 'eegdata' dictionary, wich is used for further processing and analysis.

    The dataset can be found at:
    - https://physionet.org/content/eegmmidb/1.0.0/

    Parameters
    ----------
        subject : int
            Index of the subject to load.
        session_list : list, optional
            List of session codes
        labels : dict
            Dictionary mapping event names to event codes
        path : str
            Path to the dataset files
    
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
        - data_type: Type of the data ('epochs'). TODO: O que é?
        
    Examples
    --------
    Load EEG data for subject 1, all sessions and default labels:
    >>> from bciflow.datasets import physio_net
    >>> eegdata = physio_net(subject=1)
    >>> print(eegdata['X'].shape)  # (trials, 1, channels, time)
    >>> print(eegdata['y'])  # Labels    
    '''
    """

    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject < 1 or subject > 109:
        raise ValueError("Subject index out of range. Must be between 1 and 109.")    

    if type(labels) != list:
        raise ValueError("Labels must be provided as a list.")
    for i in labels:
        if i not in ['rest', 'left-hand', 'right-hand', 'both-hands', 'both-feet']:
            raise ValueError("labels has to be a sublist of ['rest', 'left-hand', 'right-hand', 'both-hands', 'both-feet'],")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if path[-1] != '/':
        path += '/'

    X = np.empty((360, 1, 64, 672))
    Y = [] 
    ch_names = []
    for i in range(3, 15):
        newPath = path + f'S{subject:03d}/S{subject:03d}R{i:02d}.edf'
        
        eventFile = mne.io.read_raw_edf(newPath)

        annotations = eventFile.annotations

        description = annotations.description.tolist()
        description = [string_to_number(label, i) for label in description]
        Y.extend(description)

        signals, signals_header, header = plib.highlevel.read_edf(newPath)

        for session_idx in range(len(annotations.onset) - 1):
            start_time_hz = int(annotations.onset[session_idx] * 160)
            end_time_hz = int(annotations.onset[session_idx + 1] * 160)

            session = signals[:, start_time_hz:end_time_hz]
            if session.shape[1] < 672:
                padding = 672 - session.shape[1]
                session = np.pad(session, ((0, 0), (0, padding)), mode='constant')

            X[i-1 * 30 + session_idx, 0, :, 0:672] = session
            
        if i == 1:
            for header in signals_header:
                newValue = header['label'].replace('.', '')
                ch_names.append(newValue)


    for i in range(len(Y) - 1, -1, -1):
        if Y[i] == 0:
            Y.pop(i)
            X = np.delete(X, i, axis=0)

    y_dict = {
        0: 'Rest',
        1: 'Left-hand',
        2: 'Right-hand',
        3: 'Both-hands',
        4: 'Both-feet'
    }
    
    eegdata : Dict[str, Any] = {
        'X': X,
        'y': Y,
        'y_dict': y_dict,
        'ch_names': ch_names,
        'sfreq': 160.,
        'tmin': 0.,
    }

    return eegdata