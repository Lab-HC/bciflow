import pyedflib as plib
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def _string_to_number(label : str, run) -> int:

    #real_left_right = [3, 7, 11] # Left hand and Right hand
    real_both = [5, 9, 13] # Both hands and Both feet
    imagine_left_right = [4, 8, 12]
    imagine_both = [6, 10, 14]

    mapping_real_left_right = {
        'T0': 0, # Rest
        'T1': 1, # Left hand
        'T2': 2 # Right hand 
    }
    mapping_real = {
        'T0': 0, # Rest
        'T1': 3, # Both hands
        'T2': 4, # Both feet
    }
    mapping_imagine_left_right = {
        'T0': 0, # Rest
        'T1': 5, # Imagine left hand
        'T2': 6 # Imagine right hand
    }
    mapping_imagine_both = {
        'T0': 0, # Rest
        'T1': 7, # Imagine both hands
        'T2': 8 # Imagine both feet
    }
    
    mapping = mapping_real_left_right

    if run in real_both:
        mapping = mapping_real
    elif run in imagine_left_right:
        mapping = mapping_imagine_left_right
    elif run in imagine_both:
        mapping = mapping_imagine_both

    return mapping.get(label, -1)  # Retorna -1 se o rótulo não for encontrado


def physio_net(subject : int = 1,
                session_list : Optional[List[str]] = None,
                labels : List[str] = ['rest', 'left-hand', 'right-hand', 'both-hands', 'both-feet', 'imagine-left-hand', 'imagine-right-hand', 'imagine-both-hands', 'imagine-both-feet'],
                path : str = 'data/PhysioNET/', 
                verbose:str ='ERROR') -> Dict[str, Any]: 
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
            List of session numbers to load (e.g., [3, 4, 5] for sessions 3, 4, and 5). If None, sessions (3-14) are loaded.
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
        if i not in ['rest', 'left-hand', 'right-hand', 'both-hands', 'both-feet', 'imagine-left-hand', 'imagine-right-hand', 'imagine-both-hands', 'imagine-both-feet']:
            raise ValueError("labels has to be a sublist of ['rest', 'left-hand', 'right-hand', 'both-hands', 'both-feet', 'imagine-left-hand', 'imagine-right-hand', 'imagine-both-hands', 'imagine-both-feet'],")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if session_list == None:
        session_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    else:
        for i in session_list:
            if type(i) != int or (i < 1 or i > 14):
                raise ValueError("Session list has to be a sublist of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],")
    
    if type(path) != str:
        raise ValueError("Has to be a string type value")
    if path[-1] != '/':
        path += '/'
    
    if type(verbose) != str:
        raise ValueError("Has to be a string type value")
    if verbose not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError("verbose has to be one of the following: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

    X_og = np.empty((2, 1, 64, 9760))
    Y = [] 

    if 1 in session_list:
        session_list.remove(1)
        newPath = path + f'S{subject:03d}/S{subject:03d}R01.edf'
        signals, signals_header, header = plib.highlevel.read_edf(newPath)
        X_og[0, 0, :, :] = signals[:, 0:9760]
        Y.append(0) # Rest

    if 2 in session_list:
        session_list.remove(2)
        newPath = path + f'S{subject:03d}/S{subject:03d}R02.edf'
        signals, signals_header, header = plib.highlevel.read_edf(newPath)
        X_og[1, 0, :, :] = signals[:, 0:9760]
        Y.append(0) # Rest

    X = np.empty((30*len(session_list), 1, 64, 640))
    ch_names = []
    min_index = min(session_list)

    for i in session_list:
        newPath = path + f'S{subject:03d}/S{subject:03d}R{i:02d}.edf'
        
        #eventFile = mne.io.read_raw_edf(newPath)

        #annotations = eventFile.annotations

        #description = annotations.description.tolist()
        #description = [_string_to_number(label, i) for label in description]

        signals, signals_header, header = plib.highlevel.read_edf(newPath)
        annotations = header['annotations']

        for session_idx in range(0, 30):
            start_time_hz = int(annotations[session_idx][0] * 160)
            end_time_hz = int((annotations[session_idx][0] + 4.0) * 160)

            Y.append(_string_to_number(annotations[session_idx][2], i))

            session = signals[:, start_time_hz:end_time_hz]
            X[(i-min_index) * 30 + session_idx, 0, :, 0:640] = session
            
        if i == min_index:
            for h in signals_header:
                newValue = h['label'].replace('.', '')
                ch_names.append(newValue)


    for i in range(len(Y) - 1, -1, -1):
        if Y[i] == 0:
            Y.pop(i)
            X = np.delete(X, i, axis=0)

    y_dict = {
        0: 'rest',
        1: 'left-hand',
        2: 'right-hand',
        3: 'both-hands',
        4: 'both-feet',
        5: 'imagine-left-hand',
        6: 'imagine-right-hand',
        7: 'imagine-both-hands',
        8: 'imagine-both-feet'
    }

    events = {
        'get-start': [0, 0],
        'beep-sound': [0],
        'cue': [0, 4],
        'task_exec': [0, 4]
    }
    
    eegdata : Dict[str, Any] = {
        'X': X,
        'X_og': X_og,
        'y': Y,
        'y_dict': y_dict,
        'events': events,
        'ch_names': ch_names,
        'sfreq': 160.,
        'tmin': 0.,
        'data_type': 'epochs'
    }

    return eegdata

if __name__ == "__main__":
    data = physio_net(subject=1, path='../../data/PhysioNET/', session_list=[3])
    print(data['data_type'])
    print(data['X'].shape)
    print(data['y'])
    print(data['y_dict'])
    print(data['events'])
    print(data['ch_names'])
    print(data['sfreq'])
    print(data['tmin'])