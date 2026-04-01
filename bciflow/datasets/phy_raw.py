import pyedflib as plib
from typing import List, Dict, Any, Optional
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

def physionet_raw(subject : int = 1,
                session_list : Optional[List[str]] = None,
                labels : List[str] = ['rest', 'left-hand', 'right-hand', 'both-hands', 'both-feet'],
                path : str = 'data/PhysioNET/',
                verbose : str = 'ERROR') -> Dict[str, Any]: 
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
        - X: EEG data array of shape (n_trials, n_channels, n_times).
        - y: Labels array of shape (n_trials,).
        - sfreq: Sampling frequency of the EEG data.
        - y_dict: Mapping of labels to integers.
        - ch_names: List of channel names.
        - tmin: Start time of the EEG data.
        
    Examples
    --------
    Load EEG data for subject 1, all sessions and default labels:
    >>> from bciflow.datasets import physionet_raw
    >>> eegdata = physionet_raw(subject=1)
    >>> print(eegdata['X'].shape)  # (channels, time)
    >>> print(eegdata['ch_names'])  # Channel names
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
    if session_list == None:
        session_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    else:
        for i in session_list:
            if type(i) != int or i < 1 or i > 14:
                raise ValueError("Session list has to be a sublist of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],")
    if type(path) != str:
        raise ValueError("Has to be a string type value")
    if type(verbose) != str:
        raise ValueError("Has to be a string type value")
    if verbose not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError("verbose has to be one of the following: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")
    
    if path[-1] != '/':
        path += '/'

    # Aqui, sessions é 12 * 30 para que o Y ainda esteja de acordo

    x_length = 20000 * len(session_list) # 20000 ticks por sessão, exceto a 1 e 2.
    if 1 in session_list:
        x_length += 9760 - 20000 # Remove os 2000 padrão e adiciona os 9760
    
    if 2 in session_list:
        x_length += 9760 - 20000


    X = np.empty((64, x_length)) # (channels, time) 
    Y = []
    ch_names = []
    
    min_session_list = min(session_list)
    start_index = 0

    for i in session_list:
        newPath = path + f'S{subject:03d}/S{subject:03d}R{i:02d}.edf'

        signals, signals_header, header = plib.highlevel.read_edf(newPath)

        annotations = header['annotations']

        X[:, start_index:(start_index + signals.shape[1])] = signals
        start_index += signals.shape[1]

        for j in range(len(annotations)):
            Y.append(_string_to_number(annotations[j][2], i))
            
        if i == min_session_list:
            for header in signals_header:
                newValue = header['label'].replace('.', '')
                ch_names.append(newValue)
    
    eegdata : Dict[str, Any] = {
        'X': X,
        'y': Y,
        'y_dict': {
            0: 'Rest',
            1: 'Left-hand',
            2: 'Right-hand',
            3: 'Both-hands',
            4: 'Both-feet'
        },
        'ch_names': ch_names,
        'sfreq': 160.,
        'tmin': 0.,
        'data_type': 'raw'
    }

    return eegdata