
'''
BCICIV2a.py

Description
-----------
This code is used to load EEG data from the BCICIV2a dataset. It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data. 

Dependencies
------------
numpy
pandas
scipy
mne 

'''
import numpy as np
import pandas as pd
import scipy

def bciciv2a(subject:int = 1, session_list: list=None, run_list: list=None, labels: list = ['left-hand', 'right-hand', 'both-feet', 'tongue'], path: str = 'data/BCICIV2a/'):

    """
    Description
    -----------
    
    Load EEG data from the BCICIV2a dataset. 
    It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data. 

    Parameters
    ----------
        subject : int
            index of the subject to retrieve the data from
        session_list : list, optional
            list of session codes
        run_list : list, optional
            list of run numbers
        labels : list
            list mapping event names to event codes
        verbose : str
            verbosity level


    Returns:
    ----------
        eegdata: An instance of the eegdata class containing the loaded EEG data.

    """
    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9:
        raise ValueError("Has to be an existing subject")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if type(run_list) != list and run_list != None:
        raise ValueError("Has to be an List or None type")
    if type(labels) != list:
        raise ValueError("Has to be an List type")
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'
    sfreq = 250.
    events = {'get_start': [0, 2],
                'beep_sound': [0],
                'cue': [2, 3.25],
                'task_exec': [3, 6],
                'break': [6, 7.5]}
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    ch_names = np.array(ch_names)
    tmin = 0.

    """
    'sfreq' is set to 250. This represents the sampling frequency of the EEG data.
    'events' is a dictionary that maps event names to their corresponding time intervals.
    'ch_names' is a list of channel names.
    'tmin' is set to 0, representing the starting time of the EEG data.
    """

    if session_list is None:
        session_list = ['T', 'E']

    rawData, rawLabels = [], []

    """
    If 'session_list' is not provided, it is set to ['T', 'E'].
    'rawData' and 'rawLabels' are empty lists that will store the EEG data and labels for each session.
    """
    for sec in session_list:
        file_name = 'parsed_P%02d%s.mat' % (subject, sec)
        try:
            raw = scipy.io.loadmat(path + file_name)
        except:
            raise ValueError("The file %s does not exist in the path %s" % (file_name, path))

        rawData_ = raw['RawEEGData']
        rawLabels_ = np.reshape(raw['Labels'], -1)
        rawData_ = np.reshape(rawData_, (rawData_.shape[0], 1, rawData_.shape[1], rawData_.shape[2]))
        rawData.append(rawData_)
        rawLabels.append(rawLabels_)

    """
    For each session in the 'session_list', the raw EEG data is loaded using mne.io.read_raw_gdf.
    The data is filtered to include only the first 22 channels.
    The 'annotations' (relevant timestamps) are extracted and converted to a DataFrame.
    The onset times are normalized to start from zero.
    The event descriptions are converted to integers.
    The 'new_trial_time' is obtained by extracting the onset times of the '768' event ('768' is the code for new trial in the dataset).
    The 'times_' array is obtained from the raw data.
    The EEG data is extracted for each trial based on the 'new_trial_time'.
    The data is reshaped to include only the relevant channels and time points.
    The class labels are loaded from the corresponding .mat file.
    The raw data and labels are appended to the 'rawData' and 'rawLabels' lists.
    """

    X, y = np.concatenate(rawData), np.concatenate(rawLabels)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
    labels_dict = {1: 'left-hand', 2: 'right-hand',3:"both-feet",4:"tongue"}
    y = np.array([labels_dict[i] for i in y])
    selected_labels = np.isin(y, labels)
    X, y = X[selected_labels], y[selected_labels]
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y])

    return {'X': X, 
            'y': y, 
            'sfreq': sfreq, 
            'y_dict': y_dict,
            'events': events, 
            'ch_names': ch_names,
            'tmin': tmin}
