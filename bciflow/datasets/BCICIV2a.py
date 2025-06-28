'''
BCICIV2a.py

Description
-----------
This code is used to load EEG data from the BCICIV2a dataset. 
It modifies the data to fit the requirements of the eegdata class, 
which is used to store and process EEG data. 

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
import mne

def bciciv2a(subject:int = 1, 
            session_list: list=None, 
            run_list: list=None, 
            labels=['left-hand', 'right-hand', 'both-feet', 'tongue'],
            path = 'data/BCICIV2a/'):
    """
        Description
        -----------
        
        Load EEG data from the BCICIV2a dataset. 
        The data is loaded for a specific subject, session, and run.
        The data is filtered based on the event codes specified in the 'labels_dict'.

        Parameters
        ----------
            subject : int
                index of the subject to retrieve the data from
            session_list : list, optional
                list of session codes
            run_list : list, optional
                list of run numbers
            labels_dict : dict
                dictionary mapping event names to event codes


        Returns
        -------
        dict
            A dictionary containing the following keys:

            X: EEG data as a numpy array.
            y: Labels corresponding to the EEG data.
            sfreq: Sampling frequency of the EEG data.
            y_dict: Mapping of labels to integers.
            events: Dictionary describing event markers.
            ch_names: List of channel names.
            tmin: Start time of the EEG data.
            data_type: Explains how the data is placed inside the dictionary. Type 'epochs' means labels per trial and 'raw' means labels per time.

        """

    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9 or subject < 1:
        raise ValueError("Has to be an existing subject")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if type(run_list) != list and run_list != None:
        raise ValueError("Has to be an List or None type")
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

    if session_list is None:
        session_list = ['T', 'E']

    rawData, rawLabels = [], []

    for sec in session_list:
        raw=mne.io.read_raw_gdf(path+'/A%02d%s.gdf'%(subject, sec), preload=True, verbose='ERROR')
        raw_data = raw.get_data()[:22]
        annotations = raw.annotations.to_data_frame()
        first_timestamp = pd.to_datetime(annotations['onset'].iloc[0])
        annotations['onset'] = (pd.to_datetime(annotations['onset']) - first_timestamp).dt.total_seconds()
        annotations['description'] = annotations['description'].astype(int)
        new_trial_time = np.array(annotations[annotations['description']==768]['onset'])

        times_ = np.array(raw.times)
        rawData_ = []
        for trial_ in new_trial_time:
            idx_ = np.where(times_ == trial_)[0][0]
            rawData_.append(raw_data[:, idx_:idx_+1875])
        rawData_ = np.array(rawData_)
        rawLabels_ = np.array(scipy.io.loadmat(path+'/A%02d%s.mat'%(subject, sec))['classlabel']).reshape(-1)

        rawData.append(rawData_)
        rawLabels.append(rawLabels_)



    X, y = np.concatenate(rawData), np.concatenate(rawLabels)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
    labels_dict = {1: 'left-hand', 2: 'right-hand',3:"both-feet",4:"tongue"}
    y = np.array([labels_dict[i] for i in y])
    selected_labels = np.isin(y, labels)
    X, y = X[selected_labels], y[selected_labels]
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y])
    print(X.shape, y.shape)
    return {'X': X, 
            'y': y, 
            'sfreq': sfreq, 
            'y_dict': y_dict,
            'events': events, 
            'ch_names': ch_names,
            'tmin': tmin,
            'data_type': "epochs"}