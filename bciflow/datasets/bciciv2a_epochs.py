"""BCICIV2a epochs dataset."""

import numpy as np
import pandas as pd
import scipy
import mne

def __set_default_bciciv2a_params(
        subject,
        labels,
        session_list,
        eog,
        path
    ):

    labels = labels if labels is not None else [
            'left-hand', 
            'right-hand', 
            'both-feet', 
            'tongue'
        ]

    session_list = session_list if session_list is not None else ['T', 'E']

    return (
            subject,
            labels,
            session_list,
            eog,
            path
        )

def __check_bciciv2a_params(
        subject,
        labels,
        session_list,
        eog,
        path
    ):

    (
            subject,
            labels,
            session_list,
            eog,
            path
        ) = __set_default_bciciv2a_params(
                subject,
                labels,
                session_list,
                eog,
                path
            )

    return (
            subject,
            labels,
            session_list,
            eog,
            path
        )

def __set_bciciv2a_channel_names(
        eog
    ):

    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    if eog:
        ch_names += ['EOG-left', 'EOG-central', 'EOG-right']
    return np.array(ch_names)

def __read_trial_time(
        raw
    ):

    raw_data = raw.get_data()
    annotations = raw.annotations.to_data_frame()
    first_timestamp = pd.to_datetime(annotations['onset'].iloc[0])
    annotations['onset'] = (pd.to_datetime(annotations['onset'])
                            - first_timestamp).dt.total_seconds()
    annotations['description'] = annotations['description'].astype(int)
    new_trial_time = np.array(
            annotations[annotations['description']==768]['onset']
        )

    return raw_data, new_trial_time

def __raw_to_epochs(
        raw_data,
        new_trial_time,
        times,
    ):

    times_ = np.array(times)
    _raw_data_cropped = []
    for trial_ in new_trial_time:
        idx_ = np.where(times_ == trial_)[0][0]
        _raw_data_cropped.append(raw_data[:, idx_:idx_+1875])
    _raw_data_cropped = np.array(_raw_data_cropped)

    return _raw_data_cropped

def __read_raw_data(
        subject,
        session_list,
        path,
    ):

    raw_data, raw_labels = [], []

    for sec in session_list:

        _raw=mne.io.read_raw_gdf(f"{path}A{subject:02d}{sec}.gdf",
                                preload=True,
                                verbose='ERROR')

        _raw_labels = np.array(scipy.io.loadmat(
                f"{path}/A{subject:02d}{sec}.mat")['classlabel']
            ).reshape(-1)

        _raw_data, new_trial_time = __read_trial_time(_raw)

        raw_data_cropped = __raw_to_epochs(_raw_data, new_trial_time, _raw.times)

        raw_data.append(raw_data_cropped)
        raw_labels.append(_raw_labels)

    x, y = np.concatenate(raw_data), np.concatenate(raw_labels)
    return x, y

def __select_labels(
        x,
        y,
        labels,
        labels_dict
    ):

    y = np.array([labels_dict[i] for i in y])
    selected_labels = np.isin(y, labels)
    x, y = x[selected_labels], y[selected_labels]
    y_dict = {labels[i]: i for i in range(len(labels))}
    y = np.array([y_dict[i] for i in y])

    return x, y, y_dict

def bciciv2a_epochs(
        subject: int,
        labels: list[str] | None = None,
        session_list: list[str] | None = None,
        eog: bool = False,
        path: str = 'data/BCICIV2a/'
    ) -> dict[str, any]:
    """Load the BCICIV2a dataset in epoched format.
    
    The BCICIV2a dataset consists of EEG recordings from 9 subjects
    performing 4 different motor imagery tasks: left-hand,
    right-hand, both-feet, and tongue. Each subject participated in
    two sessions (T and E), with each session containing multiple 
    trials of the motor imagery tasks. The dataset includes 22 EEG
    channels and optionally 3 EOG channels for eye movement artifacts.
    The data is sampled at 250 Hz and can be used for various BCI 
    applications, such as classification of motor imagery tasks.
    
    The dataset can be found at:
     - https://www.bbci.de/competition/iv/#download
     - https://www.bbci.de/competition/iv/results/index.html#labels

    Parameters
    ----------
    subject : int
        Subject number (1-9).
    labels : list of str, optional
        A list of labels to include in the dataset (default is None).
    session_list : list of str, optional
        A list of session identifiers to load (e.g., ['T', 'E']). 
        If None, all sessions will be loaded (default is None).
    eog : bool, optional
        Whether to include EOG channels (default is False).
    path : str, optional
        The path to the dataset directory (default is
        'data/BCICIV2a/').

    Returns
    -------
    dict
        A dictionary containing the following
        keys and values:
        - 'data_type': A string indicating the type of data
          ('epochs').
        - 'X-shape': A list describing the shape of the data
          (['trials', 'channels', 'time_points']).
        - 'X': A numpy array containing the epoched EEG data with
          shape (n_trials, n_channels, n_time_points).
        - 'y': A numpy array containing the labels for each
          trial with shape (n_trials,).
        - 'sfreq': A float representing the sampling frequency
          of the data (250.0 Hz).
        - 'y_dict': A dictionary mapping label integers to
          descriptive strings for each class.
        - 'events': A dictionary mapping event names to their
          corresponding time intervals.
        - 'ch_names': A numpy array containing the names of the
          channels.
        - 'tmin': A float representing the starting time of the
          data (0.0 seconds).
    """

    (
            subject,
            labels,
            session_list,
            eog,
            path
        ) = __check_bciciv2a_params(
                subject,
                labels,
                session_list,
                eog,
                path
            )

    sfreq = 250.
    tmin = 0.
    ch_names = __set_bciciv2a_channel_names(eog)
    events = {'get_start': [0, 2],
            'beep_sound': [0],
            'cue': [2, 3.25],
            'task_exec': [3, 6],
            'break': [6, 7.5]}

    labels_dict = {
            1: 'left-hand',
            2: 'right-hand',
            3: 'both-feet',
            4: 'tongue'
        }

    x, y = __read_raw_data(subject, session_list, path)
    x = x[:, :22, :] if not eog else x

    x, y, y_dict = __select_labels(x, y, labels, labels_dict)

    return {
            'data_type': "epochs",
            'X-shape': ['trials', 'channels', 'time_points'],
            'X': x, 
            'y': y, 
            'sfreq': sfreq, 
            'y_dict': y_dict,
            'events': events, 
            'ch_names': ch_names,
            'tmin': tmin
        }
