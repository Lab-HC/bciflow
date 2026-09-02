"""BCICIV2a raw dataset."""

import numpy as np
import pandas as pd
import scipy
import mne

def __set_default_bciciv2a_raw_params(
        subject,
        session_list,
        eog,
        path,
    ):

    session_list = session_list if session_list is not None else ['T', 'E']

    return (
            subject,
            session_list,
            eog,
            path,
        )

def __check_bciciv2a_raw_params(
        subject,
        session_list,
        eog,
        path,
    ):

    (
            subject,
            session_list,
            eog,
            path,
        ) = __set_default_bciciv2a_raw_params(
                subject,
                session_list,
                eog,
                path,
            )

    return (
            subject,
            session_list,
            eog,
            path,
        )

def __set_bciciv2a_raw_channel_names(eog):
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    if eog:
        ch_names += ['EOG-left', 'EOG-central', 'EOG-right']
    return np.array(ch_names)

def __get_times(
        raw
    ):

    annotations = raw.annotations.to_data_frame()
    first_timestamp = pd.to_datetime(annotations['onset'].iloc[0])
    annotations['onset'] = (pd.to_datetime(annotations['onset']) 
                            - first_timestamp).dt.total_seconds()
    annotations['description'] = annotations['description'].values
    annotations['description'] = annotations['description'].astype(int)
    time_276 = np.array(
            annotations['onset'][annotations['description']==276]
        )
    time_277 = np.array(
            annotations['onset'][annotations['description']==277]
        )
    time_768 = np.array(
            annotations['onset'][annotations['description']==768]
        )
    
    return time_276, time_277, time_768

def __set_eyes_open_closed_labels(
        y_labels,
        times_,
        time_276,
        time_277
    ):

    for _, start_trial in enumerate(time_276):
        y_labels[np.searchsorted(times_, start_trial):] = 10

    for _, start_trial in enumerate(time_277):
        y_labels[np.searchsorted(times_, start_trial):] = 11

    return y_labels

def __set_trial_labels(
        y_labels,
        times_,
        time_768,
        raw_labels_
    ):

    for i, start_trial in enumerate(time_768):
        start_cue = start_trial + 1
        start_imagery = start_trial + 2
        start_break = start_trial + 5
        start_trial_idx = np.searchsorted(times_, start_trial)
        start_cue_idx = np.searchsorted(times_, start_cue)
        start_imagery_idx = np.searchsorted(times_, start_imagery)
        start_break_idx = np.searchsorted(times_, start_break)

        y_labels[start_trial_idx:start_cue_idx] = 0
        y_labels[start_cue_idx:start_imagery_idx] = raw_labels_[i]
        y_labels[start_imagery_idx:start_break_idx] = raw_labels_[i] + 4
        y_labels[start_break_idx:] = 9

    return y_labels

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

        raw_data_ = _raw.get_data()
        time_276, time_277, time_768 = __get_times(_raw)

        y_labels = np.zeros(len(_raw.times))
        y_labels = __set_eyes_open_closed_labels(
                y_labels,
                _raw.times,
                time_276,
                time_277
            )
        y_labels = __set_trial_labels(
                y_labels,
                _raw.times,
                time_768,
                _raw_labels
            )

        raw_data.append(raw_data_)
        raw_labels.append(y_labels)

    x, y = np.concatenate(raw_data, axis=1), np.concatenate(raw_labels)
    return x, y

def bciciv2a_raw(
        subject: int=1,
        session_list: list[str] | None = None,
        eog: bool = False,
        path: str = 'data/BCICIV2a/',
    ) -> dict[str, any]:
    """Load BCICIV2a raw dataset.
    
    This function loads the BCICIV2a dataset in its raw format,
    which includes the continuous EEG data along with the
    corresponding labels for each time point. The dataset is
    organized into sessions, and each session contains a series
    of trials with specific events and conditions.
    
    The dataset can be found at:
     - https://www.bbci.de/competition/iv/#download
     - https://www.bbci.de/competition/iv/results/index.html#labels
    
    Parameters
    ----------
    subject : int, optional
        The subject number to load (default is 1).
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
          ('raw').
        - 'X-shape': A list describing the shape of the data
          (['channels', 'time_points']).
        - 'X': A numpy array containing the raw EEG data with
          shape (n_channels, n_time_points).
        - 'y': A numpy array containing the labels for each
          time point with shape (n_time_points,).
        - 'sfreq': A float representing the sampling frequency
          of the data (250.0 Hz).
        - 'y_dict': A dictionary mapping label integers to
          descriptive strings for each class.
        - 'ch_names': A numpy array containing the names of the
          channels.
        - 'tmin': A float representing the starting time of the
          data (0.0 seconds).

    """

    (
            subject,
            session_list,
            eog,
            path,
        ) = __check_bciciv2a_raw_params(
                subject,
                session_list,
                eog,
                path,
            )

    sfreq = 250.
    ch_names = __set_bciciv2a_raw_channel_names(eog)

    y_dict = {
            "fixation-cross": 0,
            "left-cue": 1,
            "right-cue": 2,
            "both-feet-cue": 3,
            "tongue-cue": 4,
            "left-imagery": 5,
            "right-imagery": 6,
            "both-feet-imagery": 7,
            "tongue-imagery": 8,
            "break": 9,
            "idling-eyes-open": 10,
            "idling-eyes-closed": 11
        }

    x, y = __read_raw_data(subject, session_list, path)
    x = x[:22, :] if not eog else x

    return {
            'data_type': "raw",
            'X-shape': ['channels', 'time_points'],
            'X': x, 
            'y': y, 
            'sfreq': sfreq, 
            'y_dict': y_dict, 
            'ch_names': ch_names
        }
