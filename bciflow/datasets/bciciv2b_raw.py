import numpy as np
import pandas as pd
import scipy
import mne
from typing import List, Optional, Dict, Any

def bciciv2b_raw(subject: int=1, 
             session_list: Optional[List[str]] = None, 
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

    >>> from bciflow.datasets import bciciv2b_raw
    >>> eeg_data = bciciv2b_raw(subject=1)
    >>> print(eeg_data['X'].shape)  # Shape of the EEG data
    >>> print(eeg_data['y'])  # Labels
    '''
    """

    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9 or subject < 1:
        raise ValueError("Has to be an existing subject")
    if type(session_list) != list and session_list is not None:
        raise ValueError("Has to be a List or None type")
    if path[-1] != '/':
        path += '/'
        
    sfreq = 250.
    ch_names = np.array(['C3', 'Cz', 'C4'])
    tmin = 0.

    if session_list is None:
        session_list = ['01T', '02T', '03T', '04E', '05E']

    raw_data, raw_labels = [], []
    event_conversion = {
        276: 11,   # idling-eyes-open
        277: 12,   # idling-eyes-closed

        768: 20,   # start-of-trial
        769: 2,    # left-imagery
        770: 3,    # right-imagery

        781: 21,   # feedback
        783: 22,   # cue-unknown

        1023: 30,  # rejected-trial

        1077: 40,  # horizontal-eye-movement
        1078: 41,  # vertical-eye-movement
        1079: 42,  # eye-rotation
        1081: 43,  # eye-blink

        32766: 50  # start-of-run
    }

    for sec in session_list:

        raw = mne.io.read_raw_gdf(
            path + f'B{subject:02d}{sec}.gdf',
            preload=True,
            verbose='ERROR'
        )

        raw_data_ = raw.get_data()[:3]

        annotations = raw.annotations.to_data_frame()

        first_timestamp = pd.to_datetime(annotations['onset'].iloc[0])
        annotations['onset'] = (
            pd.to_datetime(annotations['onset']) - first_timestamp
        ).dt.total_seconds()

        annotations['description'] = annotations['description'].astype(int)

        times_ = raw.times
        y_labels = np.zeros(len(times_))

        for i, row in annotations.iterrows():

            gdf_code = int(row['description'])

            if gdf_code not in event_conversion:
                continue

            onset_time = row['onset']
            start_idx = np.searchsorted(times_, onset_time)

            if i < len(annotations) - 1:
                next_onset = annotations.iloc[i + 1]['onset']
                end_idx = np.searchsorted(times_, next_onset)
            else:
                end_idx = len(times_)

            y_labels[start_idx:end_idx] = event_conversion[gdf_code]

        raw_data.append(raw_data_)
        raw_labels.append(y_labels)

    X = np.concatenate(raw_data, axis=1)
    y = np.concatenate(raw_labels)

    y_dict = {
        0: "background",

        2: "left-imagery",
        3: "right-imagery",

        11: "idling-eyes-open",
        12: "idling-eyes-closed",

        20: "start-of-trial",
        21: "bci-feedback",
        22: "cue-unknown",

        30: "rejected-trial",

        40: "horizontal-eye-movement",
        41: "vertical-eye-movement",
        42: "eye-rotation",
        43: "eye-blink",

        50: "start-of-run"
    }

    return {
        'data_type': 'raw',
        'X': X,
        'y': y,
        'sfreq': sfreq,
        'y_dict': y_dict,
        'ch_names': ch_names,
        'tmin': tmin
    }

if __name__ == "__main__":
    data = bciciv2b_raw(subject=1, path='C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/BCICIV2b')
    print(data['data_type'])
    print(data['X'].shape)
    print(data['y'].shape)
    print(data['sfreq'])
    print(data['y_dict'])
    print(data['ch_names'])
    print(data['tmin'])