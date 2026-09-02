"""Function to remove labels from eegdata."""

import numpy as np

def remove_label_raw(eegdata,
        labels,
        delete_data=False
    ):
    """Remove labels from eegdata.
    
    Parameters
    ----------
    eegdata : dict
        EEG data in the format of bciflow.
    labels : list
        List of labels to remove.
    delete_data : bool, optional
        Whether to delete the data corresponding to the removed labels, by default False.

    Returns
    -------
    dict
        EEG data with the specified labels removed.
    """

    ids = []
    for label in labels:
        if label in eegdata['y_dict']:
            ids.append(eegdata['y_dict'][label])
        else:
            print(f"Label {label} not found in y_dict.")

    for _id in labels:
        if _id in eegdata['y_dict']:
            del eegdata['y_dict'][_id]
        else:
            print(f"Label {_id} not found in y_dict.")

    if delete_data:
        mask = np.isin(eegdata['y'], ids)
        eegdata['X'] = eegdata['X'][:, ~mask]
        eegdata['y'] = eegdata['y'][~mask]

    else:
        mask = np.isin(eegdata['y'], ids)
        eegdata['y'][mask] = -1
        eegdata['y_dict']['none'] = -1

    # sort y_dict by value
    eegdata['y_dict'] = dict(sorted(eegdata['y_dict'].items(), key=lambda item: item[1]))

    # reorder y_dict values
    for i, key in enumerate(eegdata['y_dict'].keys()):
        eegdata['y_dict'][key] = i

    eegdata['y'] += 1000
    for i, v in enumerate(np.unique(eegdata['y'])):
        eegdata['y'][eegdata['y'] == v] = i

    return eegdata
