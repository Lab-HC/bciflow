"""Function to separate raw data into folds for cross-validation."""

import numpy as np

def separate_folds(raw_data, folds=5, idx_fold=0, gap=2):
    """Separate raw data into folds for cross-validation.
    
    Parameters
    ----------
    raw_data : dict
        Raw data in the format of bciflow.
    folds : int, optional
        Number of folds, by default 5.
    idx_fold : int, optional
        Index of the fold to be used as test set, by default 0.
    gap : float, optional
        Gap in seconds between train and test sets, by default 2.
    
    Returns
    -------
    tuple
        Tuple containing the training and test data.
    """

    gap = int(gap * raw_data['sfreq'])

    t_size = raw_data['X'].shape[-1]

    idx_fold = [t_size//folds * idx_fold, t_size//folds * (idx_fold + 1)]

    data_train = raw_data.copy()
    x_before = raw_data['X'][:, :idx_fold[0]-gap]
    x_after = raw_data['X'][:, idx_fold[1]+gap:]
    data_train['X'] = np.concatenate((x_before, x_after), axis=-1)
    y_before = raw_data['y'][:idx_fold[0]-gap]
    y_after = raw_data['y'][idx_fold[1]+gap:]
    data_train['y'] = np.concatenate((y_before, y_after), axis=0)

    data_test = raw_data.copy()
    data_test['X'] = raw_data['X'][:, idx_fold[0]:idx_fold[1]]
    data_test['y'] = raw_data['y'][idx_fold[0]:idx_fold[1]]

    return data_train, data_test
