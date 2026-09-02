"""Module for performing k-fold cross-validation on EEG data."""

import inspect
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from .util_epochs.crop_epochs import crop_epochs
from .util_epochs.apply_func_to_trials import apply_func_to_trials
from .util_epochs.get_trials import get_trials
from .util_epochs.concatenate import concatenate

def __set_default_kfold_params(
        target,
        pos_folding,
        start_window,
        start_test_window,
        window_size,
        pre_folding,
        random_state
    ):

    if start_window is None:
        start_window = 0
    if start_test_window is None:
        start_test_window = start_window
    if pre_folding is None:
        pre_folding = {}

    if isinstance(start_window, (int, float)):
        start_window = [start_window]
    if isinstance(start_test_window, (int, float)):
        start_test_window = [start_test_window]

    return (
            target,
            pos_folding,
            start_window,
            start_test_window,
            window_size,
            pre_folding,
            random_state
        )

def __check_kfold_params(
        target,
        pos_folding,
        start_window,
        start_test_window,
        window_size,
        pre_folding,
        random_state
    ):

    (
            target,
            pos_folding,
            start_window,
            start_test_window,
            window_size,
            pre_folding,
            random_state
        ) = __set_default_kfold_params(
                target,
                pos_folding,
                start_window,
                start_test_window,
                window_size,
                pre_folding,
                random_state
            )

    return (
            target,
            pos_folding,
            start_window,
            start_test_window,
            window_size,
            pre_folding,
            random_state
        )

def __find_key_with_value(
        dictionary,
        i
    ):

    for key, value in dictionary.items():
        if value == i:
            return key
    return None

def kfold_epochs(
        target: dict[str, any],
        pos_folding: dict[str, tuple],
        start_window: float | list[float] = 0,
        start_test_window: list[float] | None = None,
        window_size: float = 2,
        pre_folding: dict[str, tuple] | None = None,
        random_state: int = 42
    ) -> pd.DataFrame:
    """Performs a k-fold cross-validation on the given data.

    Performs a k-fold cross-validation on the given data, applying the 
    specified preprocessing and postprocessing functions, and returns
    a DataFrame with the results.

    Parameters
    ----------
    target : dict[str, any]
        A dictionary containing the data to be used in the k-fold
        cross-validation. The dictionary must contain the
        following keys:
        - "X": A numpy array of shape (n_trials, n_channels, n_times)
        containing the EEG data.
        - "y": A numpy array of shape (n_trials,) containing the labels
        for each trial.
        - "tmin": A float representing the starting time of the epochs
        in seconds.
        - "sfreq": A float representing the sampling frequency of the
        data in Hz.
        - "y_dict": A dictionary mapping label names to their
        corresponding integer values.

    pos_folding : dict[str, tuple]
        A dictionary containing the postprocessing functions to be
        applied after the k-fold split. The keys of the dictionary
        are the names of the functions, and the values are tuples
        containing the function and its parameters. The function can
        be either a callable or an object with a fit_transform method.

    start_window : float | list[float], optional
        A float or a list of floats representing the starting times 
        of the training windows in seconds. If a single float is
        provided, it will be used for all training windows. 
        Default is 0.

    start_test_window : list[float] | None, optional
        A list of floats representing the starting times of the testing
        windows in seconds. If None, the starting times of the training
        windows will be used. Default is None.

    window_size : float, optional
        A float representing the size of the windows in seconds. 
        Default is 2.

    pre_folding : dict[str, tuple] | None, optional
        A dictionary containing the preprocessing functions to be
        applied before the k-fold split. The keys of the dictionary
        are the names of the functions, and the values are tuples
        containing the function and its parameters. The function can
        be either a callable or an object with a fit_transform method.
        Default is None.

    random_state : int, optional
        An integer representing the random state to be used in the 
        StratifiedKFold. Default is 42.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the k-fold cross-validation.
        The DataFrame has the following columns:
        - "fold": The fold number (1 to n_splits).
        - "tmax": The ending time of the testing window in seconds.
        - "true_label": The true label of the trial.
        - One column for each label in target["y_dict"], containing the
        predicted probabilities for each label.
    """


    (
            target,
            pos_folding,
            start_window,
            start_test_window,
            window_size,
            pre_folding,
            random_state
        ) = __check_kfold_params(
                target,
                pos_folding,
                start_window,
                start_test_window,
                window_size,
                pre_folding,
                random_state
            )

    target_dict = {}
    for _tmin in start_test_window:
        target_copy = target.copy()
        target_dict[_tmin] = crop_epochs(
                data=target_copy,
                tmin=_tmin,
                window_size=window_size
            )

    for _tmin in start_test_window:
        for name, pre_func in pre_folding.items():

            if inspect.isfunction(pre_func[0]):
                target_dict[_tmin] = apply_func_to_trials(
                        data=target_dict[_tmin],
                        func=pre_func[0],
                        func_param=pre_func[1],
                        new_data_values={}
                    )
            else:
                target_dict[_tmin] = apply_func_to_trials(
                        data=target_dict[_tmin],
                        func=pre_func[0].transform,
                        func_param=pre_func[1],
                        new_data_values={}
                    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    fold_id = 0
    results = []
    for train_index, test_index in skf.split(target["y"], target["y"]):
        fold_id += 1

        target_train = []
        for tmin_ in start_window:
            target_train.append(
                    get_trials(data=target_dict[tmin_], ids=train_index)
                )
        target_train = concatenate(target_train)

        target_test = {}
        for tmin_ in start_test_window:
            target_test[tmin_] = get_trials(
                    data=target_dict[tmin_], ids=test_index
                )

        for name, pos_func in pos_folding.items():

            if name != 'clf':
                if inspect.isfunction(pos_func[0]):
                    target_train = pos_func[0](target_train, **pos_func[1])
                else:
                    target_train = pos_func[0].fit_transform(
                            target_train, **pos_func[1]
                        )

                for tmin_ in start_test_window:
                    if inspect.isfunction(pos_func[0]):
                        target_test[tmin_] = pos_func[0](
                                target_test[tmin_], **pos_func[1]
                            )
                    else:
                        target_test[tmin_] = pos_func[0].transform(
                                target_test[tmin_]
                            )


        clf, clf_param = pos_folding['clf']
        if not inspect.isfunction(clf):
            clf = clf.fit(target_train['X'], target_train['y'], **clf_param)

        for tmin_ in start_test_window:
            y_pred = clf.predict_proba(target_test[tmin_]['X'])
            y_pred = np.round(y_pred, 4)
            for trial_, y_pred_trial in enumerate(y_pred):
                true_label = __find_key_with_value(
                    target['y_dict'],
                    target_test[tmin_]['y'][trial_],
                )
                results.append(
                    [
                        fold_id,
                        tmin_+window_size,
                        true_label,
                        *y_pred_trial,
                    ]
                )

    results = np.array(results)
    results = pd.DataFrame(results,
            columns=[
                    'fold', 'tmax', 'true_label',
                    *target['y_dict'].keys()
                ]
        )

    return results
