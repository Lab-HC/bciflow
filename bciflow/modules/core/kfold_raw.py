"""Module for performing k-fold cross-validation on raw EEG data."""

import inspect
import numpy as np
import pandas as pd
from .util_epochs import apply_func_to_trials
from .util_raw import raw_to_epochs, separate_folds

def __set_default_kfold_params(
        target,
        pos_folding,
        window_size,
        step_size,
        pre_folding,
        random_state
    ):

    if pre_folding is None:
        pre_folding = {}

    return (
            target,
            pos_folding,
            window_size,
            step_size,
            pre_folding,
            random_state
        )

def __check_kfold_params(
        target,
        pos_folding,
        window_size,
        step_size,
        pre_folding,
        random_state
    ):

    (
            target,
            pos_folding,
            window_size,
            step_size,
            pre_folding,
            random_state
        ) = __set_default_kfold_params(
                target,
                pos_folding,
                window_size,
                step_size,
                pre_folding,
                random_state
            )

    return (
            target,
            pos_folding,
            window_size,
            step_size,
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

def kfold_raw(
        target: dict[str, any],
        pos_folding: dict[str, tuple],
        window_size: float = 2,
        step_size: tuple = (0.5, 0.5),
        pre_folding: dict[str, tuple] | None = None,
        random_state: int = 42
    ) -> pd.DataFrame:
    """
    Perform k-fold cross-validation on raw EEG data.
    
    Parameters
    ----------
    target : dict
        Raw EEG data in the format of bciflow.
    pos_folding : dict
        Dictionary of post-folding functions and their parameters.
    window_size : float, optional
        Size of the window in seconds, by default 2.
    step_size : tuple, optional
        Step size for the sliding window in seconds, by default (0.5, 0.5).
    pre_folding : dict, optional
        Dictionary of pre-folding functions and their parameters, by default None.
    random_state : int, optional
        Random state for reproducibility, by default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the true labels and predicted probabilities for each class.
    """

    (
            target,
            pos_folding,
            window_size,
            step_size,
            pre_folding,
            random_state
        ) = __check_kfold_params(
                target,
                pos_folding,
                window_size,
                step_size,
                pre_folding,
                random_state
            )

    y_pred = []
    _y_pred = []
    y_true = []
    _y_true = []

    for idx_fold in range(5):

        target_train, target_test = separate_folds(
                raw_data=target,
                folds=5,
                idx_fold=idx_fold,
                gap=window_size
            )

        target_train = raw_to_epochs(
                raw_data=target_train,
                window_size=window_size,
                step_size=step_size[0]
            )

        target_test = raw_to_epochs(
                raw_data=target_test,
                window_size=window_size,
                step_size=step_size[1]
            )

        for name, pre_func in pre_folding.items():

            if inspect.isfunction(pre_func[0]):
                target_train = apply_func_to_trials(
                        data=target_train,
                        func=pre_func[0],
                        func_param=pre_func[1],
                        new_data_values={}
                    )
                target_test = apply_func_to_trials(
                        data=target_test,
                        func=pre_func[0],
                        func_param=pre_func[1],
                        new_data_values={}
                    )
            else:
                target_train = apply_func_to_trials(
                        data=target_train,
                        func=pre_func[0].transform,
                        func_param=pre_func[1],
                        new_data_values={}
                    )
                target_test = apply_func_to_trials(
                        data=target_test,
                        func=pre_func[0].transform,
                        func_param=pre_func[1],
                        new_data_values={}
                    )

        for name, pro_func in pos_folding.items():

            if name != 'clf':
                if inspect.isfunction(pro_func[0]):
                    target_train = apply_func_to_trials(
                            data=target_train,
                            func=pro_func[0],
                            func_param=pro_func[1],
                            new_data_values={}
                        )
                    target_test = apply_func_to_trials(
                            data=target_test,
                            func=pro_func[0],
                            func_param=pro_func[1],
                            new_data_values={}
                        )
                else:
                    target_train = pro_func[0].fit_transform(
                            target_train, **pro_func[1]
                        )

                    target_test = pro_func[0].fit_transform(
                            target_test, **pro_func[1]
                        )

            else:
                clf, clf_param = pro_func
                if not inspect.isfunction(clf):
                    clf = clf.fit(
                            target_train['X'],
                            target_train['y'],
                            **clf_param
                        )
                    _y_pred = clf.predict_proba(target_test['X'])
                    _y_pred = np.round(_y_pred, 4)
                    _y_true = target_test['y']

        y_pred.append(_y_pred)
        y_true.append(_y_true)

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    results = pd.DataFrame(y_pred)
    print(target['y_dict'])

    y_inv = {v: k for k, v in target['y_dict'].items()}
    columns = [y_inv[int(i)] for i in range(results.shape[1])]
    results.columns = columns
    results.insert(
           0, 'true_label', [y_inv[int(i)] for i in y_true]
        )

    return results
