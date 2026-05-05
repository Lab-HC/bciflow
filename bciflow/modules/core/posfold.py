import numpy as np
import pandas as pd
import inspect
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold
from ..core.util import util

def apply_posfold(target_dict: Dict[float, Dict[str, Any]],
                  train_index,
                  test_index,
                  start_window: float or list, 
                  start_test_window: Optional[float or list],
                  pos_folding: Dict[str, tuple],
                  target: Dict[str, Any],
                  fold_id: int,
                  results: list) -> list:
    '''
    This method performs the post-fold operations within a stratified
    k-fold cross-validation procedure.

    The method is responsible for:
    - Splitting the data into training and testing sets for the current fold.
    - Applying post-processing transformations (feature extraction, 
      spatial filtering, normalization, etc.).
    - Training the classifier using only the training data.
    - Generating predictions for the test data.
    - Appending the results of the current fold to the results list.

    Parameters
    ----------
    target_dict : dict
        Dictionary containing cropped and preprocessed EEG data for each
        time window. The keys correspond to the tmin values and the values
        are eegdata dictionaries.
    
    train_index : array-like
        Indices of the training samples for the current fold.

    test_index : array-like
        Indices of the test samples for the current fold.

    start_window : list
        List of starting times (tmin) used to construct the training set.
        Each window is concatenated to form the final training data.

    start_test_window : list
        List of starting times (tmin) used to construct the test set.
        Predictions are generated independently for each test window.

    pos_folding : dict
        Dictionary containing post-processing functions and classifier.
        The keys are the names of the transformations.
        Each value must be a tuple containing:
            (transform_function_or_object, parameters_dict)

        The key 'clf' is reserved for the classifier and must contain:
            ('classifier_object', classifier_parameters_dict)

    target : dict
        Original EEG dictionary containing:
        - 'X': EEG data
        - 'y': Labels
        - 'y_dict': Label mapping dictionary

    fold_id : int
        Identifier of the current fold.

    results : list
        List used to accumulate the prediction results across folds.

    Returns
    -------
    results : list
        Updated results list including:
        - fold id
        - tmin
        - true label (string format)
        - predicted probabilities for each class

    Raises
    ------
    KeyError
        If 'clf' is not defined in pos_folding.

    ValueError
        If any transformation in pos_folding is improperly formatted.

    Notes
    -----
    - All transformations (except the classifier) are fitted exclusively
      on the training set to prevent data leakage.
    - The classifier must implement `fit` and preferably `predict_proba`.
    - If `predict_proba` is unavailable, zero probabilities are assigned.

    Example
    -------
    Applying post-fold operations inside a k-fold loop:

    >>> results = []
    >>> results = apply_posfold(
    ...     target_dict=target_dict,
    ...     train_index=train_idx,
    ...     test_index=test_idx,
    ...     start_window=[0.0],
    ...     start_test_window=[0.5],
    ...     pos_folding={
    ...         'csp': (CSP(), {}),
    ...         'clf': (LDA(), {})
    ...     },
    ...     target=target,
    ...     fold_id=1,
    ...     results=results
    ... )
    >>> print(len(results))
    '''
    # target_dict
    if type(target_dict) != dict:
        raise ValueError("target_dict has to be a dict type value")

    for key, value in target_dict.items():
        if type(key) not in [float, int]:
            raise ValueError("target_dict keys must be float (tmin values)")
        if type(value) != dict:
            raise ValueError("target_dict values must be eegdata dictionaries")

    # train_index
    if not hasattr(train_index, "__iter__"):
        raise ValueError("train_index must be an iterable")

    # test_index
    if not hasattr(test_index, "__iter__"):
        raise ValueError("test_index must be an iterable")

    # start_window
    if type(start_window) not in [float, list]:
        raise ValueError("start_window has to be a float or list of floats")

    if type(start_window) == list:
        for value in start_window:
            if type(value) not in [float, int]:
                raise ValueError("start_window list must contain only float values")

    # start_test_window
    if start_test_window is not None:
        if type(start_test_window) not in [float, list]:
            raise ValueError("start_test_window has to be a float, list of floats or None")

        if type(start_test_window) == list:
            for value in start_test_window:
                if type(value) not in [float, int]:
                    raise ValueError("start_test_window list must contain only float values")

    # pos_folding
    if type(pos_folding) != dict:
        raise ValueError("pos_folding has to be a dict type value")

    if 'clf' not in pos_folding:
        raise ValueError("pos_folding must contain the key 'clf'")

    for name, value in pos_folding.items():

        if type(value) != tuple or len(value) != 2:
            raise ValueError("Each pos_folding entry must be a tuple (function_or_object, params_dict)")

        if not callable(value[0]) and not hasattr(value[0], 'fit'):
            raise ValueError(f"Post-folding step '{name}' must be callable or implement fit()")

        if type(value[1]) != dict:
            raise ValueError(f"Parameters of pos_folding '{name}' must be a dict")

    # target
    if type(target) != dict:
        raise ValueError("target has to be a dict type value")

    required_keys = ['X', 'y', 'y_dict']
    for key in required_keys:
        if key not in target:
            raise ValueError(f"target dictionary must contain the key '{key}'")

    if type(target['y_dict']) != dict:
        raise ValueError("target['y_dict'] must be a dict")

    # fold_id
    if type(fold_id) != int:
        raise ValueError("fold_id has to be an int type value")

    if fold_id <= 0:
        raise ValueError("fold_id must be greater than zero")

    # results
    if type(results) != list:
        raise ValueError("results has to be a list type value")
    

    if type(start_window) is float:
        start_window = [start_window]

    if start_test_window is None:
        start_test_window = start_window
    elif type(start_test_window) is float:
        start_test_window = [start_test_window]

    # =========================
    # Train set
    # =========================
    target_train = []

    for tmin_ in start_window:
        target_train.append(
            util.get_trial(
                data=target_dict[tmin_],
                ids=train_index
            )
        )

    target_train = util.concatenate(target_train)

    # =========================
    # Test set
    # =========================
    target_test = {}

    for tmin_ in start_test_window:
        target_test[tmin_] = util.get_trial(
            data=target_dict[tmin_],
            ids=test_index
        )

    # =========================
    # Post-processing transforms
    # =========================
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

    # =========================
    # Classifier
    # =========================
    clf, clf_param = pos_folding['clf']

    if not inspect.isfunction(clf):
        clf = clf.fit(
            target_train['X'],
            target_train['y'],
            **clf_param
        )

    # =========================
    # Prediction
    # =========================
    for tmin_ in start_test_window:

        try:
            y_pred = clf.predict_proba(
                target_test[tmin_]['X']
            )
        except:
            y_pred = np.zeros(
                (len(target_test[tmin_]['y']),
                 len(target['y_dict']))
            )

        y_pred = np.round(y_pred, 4)

        for trial_ in range(len(y_pred)):
            results.append([
                fold_id,
                tmin_,
                util.find_key_with_value(
                    target['y_dict'],
                    target_test[tmin_]['y'][trial_]
                ),
                *y_pred[trial_]
            ])

    return results
