from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import inspect
from ..core.util import util
from ..core.posfold import *
from ..core.prefold import *
from typing import Dict, Any, List, Optional

def classification_kfold(target: Dict[str, Any],
          start_window: float or list,
          start_test_window: Optional[float or list] = None,
          pre_folding: Optional[Dict[str, tuple]] = None,
          pos_folding: Dict[str, tuple] = {},
          window_size: float = 1.0) -> pd.DataFrame:
    '''
    This method performs a complete stratified k-fold cross-validation
    pipeline for EEG classification using a modular pre-fold and post-fold
    architecture.

    The method is responsible for:
    - Normalizing window parameters.
    - Executing pre-fold operations (cropping and preprocessing).
    - Performing stratified 5-fold cross-validation.
    - Executing post-fold operations (train/test split, feature
      transformation, classifier training and prediction).
    - Returning the prediction results in a structured pandas DataFrame.

    Parameters
    ----------
    target : dict
        Input EEG data in dictionary format. The dictionary must contain:
        - 'X': EEG data as a numpy array (trials × channels × samples) or (trials × bands × channels × samples).
        - 'y': Labels corresponding to each trial.
        - 'sfreq': Sampling frequency.
        - 'y_dict': Dictionary mapping class labels to integers.
        - 'events': Event markers.
        - 'ch_names': Channel names.
        - 'tmin': Initial time reference.

    start_window : float or list
        Starting time(s) used to construct the training windows.
        If a single float is provided, it is converted to a list.

    start_test_window : float or list, optional
        Starting time(s) used to construct the test windows.
        If None, it defaults to start_window.
        If a single float is provided, it is converted to a list.

    pre_folding : dict, optional
        Dictionary containing preprocessing functions applied before
        cross-validation.

        Each key represents a preprocessing step name.
        Each value must be a tuple:
            (function_or_object, parameters_dict)

        These operations are applied trial-wise before the fold split.

    pos_folding : dict
        Dictionary containing post-processing transformations and
        classifier definition.

        Each key represents a transformation step.
        Each value must be a tuple:
            (transform_function_or_object, parameters_dict)

        The key 'clf' is reserved for the classifier and must contain:
            (classifier_object, classifier_parameters_dict)

    window_size : float, default=1.0
        Size (in seconds) of the time window used during cropping.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame containing cross-validation results with columns:
        - 'fold': Fold identifier (1–5).
        - 'tmin': Test window starting time.
        - 'true_label': True class label (string).
        - One column per class containing predicted probabilities.

    Raises
    ------
    ValueError
        If window parameters are invalid.

    KeyError
        If 'clf' is not defined in pos_folding.

    Notes
    -----
    - A StratifiedKFold strategy with 5 splits is used to preserve
      class distribution across folds.
    - Preprocessing is executed before the split.
    - All fitting operations (feature transforms and classifier) are
      performed strictly on the training set to prevent data leakage.
    - The classifier is expected to implement `fit` and `predict_proba`.

    Example
    -------
    Applying classification k-fold on EEG data:

    >>> from bciflow.modules.core.kfold import classification_kfold
    >>> import numpy as np
    >>> target = {
    ...     'X': np.random.rand(100, 64, 256),
    ...     'y': np.random.randint(0, 2, size=100),
    ...     'sfreq': 256,
    ...     'y_dict': {0: 'class_0', 1: 'class_1'},
    ...     'events': {'event_1': [0, 50], 'event_2': [51, 100]},
    ...     'ch_names': [f'ch_{i}' for i in range(64)],
    ...     'tmin': -0.5
    ... }
    >>> results = classification_kfold(
    ...     target=target,
    ...     start_window=0.0,
    ...     start_test_window=0.5,
    ...     pre_folding={},
    ...     pos_folding={
                'csp': (CSP(), {}),
    ...         'clf': (LDA(), {})
    ...     }
    ... )
    >>> print(results.head())
    '''

    # target
    if type(target) != dict:
        raise ValueError("target has to be a dict type value")

    required_keys = ['X', 'y', 'sfreq', 'y_dict', 'events', 'ch_names', 'tmin']
    for key in required_keys:
        if key not in target:
            raise ValueError(f"target dictionary must contain the key '{key}'")

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

    # pre_folding
    if pre_folding is not None:
        if type(pre_folding) != dict:
            raise ValueError("pre_folding has to be a dict type value")

        for name, value in pre_folding.items():
            if type(value) != tuple or len(value) != 2:
                raise ValueError("Each pre_folding entry must be a tuple (function, params_dict)")

            if not callable(value[0]):
                raise ValueError(f"Pre-folding function '{name}' is not callable")

            if type(value[1]) != dict:
                raise ValueError(f"Parameters of pre-folding '{name}' must be a dict")

    # pos_folding
    if type(pos_folding) != dict:
        raise ValueError("pos_folding has to be a dict type value")

    if 'clf' not in pos_folding:
        raise ValueError("pos_folding must contain the key 'clf'")

    for name, value in pos_folding.items():
        if type(value) != tuple or len(value) != 2:
            raise ValueError("Each pos_folding entry must be a tuple (function, params_dict)")

        if not callable(value[0]) and not hasattr(value[0], 'fit'):
            raise ValueError(f"Post-folding function '{name}' must be callable or have a fit method")

        if type(value[1]) != dict:
            raise ValueError(f"Parameters of pos_folding '{name}' must be a dict")

    # window_size
    if type(window_size) not in [float, int]:
        raise ValueError("window_size has to be a float type value")

    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")


    if type(start_window) is float:
        start_window = [start_window]

    if start_test_window is None:
        start_test_window = start_window
    elif type(start_test_window) is float:
        start_test_window = [start_test_window]

    # =========================
    # PRE-FOLD
    # =========================
    target_dict = {}
    if pre_folding is None:
        pre_folding = {}

   

    
    # =========================
    # Crop
    # =========================

    for tmin_ in start_test_window:
        target_dict[tmin_] = util.crop(
            data=target,
            tmin=tmin_,
            window_size=window_size,
            inplace=False
        )

    target_dict = apply_prefold(
        target=target,
        start_window=start_window,
        start_test_window=start_test_window,
        window_size=window_size,
        pre_folding=pre_folding,
    )

            
    # =========================
    # K-FOLD
    # =========================

    
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_id = 0
    results = []

    for train_index, test_index in skf.split(target["y"], target["y"]):

        fold_id += 1
        # =========================
        # POS-FOLD
        # =========================
        results = apply_posfold(
            target_dict=target_dict,
            train_index=train_index,
            test_index=test_index,
            start_window=start_window,
            start_test_window=start_test_window,
            pos_folding=pos_folding,
            target=target,
            fold_id=fold_id,
            results=results,
        )

    results = np.array(results)

    results = pd.DataFrame(
        results,
        columns=[
            'fold',
            'tmin',
            'true_label',
            *target['y_dict'].keys()
        ]
    )

    return results