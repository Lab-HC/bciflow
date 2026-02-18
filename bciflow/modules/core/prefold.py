import numpy as np
import pandas as pd
import inspect
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold
from ..core.util import util

def apply_prefold(target: Dict[str, Any],
                  start_window: float or list,
                  start_test_window: Optional[float or list],
                  window_size: float,
                  pre_folding: Dict[str, tuple],
                  ) -> Dict[float, Dict[str, Any]]:
    '''
    This method performs the pre-fold operations before the stratified
    k-fold cross-validation procedure.

    The method is responsible for:
    - Normalizing the window parameters.
    - Cropping the EEG data according to the specified time windows.
    - Applying preprocessing transformations trial-wise.
    - Preparing the data structure that will be used during
      the cross-validation folds.

    Parameters
    ----------
    target : dict
        Input EEG data in dictionary format. The dictionary must contain:
        - 'X': EEG data as a numpy array.
        - 'y': Labels corresponding to the trials.
        - 'sfreq': Sampling frequency.
        - 'y_dict': Label mapping dictionary.
        - 'events': Event markers.
        - 'ch_names': Channel names.
        - 'tmin': Initial time reference.

    start_window : float or list
        Starting time(s) used to define training windows.
        If a single float is provided, it is converted into a list.

    start_test_window : float or list
        Starting time(s) used to define test windows.
        If None, it defaults to start_window.
        If a single float is provided, it is converted into a list.

    window_size : float
        Size (in seconds) of the time window used in the cropping
        procedure.

    pre_folding : dict
        Dictionary containing preprocessing functions applied before
        cross-validation.

        Each key represents the name of the preprocessing step.
        Each value must be a tuple containing:
            (function_or_object, parameters_dict)

        If an object is provided instead of a function, its `.transform`
        method will be applied trial-wise.

    Returns
    -------
    target_dict : dict
        Dictionary where:
        - Keys correspond to each tmin value in start_test_window.
        - Values are cropped and preprocessed EEG dictionaries.

    Raises
    ------
    ValueError
        If window parameters are improperly formatted.

    TypeError
        If preprocessing functions are not callable or improperly defined.

    Notes
    -----
    - All preprocessing operations are applied independently to each trial.
    - No fitting occurs at this stage; this prevents data leakage.
    - The returned structure is later used by the post-fold procedure.

    Example
    -------
    Applying pre-fold preprocessing:

    >>> target_dict = apply_prefold(
    ...     target=target,
    ...     start_window=0.0,
    ...     start_test_window=0.5,
    ...     window_size=1.0,
    ...     pre_folding={
    ...         'bp': (bandpass_conv, {'low': 8, 'high': 30}),
    ...     }
    ... )
    >>> print(list(target_dict.keys()))
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
    
    # window_size
    if type(window_size) not in [float, int]:
        raise ValueError("window_size has to be a float type value")

    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")

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
                

    if type(start_window) is float:
        start_window = [start_window]

    if start_test_window is None:
        start_test_window = start_window
    elif type(start_test_window) is float:
        start_test_window = [start_test_window]

    target_dict = {}

    

    # Crop
    for tmin_ in start_test_window:
        target_dict[tmin_] = util.crop(
            data=target,
            tmin=tmin_,
            window_size=window_size,
            inplace=False
        )


    #TL and DA here <---

    # Apply preprocessing
    for tmin_ in start_test_window:
        for name, pre_func in pre_folding.items():
            
            if inspect.isfunction(pre_func[0]):
                target_dict[tmin_] = util.apply_to_trials(
                    data=target_dict[tmin_],
                    func=pre_func[0],
                    func_param=pre_func[1],
                    inplace=False
                )
                
            else:
                target_dict[tmin_] = util.apply_to_trials(
                    data=target_dict[tmin_],
                    func=pre_func[0].transform,
                    func_param=pre_func[1],
                    inplace=False
                )
    return target_dict