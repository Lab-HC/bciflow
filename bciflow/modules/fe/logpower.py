"""Module for extracting log-power features in the BCIFlow pipeline."""

import numpy as np

def __set_default_logpower_params(
        eegdata: dict[str, any]
    ):

    return (
            eegdata
        )

def __check_logpower_params(
            eegdata: dict[str, any]
        ):

    (
            eegdata
        ) = __set_default_logpower_params(
                eegdata
            )

    return (
            eegdata
        )

def logpower(
        eegdata: dict[str, any]
    ) -> dict[str, any]:
    """Extract log-power features from the EEG data.
    
    Parameters
    ----------
    eegdata : dict
        A dictionary containing the EEG data and metadata.
    Returns
    -------
    dict[str, any]
        The EEG data with log-power features extracted.
    """

    (
            eegdata
        ) = __check_logpower_params(
                eegdata
            )

    x = eegdata['X'].copy()
    oldshape = x.shape
    trial_bands = int(np.prod(x.shape) / x.shape[-1])
    x = x.reshape(trial_bands, x.shape[-1])

    _x = []
    for signal_ in range(x.shape[0]):
        filtered = np.log(np.mean(x[signal_]**2))
        _x.append(filtered)

    _x = np.array(_x)

    _x = _x.reshape(*oldshape[:-1])

    eegdata['X-shape'] = eegdata['X-shape'][:-1]
    eegdata['X'] = _x

    return eegdata
