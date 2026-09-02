"""Module for flattening features in the BCIFlow pipeline."""

import numpy as np

def __set_default_flatting_params(
        eegdata: dict[str, any]
    ):

    return (
            eegdata
        )

def __check_flatting_params(
            eegdata: dict[str, any]
        ):

    (
            eegdata
        ) = __set_default_flatting_params(
                eegdata
            )

    return (
            eegdata
        )

def flatting(
        eegdata: dict[str, any]
    ) -> dict[str, any]:
    """Flatten the features in the EEG data.
    
    Parameters
    ----------
    eegdata : dict[str, any]
        The EEG data to be flattened.
    Returns
    -------
    dict[str, any]
        The flattened EEG data.
    """

    (
            eegdata
        ) = __check_flatting_params(
                eegdata
            )

    x = eegdata['X'].copy()
    bands = int(np.prod(x.shape) / x.shape[0])
    x = x.reshape(x.shape[0], bands)

    eegdata['X-shape'] = ['flat-features']
    eegdata['X'] = x

    return eegdata
