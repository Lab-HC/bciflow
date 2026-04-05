'''
Description
-----------
This module implements the Continuous Wavelet Transform (CWT) using the Morlet wavelet. 
The CWT is a time-frequency analysis tool that decomposes a signal into wavelets, 
providing localized frequency information over time.

Function
-----------
'''
import numpy as np
import pywt

def wavelet(eegdata, levels=5, inplace=False):
    '''
    Parameters
    ----------
    eegdata : dict
        A dictionary containing the EEG data, where the key 'X' holds the raw signal.
    levels : int
        The number of decomposition levels (scales) for the wavelet transform.
    inplace : bool
        If False, the input dictionary is copied before modification.

    Returns
    -------
    dict
        The same dictionary passed in parameters, but with the transformed data stored under the key 'X'.
    '''

    # ---- Validation of eegdata ----
    if not isinstance(eegdata, dict):
        raise ValueError("eegdata must be a dictionary.")

    if 'X' not in eegdata:
        raise ValueError("eegdata must contain the key 'X'.")

    # ---- Validation of levels ----
    if not isinstance(levels, int):
        raise ValueError("levels must be an integer.")

    if levels <= 0:
        raise ValueError("levels must be greater than 0.")

    # ---- Validation of X ----
    X = eegdata['X']

    if not isinstance(X, np.ndarray):
        raise ValueError("eegdata['X'] must be a numpy array.")

    if X.ndim != 4:
        raise ValueError(
            "eegdata['X'] must have 4 dimensions (trials, channels, samples, segments)."
        )

    if X.shape[-1] <= 0:
        raise ValueError("The last dimension of X must contain signal samples.")

    if levels > X.shape[-1]:
        raise ValueError(
            "levels cannot be greater than the number of samples in the signal."
        )

    if not inplace:
        eegdata = eegdata.copy()
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    widths = np.arange(1, levels+1)
    X_ = []
    for signal_ in range(X.shape[0]):
        coef_, freqs_ = pywt.cwt(X[signal_], widths, 'morl')
        X_.append(coef_)

    X_ = np.array(X_)
    X_ = X_.reshape(eegdata['X'].shape[0], eegdata['X'].shape[2], eegdata['X'].shape[1]*levels , eegdata['X'].shape[3])
    X_ = np.transpose(X_, (0, 2, 1, 3))
    eegdata['X'] = X_

    return eegdata