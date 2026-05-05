'''
Description
-----------
This module implements cubic resampling for EEG data. 
The `cubic_resample` function uses cubic splines to resample the input 
signals to a new sampling frequency, providing smooth interpolation between data points.

Function
-----------
'''
import numpy as np
from scipy.interpolate import CubicSpline

def cubic_resample(eegdata, new_sfreq,inplace=False):
    '''
    Parameters
    ----------
    eegdata : dict
        A dictionary containing the EEG data, where the key 'X' 
        holds the raw signal and 'sfreq' holds the original sampling frequency.
    new_sfreq : float
        The new sampling frequency to which the data will be resampled.
    inplace : bool
        If False, the input dictionary is copied before modification.

    Returns
    -------
    dict
        The same dictionary passed in parameters, but with the resampled data stored under the key 'X' and the new sampling frequency under the key 'sfreq'.
    '''

    # -------- Validate eegdata --------
    if not isinstance(eegdata, dict):
        raise ValueError("eegdata must be a dictionary.")

    if 'X' not in eegdata:
        raise ValueError("eegdata must contain the key 'X'.")

    if 'sfreq' not in eegdata:
        raise ValueError("eegdata must contain the key 'sfreq'.")

    # -------- Validate X --------
    X = eegdata['X']

    if not isinstance(X, np.ndarray):
        raise ValueError("eegdata['X'] must be a numpy array.")

    if X.ndim < 2:
        raise ValueError("eegdata['X'] must have at least 2 dimensions.")

    if X.shape[-1] <= 1:
        raise ValueError("The last dimension of X must contain signal samples.")

    # -------- Validate sfreq --------
    sfreq = eegdata['sfreq']

    if not isinstance(sfreq, (int, float)):
        raise ValueError("sfreq must be numeric.")

    if sfreq <= 0:
        raise ValueError("sfreq must be greater than zero.")

    # -------- Validate new_sfreq --------
    if not isinstance(new_sfreq, (int, float)):
        raise ValueError("new_sfreq must be numeric.")

    if new_sfreq <= 0:
        raise ValueError("new_sfreq must be greater than zero.")

    if new_sfreq > sfreq:
        raise ValueError("new_sfreq cannot be greater than the original sfreq.")

    if sfreq % new_sfreq != 0:
        raise ValueError("sfreq must be divisible by new_sfreq.")

    divisor = int(sfreq // new_sfreq)

    if X.shape[-1] // divisor <= 1:
        raise ValueError("Resampling would result in an invalid signal length.")

    # -------- Copy logic --------
    if not inplace:
        eegdata = eegdata.copy()
    X = eegdata['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
    sfreq = eegdata['sfreq']
    divisor = sfreq//new_sfreq
    duration = X.shape[-1]/sfreq
    old_times = np.arange(0, duration, 1./sfreq)
    new_times = np.arange(0, duration, 1./new_sfreq)
    X_ = []
    for signal_ in range(X.shape[0]):
                cubic_spline = CubicSpline(old_times, X[signal_])
                new_signal = cubic_spline(new_times)
                X_.append(new_signal)

    X_ = np.array(X_)
    X_ = X_.reshape(*eegdata['X'].shape[:-1],eegdata['X'].shape[-1]//divisor )

    eegdata['X'] = X_
    eegdata['sfreq'] = new_sfreq

    return eegdata