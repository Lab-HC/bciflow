'''
Description
-----------
This module implements a filter bank for EEG data. The `filterbank` 
function applies multiple bandpass filters to the input signal, 
allowing for the extraction of frequency-specific features.

Function
------------
'''
import numpy as np
from bciflow.modules.tf.bandpass.convolution import bandpass_conv
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII

def filterbank(eegdata, low_cut=[4,8,12,16,20,24,28,32,36], high_cut=[8,12,16,20,24,28,32,36,40], kind_bp='conv', inplace=False, **kwargs):
    '''
    Parameters
    ----------
    eegdata : dict
        A dictionary containing the EEG data, where the key 'X' holds the raw signal.
    low_cut : int or list
        A list of lower cutoff frequencies for each bandpass filter.
    high_cut : int or list
        A list of upper cutoff frequencies for each bandpass filter.
    kind_bp : str
        The type of bandpass filter to use. Options are 'conv' (convolution-based) 
        and 'chebyshevII' (Chebyshev Type II filter)
    kwargs : dict
        Additional arguments to be passed to the filter function.

    Returns
    -------
    output : dict
        The original dictionary with the filtered data stored under the key 'X'.
    '''

    # -------- Validate eegdata --------
    if not isinstance(eegdata, dict):
        raise ValueError("eegdata must be a dictionary.")

    if 'X' not in eegdata:
        raise ValueError("eegdata must contain the key 'X'.")

    X = eegdata['X']

    # -------- Validate X --------
    if not isinstance(X, np.ndarray):
        raise ValueError("eegdata['X'] must be a numpy array.")

    if X.ndim != 4:
        raise ValueError(
            "eegdata['X'] must have 4 dimensions (trials, bands, electrodes, samples)."
        )

    if X.shape[-1] <= 1:
        raise ValueError("The signal must contain more than one sample.")

    # verify if the data has only one band
    if X.shape[1] != 1:
        raise ValueError("The input data must have only one band.")

    # -------- Validate cutoff lists --------
    if not isinstance(low_cut, (list, tuple, np.ndarray)):
        raise ValueError("low_cut must be a list or array.")

    if not isinstance(high_cut, (list, tuple, np.ndarray)):
        raise ValueError("high_cut must be a list or array.")

    if len(low_cut) == 0:
        raise ValueError("low_cut cannot be empty.")

    if len(low_cut) != len(high_cut):
        raise ValueError("low_cut and high_cut must have the same length.")

    # validate frequencies
    for lc, hc in zip(low_cut, high_cut):
        if not isinstance(lc, (int, float)):
            raise ValueError("low_cut values must be numeric.")
        if not isinstance(hc, (int, float)):
            raise ValueError("high_cut values must be numeric.")
        if lc <= 0 or hc <= 0:
            raise ValueError("Cutoff frequencies must be greater than zero.")
        if lc >= hc:
            raise ValueError("Each low_cut must be smaller than the corresponding high_cut.")

    # -------- Validate filter type --------
    valid_filters = ['conv', 'chebyshevII']
    if kind_bp not in valid_filters:
        raise ValueError(f"kind_bp must be one of {valid_filters}.")

    # -------- Copy logic --------
    if not inplace:
        eegdata = eegdata.copy()
        
    X = eegdata['X'].copy()
   
    X_ = []
    for trial_ in range(X.shape[0]):
        X_.append([])
        for i in range(len(low_cut)):
            eegdata_ = eegdata.copy()
            eegdata_['X'] = np.array([X[trial_]])
            if kind_bp == 'conv':
                X__ = bandpass_conv(eegdata_, 
                                    low_cut=low_cut[i], 
                                    high_cut=high_cut[i],
                                    kind='same',
                                    **kwargs)['X'][0][0]
                X_[-1].append(X__)
            elif kind_bp == 'chebyshevII':
                X__ = chebyshevII(eegdata_,
                                    low_cut=low_cut[i], 
                                    high_cut=high_cut[i],
                                    **kwargs)['X'][0][0]
                X_[-1].append(X__)

    X_ = np.array(X_)
    if not inplace:
        eegdata = eegdata.copy()
    eegdata['X'] = X_

    return eegdata