'''
Description
-----------
This module implements the Average Power Spectral Density (APSD) feature extractor, 
which computes the average power in specific frequency bands of EEG signals. 
This feature is commonly used in BCI applications to characterize brain activity.

The APSD is calculated using Welch's method, which estimates the power spectral density 
by dividing the signal into overlapping segments and averaging their periodograms.

Class
------------
'''
import numpy as np
import scipy as sp

class apsd:
    '''
    Attributes
    ----------
    flating : bool
        If True, the output data is returned in a flattened format (default is False).
    '''

    def __init__(self, flating: bool = False):
        ''' Initializes the class.
        
        Parameters
        ----------
        flating : bool, optional
            If True, the output data is returned in a flattened format (default is False).
        
        Returns
        -------
        None
        '''

        # -------- Validate flating --------
        if not isinstance(flating, bool):
            raise ValueError("flating must be a boolean value.")

        self.flating = flating

    def fit(self, eegdata):
        ''' 
        This method does nothing, as the APSD feature extractor does not require training.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
            
        Returns
        -------
        self
        '''
        # -------- Validate eegdata --------
        if not isinstance(eegdata, dict):
            raise ValueError("eegdata must be a dictionary.")

        if 'X' not in eegdata:
            raise ValueError("eegdata must contain the key 'X'.")

        return self

    def transform(self, eegdata) -> dict:
        '''
        This method computes the average power spectral density (APSD) for each trial, band, 
        and channel in the input data. The result is stored in the dictionary under the key 'X'.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
        
        Returns
        -------
        output : dict
            The transformed data.
        '''
        
        # -------- Validate eegdata --------
        if not isinstance(eegdata, dict):
            raise ValueError("eegdata must be a dictionary.")

        if 'X' not in eegdata:
            raise ValueError("eegdata must contain the key 'X'.")

        X = eegdata['X'].copy()

        # -------- Validate X --------
        if not isinstance(X, np.ndarray):
            raise ValueError("eegdata['X'] must be a numpy array.")

        if X.ndim not in [3, 4]:
            raise ValueError(
                "X must have shape (bands, channels, samples) or "
                "(trials, bands, channels, samples)."
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        # -------- Prepare data --------
        many_trials = len(X.shape) == 4
        if not many_trials:
            X = X[np.newaxis, :, :, :]

        output = []
        trials_, bands_, channels_, _ = X.shape

        for trial_ in range(trials_):
            output.append([])
            for band_ in range(bands_):
                output[trial_].append([])
                for channel_ in range(channels_):
                    psd = sp.signal.welch(X[trial_, band_, channel_])
                    output[trial_][band_].append(np.mean(psd))

        output = np.array(output)
        
        if self.flating:
            output = output.reshape(output.shape[0], -1)

        if not many_trials:
            output = output[0]
        eegdata['X'] = output
        return eegdata

    def fit_transform(self, eegdata) -> dict:
        '''
        This method combines fitting and transforming into a single step. It returns a 
        dictionary with the transformed data.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
          
        Returns
        -------
        output : dict
            The transformed data.
        '''
        return self.fit(eegdata).transform(eegdata)