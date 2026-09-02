"""Module for the Chebyshev Type II filter in the BCIFlow pipeline.
"""

import numpy as np
from scipy.signal import cheby2, filtfilt

def __set_default_chebyshevii_params(
        eegdata,
        low_cut,
        high_cut,
        order
    ):

    if isinstance(low_cut, (int, float)):
        low_cut = [low_cut]
    if isinstance(high_cut, (int, float)):
        high_cut = [high_cut]

    return (
            eegdata,
            low_cut,
            high_cut,
            order
        )

def __check_chebyshevii_params(
            eegdata,
            low_cut,
            high_cut,
            order
        ):

    (
            eegdata,
            low_cut,
            high_cut,
            order
        ) = __set_default_chebyshevii_params(
                eegdata,
                low_cut,
                high_cut,
                order
            )

    return (
            eegdata,
            low_cut,
            high_cut,
            order
        )

def __set_btype(low_cut, high_cut):

    btype = []

    for low, high in zip(low_cut, high_cut):
        if low is None and high is None:
            raise ValueError(
                    'Invalid cut-off frequencies. ' \
                    'Both low_cut and high_cut cannot be None.'
                )
        elif low is not None and high is None:
            btype.append('highpass')
        elif low is None and high is not None:
            btype.append('lowpass')
        else:
            btype.append('bandpass')

    return btype

def chebyshevii(
        eegdata: dict[str, any],
        low_cut: float | list[float] = 4,
        high_cut: float | list[float] = 40,
        order: int = 4,
    ):
    """Apply a Chebyshev Type II filter to the EEG data.
    
    Parameters
    ----------
    eegdata : dict
        A dictionary containing the EEG data and metadata.
    low_cut : float or list of float, optional
        The low cut-off frequency/frequencies for the filter.
        If a single float is provided, it will be applied to all
        bands. If a list is provided, it should have the same
        length as high_cut. Default is 4 Hz.
    high_cut : float or list of float, optional
        The high cut-off frequency/frequencies for the filter.
        If a single float is provided, it will be applied to all
        bands. If a list is provided, it should have the same
        length as low_cut. Default is 40 Hz.
    order : int, optional
        The order of the filter. Default is 4.

    Returns
    -------
    dict
        The EEG data with the Chebyshev Type II filter applied.
    """

    (
            eegdata,
            low_cut,
            high_cut,
            order
        ) = __check_chebyshevii_params(
                eegdata,
                low_cut,
                high_cut,
                order
            )


    wn = [low_cut, high_cut]
    btype = __set_btype(low_cut, high_cut)
    rs = [40 if b == 'bandpass' else 20 for b in btype]

    x = eegdata['X'].copy()
    oldshape = x.shape
    trial_bands = int(np.prod(x.shape) / x.shape[-1])
    x = x.reshape(trial_bands, x.shape[-1])

    x_new = []
    for band, band_type in enumerate(btype):
        x_new.append([])
        for signal_ in range(x.shape[0]):
            if band_type == 'bandpass':
                wn_temp = [
                        wn[0][band],
                        wn[1][band]
                    ]
            elif band_type == 'lowpass':
                wn_temp = float(wn[1][band])
            else:
                wn_temp = float(wn[0][band])

            filtered = filtfilt(
                    *cheby2(
                            order,
                            rs[band],
                            wn_temp,
                            btype=band_type,
                            fs=eegdata['sfreq']
                        ),
                    x[signal_]
                )
            x_new[-1].append(filtered)

    x_new = np.array(x_new)
    x_new = np.swapaxes(x_new, 0, 1)
    if len(btype) == 1:
        x_new = x_new.reshape(oldshape)
    else:
        x_new = x_new.reshape(*oldshape[:-1], len(btype), oldshape[-1])
        x_new = np.swapaxes(x_new, -2, -3)
        eegdata['X-shape'].insert(-2, 'cheby-bands')

    eegdata['X'] = x_new

    return eegdata
