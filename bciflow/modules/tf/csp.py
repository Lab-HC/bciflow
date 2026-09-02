"""Module for the Common Spatial Patterns (CSP) filter in the BCIFlow pipeline."""

import numpy as np
import scipy as sp

def _csp_mini(x1, x2, m_pairs):

    sigma1 = np.zeros((x1.shape[1], x1.shape[1]))
    sigma2 = np.zeros((x2.shape[1], x2.shape[1]))

    for i in range(x1.shape[0]):
        sigma1 += (x1[i] @ x1[i].T)
    for i in range(x2.shape[0]):
        sigma2 += (x2[i] @ x2[i].T)
    sigma1 /= x1.shape[0]
    sigma2 /= x2.shape[0]
    sigma_tot = sigma1 + sigma2
    _, w = sp.linalg.eigh(sigma1, sigma_tot)
    first_aux = w[:, :m_pairs]
    last_aux = w[:, -m_pairs:]
    w = np.concatenate((first_aux, last_aux), axis=1)

    return w

class CSP:
    """Common Spatial Patterns (CSP) filter for EEG data.
    """

    n_electrodes: int
    m_pairs: int
    w: np.ndarray
    bands: int

    def __init__(
            self, m_pairs: int = 2
        ):
        """Initialize the CSP filter.
        
        Parameters
        ----------
        m_pairs : int, optional
            The number of pairs of spatial filters to compute, by default 2.
        """

        self.m_pairs = m_pairs

    def fit(
            self,
            eegdata: dict
        ) -> np.ndarray:
        """Fit the CSP filter to the EEG data.
        
        Parameters
        ----------
        eegdata : dict
            A dictionary containing the EEG data and metadata.
        
        Returns
        -------
        CSP
            The fitted CSP filter.
        """

        x = eegdata['X'].copy()
        y = eegdata['y'].copy()
        y_unique = np.unique(y).astype(int)

        self.n_electrodes = x.shape[-2]
        self.bands = int(np.prod(x.shape) / (x.shape[0] * x.shape[-1] * x.shape[-2]))
        x = x.reshape(x.shape[0], self.bands, self.n_electrodes, x.shape[-1])

        y_pairs = [(
            y_unique[i], y_unique[j])
                for i in range(len(y_unique))
                for j in range(i + 1, len(y_unique)
                )
            ]

        self.w = np.zeros((len(y_pairs), self.bands, self.n_electrodes, self.m_pairs * 2))

        for _pair_idx, _label in enumerate(y_pairs):
            for _band in range(self.bands):
                x_temp1 = x[(y == y_unique[_label[0]])][:, _band]
                x_temp2 = x[(y == y_unique[_label[1]])][:, _band]

                self.w[_pair_idx, _band] = _csp_mini(x_temp1, x_temp2, self.m_pairs)

        return self

    def transform(
            self,
            eegdata: dict
        ) -> dict:

        x = eegdata['X'].copy()
        self.n_electrodes = x.shape[-2]
        self.bands = int(np.prod(x.shape) / (x.shape[0] * x.shape[-1] * x.shape[-2]))
        x = x.reshape(x.shape[0], self.bands, self.n_electrodes, x.shape[-1])

        eegdata['X-shape'][-2] = 'csp-features'
        if len(self.w) != 1:
            eegdata['X-shape'].insert(-2, 'csp-bands')

        new_x = []
        for _trial in range(x.shape[0]):
            new_x.append([])
            for _band in range(self.bands):
                new_x[-1].append([])
                for _pair in range(self.w.shape[0]):
                    new_x[-1][-1].append(self.w[_pair, _band].T @ x[_trial, _band])

        new_x = np.array(new_x)

        old_shape = eegdata['X'].shape
        if len(self.w) != 1:
            new_shape = (*old_shape[:-2], self.w.shape[0], self.m_pairs * 2, old_shape[-1])
        else:
            new_shape = (*old_shape[:-2], self.m_pairs * 2, old_shape[-1])
        new_x = new_x.reshape(new_shape)

        eegdata['X'] = new_x
        return eegdata

    def fit_transform(
            self,
            eegdata: dict
        ) -> dict:

        return self.fit(eegdata).transform(eegdata)
