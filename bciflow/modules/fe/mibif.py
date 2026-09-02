"""Module for the MIBIF feature extractor in the BCIFlow pipeline."""

import numpy as np
from sklearn.metrics import mutual_info_score

class MIBIF:
    """Mutual Information Based Feature Selection (MIBIF) extractor for EEG data.
    """
    def __init__(self, n_features, clf):
        """Initialize the MIBIF extractor.
        
        Parameters
        ----------
        n_features : int
            The number of features to select.
        clf : sklearn classifier
            The classifier to use for computing mutual information.
        """

        self.n_features = n_features
        self.order = None
        self.clf = clf

    def fit(self, eegdata):
        """Fit the MIBIF extractor to the EEG data.
        
        Parameters
        ----------
        eegdata : dict
            A dictionary containing the EEG data and metadata.
            
        Returns
        -------
        MIBIF
            The fitted MIBIF extractor.
        """
        x = eegdata['X'].copy()
        y = eegdata['y'].copy()

        bands = int(np.prod(x.shape) / x.shape[0])
        x = x.reshape(x.shape[0], bands)

        mi = []
        for i in range(x.shape[1]):
            x_ = x[:, [i]]
            self.clf.fit(x_, y)
            y_pred = self.clf.predict(np.array(x_))
            mi.append([i, mutual_info_score(y, y_pred)])

        mi = sorted(mi, key=lambda x: x[1], reverse=True)
        mi = np.array(mi)
        self.order = mi[:, 0].astype(int)

        return self

    def transform(self, eegdata):
        """Transform the EEG data using the fitted MIBIF extractor.
        
        Parameters
        ----------
        eegdata : dict
            A dictionary containing the EEG data and metadata.

        Returns
        -------
        dict
            The transformed EEG data.
        """

        x = eegdata['X'].copy()
        bands = int(np.prod(x.shape) / x.shape[0])
        x = x.reshape(x.shape[0], bands)

        x = x[:, self.order][:, :self.n_features]

        eegdata['X'] = x
        return eegdata

    def fit_transform(self, eegdata):
        """Fit and transform the EEG data using the MIBIF extractor.
        
        Parameters
        ----------
        eegdata : dict
            A dictionary containing the EEG data and metadata.
        
        Returns
        -------
        dict
            The transformed EEG data.
        """

        return self.fit(eegdata).transform(eegdata)
