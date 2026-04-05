'''
Description
-----------
This module implements the Mutual Information Best Individual Features (MIBIF) feature extractor, 
which selects the best features based on mutual information between features and labels. 
This feature is commonly used in BCI applications to reduce dimensionality and improve classification performance.

The MIBIF is calculated using mutual information scores, which measure the dependency between 
each feature and the target labels.

Class
------------
'''
import numpy as np
from sklearn.metrics import mutual_info_score

class MIBIF:
    '''
    Attributes
    ----------
    n_features : int
        The number of features to be selected.
    paired : bool
        Whether the features are paired or not (default is True).
    order : list
        The order of the features based on mutual information scores.
    clf : object
        The classifier used to calculate the mutual information.
    pairs : np.ndarray
        The pairs of features (used when paired=True).
    '''
    def __init__(self, n_features, clf, paired=True):
        ''' Initializes the class.
        
        Parameters
        ----------
        n_features : int
            The number of features to be selected.
        clf : object
            The classifier used to calculate mutual information.
        paired : bool, optional
            Whether the features are paired or not (default is True).
        
        Returns
        -------
        None
        '''
        # -------- Validate n_features --------
        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer.")

        if n_features <= 0:
            raise ValueError("n_features must be a positive integer.")

        # -------- Validate clf --------
        if clf is None:
            raise ValueError("clf must be a valid classifier object.")

        # -------- Validate paired --------
        if not isinstance(paired, bool):
            raise ValueError("paired must be a boolean.")
                             
        self.original_n_features = n_features
        self.n_features = self.original_n_features
        self.paired = paired
        self.order = [0]
        self.clf = clf

    def find_pair(self, u, max_col):
        ''' Finds the pair of a feature.

        Parameters
        ----------
        u : int
            The feature index.
        max_col : int
            The maximum number of columns.

        Returns
        -------
        int
            The index of the pair of the feature.
        '''
        i = int(u / max_col)
        j = u % max_col
        j_pair = max_col - 1 - j
        u = max_col * i + j_pair
        return u

    def fit(self, eegdata):
        ''' 
        This method fits the MIBIF feature extractor to the data by calculating mutual information scores.
        
        Parameters
        ----------
        eegdata : dict
            The input data, containing 'X' (features) and 'y' (labels).
            
        Returns
        -------
        self
        '''

        # -------- Validate eegdata --------
        if not isinstance(eegdata, dict):
            raise ValueError("eegdata must be a dictionary.")

        if 'X' not in eegdata:
            raise ValueError("eegdata must contain the key 'X'.")

        if 'y' not in eegdata:
            raise ValueError("eegdata must contain the key 'y'.")

        X = eegdata['X']
        y = eegdata['y']

        # -------- Validate X --------
        if not isinstance(X, np.ndarray):
            raise ValueError("eegdata['X'] must be a numpy array.")

        if X.ndim not in [2, 3]:
            raise ValueError(
                "X must have shape (trials, features) or (trials, bands, features)."
            )

        if X.shape[0] < 2:
            raise ValueError("X must contain at least two trials.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        # -------- Validate y --------
        if not isinstance(y, np.ndarray):
            raise ValueError("eegdata['y'] must be a numpy array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if len(y) != X.shape[0]:
            raise ValueError("Number of labels must match number of trials.")

        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values.")

        # -------- Validate classifier --------
        if not hasattr(self.clf, "fit"):
            raise ValueError("clf must implement a 'fit' method.")

        if not hasattr(self.clf, "predict"):
            raise ValueError("clf must implement a 'predict' method.")

        # -------- Initialize parameters --------
        self.n_features = self.original_n_features

        X = eegdata['X'].copy()
        y = eegdata['y'].copy()

        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]

        X_ = [X[i].reshape(-1) for i in range(len(X))]
        mi = []
        for i in range(len(X_[0])):
            X__ = []
            for j in range(len(X_)):
                X__.append([X_[j][i]])
            X__ = np.array(X__)
            #self.clf.fit(np.array(X__), y)
            try:
                self.clf.fit(X__, y)
            except:
                X__ = np.nan_to_num(X__)
                X__ += np.random.random(X__.shape) * 1e-6
                self.clf.fit(X__, y)
            y_pred = self.clf.predict(np.array(X__))
            mi.append([i, mutual_info_score(y, y_pred)])                

        mi = sorted(mi, key=lambda x: x[1], reverse=True)
        mi = np.array(mi)
        self.order = mi[:, 0].astype(int)

        if self.paired:
            self.pairs = np.zeros((int(len(X[0])/2), 2))
            max_col = X.shape[-1]
            new_order = []
            n_features = self.n_features
            for i in range(len(self.order)):
                order_ = self.order[i]
                new_order.append(order_)
                order_pair = self.find_pair(order_, max_col)
                new_order.append(order_pair)
                if order_pair not in self.order[:self.n_features] and i < self.n_features:
                    n_features += 1
            self.order = new_order
            self.n_features = n_features

        return self

    def transform(self, eegdata):
        '''
        This method transforms the input data into the selected feature space.
        
        Parameters
        ----------
        eegdata : dict
            The input data, containing 'X' (features).
        
        Returns
        -------
        output : dict
            The transformed data, containing the selected features.
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

        if X.ndim < 2:
            raise ValueError("X must have at least two dimensions (trials, features...).")

        if X.shape[0] < 1:
            raise ValueError("X must contain at least one trial.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        # -------- Validate fitted attributes --------
        if not hasattr(self, "order") or self.order is None:
            raise ValueError("The feature selector has not been fitted yet.")

        if not hasattr(self, "n_features"):
            raise ValueError("The feature selector has not been fitted yet.")
        
        # -------- Prepare data --------
        X_ = [X[i].reshape(-1) for i in range(len(X))]
        X_ = np.array(X_)
        X_ = X_[:, self.order][:, :self.n_features]

        eegdata['X'] = X_
        return eegdata

    def fit_transform(self, eegdata):
        '''
        This method combines fitting and transforming into a single step. It returns a 
        dictionary with the transformed data.
        
        Parameters
        ----------
        eegdata : dict
            The input data, containing 'X' (features) and 'y' (labels).
          
        Returns
        -------
        output : dict
            The transformed data, containing the selected features.
        '''
        return self.fit(eegdata).transform(eegdata)

