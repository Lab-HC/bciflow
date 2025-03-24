'''
Description
-----------
This module implements the Euclidean Alignment (EA) method, a spatial filtering technique 
used to align EEG data from different subjects or sessions to a common reference. 
This reduces inter-subject variability and improves the generalization of BCI models.

The EA method aligns EEG data by transforming it such that the reference matrix becomes 
an identity matrix. This is particularly useful for cross-subject or cross-session 
BCI applications.

Class
------------
'''
from scipy.linalg import fractional_matrix_power
import numpy as np

class ea:
    '''
    Attributes
    ----------
    target_transformation : list-like, size (n_bands)
        List containing the reference matrix for each band of the target subject
    '''
    def __init__(self):   
        self.target_transformation = None

    def calc_r(self, data):
        ''' 
        Computes the reference matrix for each frequency band.
        
        Parameters
        ----------
        data : np.ndarry
            The input data.
        
        returns
        -------
        np.ndarray
            The reference matrix for the input data
            
        '''
        list_r = []
        for band in range(data.shape[1]):
            r = np.zeros((data.shape[2], data.shape[2]))
            for trial in range(data.shape[0]):
                product = np.dot(data[trial][band], data[trial][band].T)
                r += product
            r = r / data.shape[0]
            list_r.append(r)
        return np.array(list_r)
    
    def full_r(self, data):
        ''' 
        Computes the transformation matrix by raising the reference matrix to the power of -1/2.
        
        Parameters
        ----------
        data : np.ndarry
            The input data.
        
        returns
        -------
        np.ndarray
            The reference matrix for the input data
            
        '''
        list_r = self.calc_r(data)
        list_r_inv = [fractional_matrix_power(r, -0.5) for r in list_r]
        return np.array(list_r_inv)

    def fit(self, eegdata):
        ''' 
        Fits the EA method to the input data, calculating the transformation matrices.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
        
        returns
        -------
        self
            
        '''
        data = eegdata['X'].copy()
        self.target_transformation = self.full_r(data)
        return self

    def transform(self, eegdata):
        ''' 
        Applies the learned transformation matrices to align the input data.
        
        Parameters
        ----------
        eegdata : dict
            The input data.
            
        returns
        -------
        output : dict
            The transformed data.
            
        '''
        X = eegdata['X'].copy()

        for band in range(X.shape[1]):
            for trial in range(X.shape[0]):
                X[trial][band] = np.dot(self.target_transformation[band], X[trial][band])

        eegdata['X'] = X
        return eegdata

    def fit_transform(self, eegdata):
        ''' 
        Combines fitting and transforming into a single step.

        Parameters
        ----------
        eegdata : dict
            The input data.
        
        returns
        -------
        output : dict
            The transformed data.
            
        '''
        return self.fit(eegdata).transform(eegdata)