"""Function to extract specific trials."""

def get_trials(data, ids):
    """Extract specific trials from the data.
    
    Parameters
    ----------
    data : dict
        A dictionary containing the data to be processed. It must have
        the keys 'X' and 'y'.
    ids : list of int
        A list of indices specifying which trials to extract.

    Returns
    -------
    dict
        A dictionary containing the extracted trials. It has the keys 'X' and 'y'.
    """

    data_copy = data.copy()
    data_copy['X'] = data_copy['X'][ids]
    data_copy['y'] = data_copy['y'][ids]

    return data_copy
