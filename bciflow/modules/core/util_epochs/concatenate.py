"""Function to concatenate multiple epoch datasets into a single dataset."""

import numpy as np

def concatenate(data_colection):
    """Concatenate multiple epoch datasets into a single dataset.
    
    Parameters
    ----------
    data_colection : list of dict
        A list of dictionaries containing the data to be concatenated. Each
        dictionary must have the keys 'X' and 'y'.
    
    Returns
    -------
    dict
        A dictionary containing the concatenated data. It has the keys 'X' and 'y'.    
    """

    data = data_colection[0].copy()
    for data_ in data_colection[1:]:
        data["X"] = np.concatenate([data["X"], data_["X"]])
        data["y"] = np.concatenate([data["y"], data_["y"]])

    return data
