"""Metric functions for evaluating epoch-wise classification results."""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

def metric_by_time_epochs(
        results: pd.DataFrame,
        metric_func: callable = cohen_kappa_score
):
    """Evaluate a metric by time epochs.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing the classification results.
    metric_func : callable, optional
        The metric function to evaluate, by default cohen_kappa_score

    Returns
    -------
    dict
        A dictionary with time epochs as keys and metric values as values.
    """

    output = {}
    labels = results.drop(columns=['fold', 'tmax', 'true_label']).columns
    labels = {label: idx for idx, label in enumerate(labels)}

    for _tmax in results['tmax'].unique():

        df_tmax = results[results['tmax'] == _tmax]
        y_true = np.array(df_tmax['true_label'].values)
        y_prob = df_tmax.drop(columns=['fold', 'tmax', 'true_label']).values
        y_prob = np.array(y_prob)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.array([labels[label] for label in y_true])

        metric_value = metric_func(y_true, y_pred)
        output[_tmax] = metric_value

    return output
