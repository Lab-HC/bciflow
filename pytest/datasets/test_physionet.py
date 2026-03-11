import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.physionet import physio_net

class TestPhysioNet:
    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            physio_net(subject="1")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            physio_net(subject=0)
        with pytest.raises(ValueError):
            physio_net(subject=110)
    
    def test_invalid_labels_type(self):
        with pytest.raises(ValueError):
            physio_net(labels="left-hand")
    
    def test_invalid_label_value(self):
        with pytest.raises(ValueError):
            physio_net(labels=["invalid"])
    
    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            physio_net(session_list="T")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            physio_net(session_list=[0])
        with pytest.raises(ValueError):
            physio_net(session_list=[15])
    
    def test_invalid_session_value_non_int(self):
        with pytest.raises(ValueError):
            physio_net(session_list=[3.5])
        with pytest.raises(ValueError):
            physio_net(session_list=["3"])
    
    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            physio_net(path=123)
    
    def test_invalid_verbose_type(self):
        with pytest.raises(ValueError):
            physio_net(verbose=123)
        with pytest.raises(ValueError):
            physio_net(verbose="INVALID")
    

    
