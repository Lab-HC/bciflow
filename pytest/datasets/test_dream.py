import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.dreams import dreams_dataset as dreams

class TestDreamsDataset():

    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================
    def test_invalid_subject_type(self):
        with pytest.raises(TypeError):
            dreams(subject="1")
    
    def test_invalid_subject_range(self):
        with pytest.raises(TypeError):
            dreams(subject=-1)
    
    def test_path_not_string(self):
        with pytest.raises(TypeError):
            dreams(path=123)
    
    def test_invalid_data_groups_type(self):
        with pytest.raises(TypeError):
            dreams(data_groups="EEG")
    
    def test_invalid_data_group_type(self):
        with pytest.raises(ValueError):
            dreams(data_groups=["EEG", "INVALID_GROUP"])
    
    # ======================================================
    # SECTION 2 - Full Execution Test (mocked)
    # ======================================================
    @patch("bciflow.datasets.os.path.exists")
    @patch("bciflow.datasets.plib.highlevel.read_edf")
    def test_full_execution(self, mock_read_edf, mock_exists):
        fake_signals = np.random.randn(360000, 23)
        fake_y = np.random.randint(0, 6, size=(360000))
        pass