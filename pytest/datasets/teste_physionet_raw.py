import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.physionet_raw import physio_net_raw


class TestPhysioNetRaw:
    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            physio_net_raw(subject="1")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            physio_net_raw(subject=0)
        with pytest.raises(ValueError):
            physio_net_raw(subject=110)
    
    def test_invalid_labels_type(self):
        with pytest.raises(ValueError):
            physio_net_raw(labels="left-hand")
    
    def test_invalid_label_value(self):
        with pytest.raises(ValueError):
            physio_net_raw(labels=["invalid"])
    
    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            physio_net_raw(session_list="T")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            physio_net_raw(session_list=[0])
        with pytest.raises(ValueError):
            physio_net_raw(session_list=[15])
    
    def test_invalid_session_value_non_int(self):
        with pytest.raises(ValueError):
            physio_net_raw(session_list=[3.5])
        with pytest.raises(ValueError):
            physio_net_raw(session_list=["3"])
    
    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            physio_net_raw(path=123)
    
    def test_invalid_verbose_type(self):
        with pytest.raises(ValueError):
            physio_net_raw(verbose=123)
        with pytest.raises(ValueError):
            physio_net_raw(verbose="INVALID")

    # ======================================================
    # SECTION 2 — Full Execution Test (mocked)
    # ======================================================

    @patch("bciflow.datasets.mne.io.read_raw_gdf")
    def test_full_execution(self, mock_read_raw_edf):

        n_channels = 64
        total_samples = 9760

        # fake raw
        fake_data = np.random.randn(n_channels, total_samples)
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = fake_data
        mock_read_raw_edf.return_value = mock_raw

        # Call the function with mocked data
        result = physio_net_raw(subject=1, session_list=[3], labels=['rest'], path='fake_path/', verbose='INFO')

        # Assertions to check if the function behaves as expected with mocked data
        assert 'X' in result and 'y' in result and 'ch_names' in result
        assert result['X'].shape == (360, 1, 64, 640)  # Expected shape based on the code logic
        assert len(result['y']) == 360  # Expected number of labels based on the code logic