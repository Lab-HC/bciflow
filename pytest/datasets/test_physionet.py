import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.phy import physio_net

class TestPhysioNet:

    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

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
    
    # ======================================================
    # SECTION 2 — Full Execution Test (mocked)
    # ======================================================
    @patch("bciflow.datasets.plib.highlevel.read_edf")
    def test_full_execution(self, mock_read_edf, mock_read_raw_edf):
        n_channels = 64
        total_samples = 20000

        fake_data = np.random.randn(n_channels, total_samples) # plib highlevel read_edf

        annotations_df = {
            "onset": np.random.rand(30),
            "duration": np.random.rand(30),
            "description": np.random.choice(['T0', 'T1', 'T2'], 30)
        }

        fake_header = {
            "annotations": annotations_df
        }

        fake_signal_header = []
        for i in range(n_channels):
            fake_signal_header.append({
                "label": f"Ch{i+1}"
            })

        mock_read_edf.return_value = (fake_data, fake_signal_header, fake_header)

        eeg = physio_net(subject=1, session_list=[3])

        assert eeg["data_type"] == "epochs"
        assert eeg["X"].shape == (360, 1, n_channels, 640)
        assert eeg["y"].shape == (360,)
        assert eeg["sfreq"] == 160.
        assert eeg["tmin"] == 0.



    
