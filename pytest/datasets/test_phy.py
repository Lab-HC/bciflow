import numpy as np
import pytest
from unittest.mock import patch

from bciflow.datasets.phy import physio_net

# ======================================================
# Test Suite: PhysioNET
# ======================================================
#
# Cobertura:
#  1. Validação de tipos e valores dos parâmetros de entrada
#  2. Normalização automática do path (adição de '/')
#  3. Mock do carregamento externo (plib highlevel read_edf)
#  4. Extração e segmentação de trials
#  5. Mapeamento de labels
#  6. Estrutura e integridade do dicionário retornado
# ======================================================


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
    def test_full_execution(self, mock_read_edf):
        n_channels = 64
        total_samples = 20000

        fake_data = np.random.randn(n_channels, total_samples) # plib highlevel read_edf

        annotations_df = []

        for i in range(15):
            annotations_df.append([
                np.random.rand(), # onset
                np.random.rand(), # duration
                np.random.choice(['T0']) # description
            ])
            annotations_df.append([
                np.random.rand(), # onset
                np.random.rand(), # duration
                np.random.choice(['T1', 'T2']) # description
            ])

        fake_header = {
            "annotations": annotations_df
        }

        fake_signal_header = []
        for i in range(n_channels):
            fake_signal_header.append({
                "label": f"Ch{i+1}"
            })

        mock_read_edf.return_value = (fake_data, fake_signal_header, fake_header)

        session_list_param = [3]

        eeg = physio_net(subject=1, session_list=session_list_param)

        assert eeg["data_type"] == "epochs"
        assert eeg["X"].shape == (15, 1, n_channels, 640)
        assert len(eeg["y"]) == 15
        assert eeg["sfreq"] == 160.
        assert eeg["tmin"] == 0.
    
    # ======================================================
    # Section 3 - Full Execution Test (with subjects 1 and 2)
    # ======================================================
    @patch("bciflow.datasets.plib.highlevel.read_edf")
    def test_full_execution_sessions_1_and_2(self, mock_read_edf):
        n_channels = 64
        total_samples = 9760

        fake_data = np.random.randn(n_channels, total_samples) # plib highlevel read_edf

        annotations_df = []

        annotations_df.append([
            np.random.rand(), # onset
            np.random.rand(), # duration
            np.random.choice(['T0']) # description
        ])

        fake_header = {
            "annotations": annotations_df
        }

        fake_signal_header = []
        for i in range(n_channels):
            fake_signal_header.append({
                "label": f"Ch{i+1}"
            })
        
        mock_read_edf.return_value = (fake_data, fake_signal_header, fake_header)
        session_list_param = [1, 2]

        eeg = physio_net(subject=1, session_list=session_list_param, path='fake_path')

        assert eeg["data_type"] == "epochs"
        assert eeg["X"].shape == (0, 1, n_channels, 640)
        assert eeg["X_og"].shape == (2, 1, n_channels, 9760)
        assert len(eeg["y"]) == 2
        assert eeg["sfreq"] == 160.
        assert eeg["tmin"] == 0.
        
    # ======================================================
    # SECTION 4 — Auxiliary Function Tests
    # ======================================================

    def test_aux_function(self):
        from bciflow.datasets.phy import _string_to_number

        assert _string_to_number("T0", 3) == 0
        assert _string_to_number("T1", 3) == 1
        assert _string_to_number("T2", 3) == 2

        assert _string_to_number("T0", 4) == 0
        assert _string_to_number("T1", 4) == 5
        assert _string_to_number("T2", 4) == 6

        assert _string_to_number("T0", 5) == 0
        assert _string_to_number("T1", 5) == 3
        assert _string_to_number("T2", 5) == 4

        assert _string_to_number("T0", 6) == 0
        assert _string_to_number("T1", 6) == 7
        assert _string_to_number("T2", 6) == 8

        assert _string_to_number("Invalid", 0) == -1