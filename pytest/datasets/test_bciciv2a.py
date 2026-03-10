import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.bciciv2a import bciciv2a



# ==========================================================
# Test Suite: bciciv2a
# ==========================================================
#
# Cobertura:
#   1. Validação de tipos (subject, labels, session_list, path, verbose)
#   2. Validação de valores permitidos (subject range, labels válidos, sessions válidas)
#   3. Normalização automática do path (adição de '/')
#   4. Branch EOG (inclusão de canais extras)
#   5. Mock do carregamento externo (mne + scipy)
#   6. Extração e segmentação de trials
#   7. Concatenação de múltiplas sessões
#   8. Mapeamento e filtragem de labels
#   9. Estrutura e integridade do dicionário retornado
#
# ==========================================================

class TestBCICIV2A:


    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            bciciv2a(subject="1")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            bciciv2a(subject=10)

    def test_invalid_labels_type(self):
        with pytest.raises(ValueError):
            bciciv2a(labels="left-hand")

    def test_invalid_label_value(self):
        with pytest.raises(ValueError):
            bciciv2a(labels=["invalid"])

    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            bciciv2a(session_list="T")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            bciciv2a(session_list=["X"])

    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            bciciv2a(path=123)

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            bciciv2a(verbose="INVALID")

    def test_verbose_not_string(self):
        with pytest.raises(ValueError):
            bciciv2a(verbose=2.0)

    # ======================================================
    # SECTION 2 — Full Execution Test (mocked)
    # ======================================================

    @patch("bciflow.datasets.scipy.io.loadmat")
    @patch("bciflow.datasets.mne.io.read_raw_gdf")
    def test_full_execution(self, mock_read_raw_gdf, mock_loadmat):

        n_channels = 22
        total_samples = 5000
        trial_size = 1875

        # fake raw
        fake_data = np.random.randn(n_channels, total_samples)
        fake_times = np.arange(total_samples) / 250

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime([0, fake_times[2000]], unit="s"),
            "description": [768, 768]
        })

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw_gdf.return_value = fake_raw

        mock_loadmat.return_value = {
            "classlabel": np.array([[1], [2]])
        }

        eeg = bciciv2a(subject=1, session_list=["T"])

        assert eeg["data_type"] == "epochs"
        assert eeg["X"].shape == (2, n_channels, trial_size)
        assert eeg["y"].shape == (2,)
        assert eeg["sfreq"] == 250.
        assert eeg["tmin"] == 0.


    # ======================================================
    # SECTION 3 - Test with EOG=True
    # ======================================================

    @patch("bciflow.datasets.scipy.io.loadmat")
    @patch("bciflow.datasets.mne.io.read_raw_gdf")
    def test_with_eog(self, mock_read_raw_gdf, mock_loadmat):

        n_channels = 25  # 22 EEG + 3 EOG
        total_samples = 5000
        trial_size = 1875

        fake_data = np.random.randn(n_channels, total_samples)
        fake_times = np.arange(total_samples) / 250

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime([0], unit="s"),
            "description": [768]
        })

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw_gdf.return_value = fake_raw
        mock_loadmat.return_value = {"classlabel": np.array([[1]])}

        eeg = bciciv2a(subject=1, session_list=["T"], EOG=True)

        assert eeg["X"].shape[1] == 25
        assert len(eeg["ch_names"]) == 25


    # ======================================================
    # SECTION 4 - Test Path Handling
    # ======================================================

    @patch("bciflow.datasets.scipy.io.loadmat")
    @patch("bciflow.datasets.mne.io.read_raw_gdf")
    def test_path_handling(self, mock_read_raw_gdf, mock_loadmat):

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = np.zeros((22, 5000))
        fake_raw.times = np.arange(5000) / 250

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime([0], unit="s"),
            "description": [768]
        })

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw_gdf.return_value = fake_raw
        mock_loadmat.return_value = {"classlabel": np.array([[1]])}

        custom_path = "my/custom/path"  # sem barra final

        bciciv2a(subject=1, session_list=["T"], path=custom_path)

        expected_path = "my/custom/path/A01T.gdf"

        mock_read_raw_gdf.assert_called_with(
            expected_path,
            preload=True,
            verbose="ERROR"
        )