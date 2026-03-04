import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.bciciv2a_raw import bciciv2a_raw



# ==========================================================
# Test Suite: bciciv2a_raw
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

class TestBCICIV2A_raw:


    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            bciciv2a_raw(subject="1")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            bciciv2a_raw(subject=10)

    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            bciciv2a_raw(session_list="T")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            bciciv2a_raw(session_list=["X"])

    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            bciciv2a_raw(path=123)

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            bciciv2a_raw(verbose="INVALID")

    def test_verbose_not_string(self):
        with pytest.raises(ValueError):
            bciciv2a_raw(verbose=2.0)

    # ======================================================
    # SECTION 2 — Full Execution Test (mocked)
    # ======================================================

    @patch("bciflow.datasets.bciciv2a.scipy.io.loadmat")
    @patch("bciflow.datasets.bciciv2a.mne.io.read_raw_gdf")
    def test_full_execution_single_session(self, mock_read_raw, mock_loadmat):

        n_channels = 22
        n_times = 3000

        fake_data = np.random.randn(n_channels, n_times)
        fake_times = np.linspace(0, 10, n_times)

        # Eventos: trial (768), eyes open (276), eyes closed (277)
        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime([
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:01",
                "2020-01-01 00:00:02"
            ]),
            "description": [768, 276, 277]
        }).reset_index(drop=True)

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw.return_value = fake_raw
        mock_loadmat.return_value = {"classlabel": np.array([[1]])}

        result = bciciv2a_raw(subject=1, session_list=["T"])

        # Estrutura básica
        assert result["data_type"] == "raw"
        assert result["X"].shape == (22, n_times)
        assert result["y"].shape == (n_times,)
        assert isinstance(result["y_dict"], dict)

        # Garantir que labels foram realmente atribuídos
        assert np.any(result["y"] > 0)


    @patch("bciflow.datasets.bciciv2a.scipy.io.loadmat")
    @patch("bciflow.datasets.bciciv2a.mne.io.read_raw_gdf")
    def test_multi_session_concatenation(self, mock_read_raw, mock_loadmat):

        n_channels = 22
        n_times = 1000

        fake_data = np.random.randn(n_channels, n_times)
        fake_times = np.linspace(0, 5, n_times)

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime(["2020-01-01 00:00:00"]),
            "description": [768]
        })

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw.return_value = fake_raw
        mock_loadmat.return_value = {"classlabel": np.array([[2]])}

        result = bciciv2a_raw(subject=1, session_list=["T", "E"])

        # concatenação no eixo temporal
        assert result["X"].shape == (22, n_times * 2)
        assert result["y"].shape == (n_times * 2,)


    # ======================================================
    # SECTION 3 - Test with EOG=True
    # ======================================================

    @patch("bciflow.datasets.bciciv2a.scipy.io.loadmat")
    @patch("bciflow.datasets.bciciv2a.mne.io.read_raw_gdf")
    def test_execution_with_eog(self, mock_read_raw, mock_loadmat):

        n_channels = 25  # 22 EEG + 3 EOG
        n_times = 1500

        fake_data = np.random.randn(n_channels, n_times)
        fake_times = np.linspace(0, 6, n_times)

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime(["2020-01-01 00:00:00"]),
            "description": [768]
        })

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw.return_value = fake_raw
        mock_loadmat.return_value = {"classlabel": np.array([[3]])}

        result = bciciv2a_raw(subject=1, EOG=True)

        assert len(result["ch_names"]) == 25
        assert result["X"].shape[0] == 25


    # ======================================================
    # SECTION 4 - Test Path Handling
    # ======================================================

    @patch("bciflow.datasets.bciciv2a.scipy.io.loadmat")
    @patch("bciflow.datasets.bciciv2a.mne.io.read_raw_gdf")
    def test_path_normalization(self, mock_read_raw, mock_loadmat):

        n_channels = 22
        n_times = 1000

        fake_data = np.random.randn(n_channels, n_times)
        fake_times = np.linspace(0, 5, n_times)

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime(["2020-01-01 00:00:00"]),
            "description": [768]
        })

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw.return_value = fake_raw
        mock_loadmat.return_value = {"classlabel": np.array([[1]])}

        bciciv2a_raw(subject=1,session_list=['T'], path="my/custom/path")

        # Verifica se a função adicionou '/' automaticamente
        mock_read_raw.assert_called_with(
            "my/custom/path/A01T.gdf",
            preload=True,
            verbose="ERROR"
        )

    