import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.bciciv2b_raw import bciciv2b_raw

# ==========================================================
# Test Suite: bciciv2b_raw
# ==========================================================
# Cobertura:
#   1. Validação de tipos (subject, session_list, path)
#   2. Validação de valores permitidos (subject range, sessions válidas)
#   3. Uso de valores default (session_list=None → todas as sessões)
#   4. Normalização automática do path (adição de '/')
#   5. Mock do carregamento externo (mne.io.read_raw_gdf)
#   6. Extração de dados brutos (seleção dos 3 primeiros canais)
#   7. Processamento de annotations e geração de labels contínuos
#   8. Mapeamento de eventos (event_conversion)
#   9. Ignorar eventos desconhecidos (fallback para background)
#   10. Tratamento do último evento (branch do else no loop)
#   11. Concatenação de múltiplas sessões no eixo temporal
#   12. Estrutura e integridade do dicionário retornado
# ==========================================================

class TestBCICIV2B_raw:


    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            bciciv2b_raw(subject=0)

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            bciciv2b_raw(subject=10)

    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            bciciv2b_raw(session_list="T")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            bciciv2b_raw(session_list=["X"])

    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            bciciv2b_raw(path=123)

    # ======================================================
    # SECTION 2 — Full Execution Test (mocked)
    # ======================================================

    @patch("bciflow.datasets.bciciv2b.mne.io.read_raw_gdf")
    def test_full_execution_single_session(self, mock_read_raw):

        n_channels = 3
        n_times = 3000

        fake_data = np.random.randn(n_channels, n_times)
        fake_times = np.linspace(0, 10, n_times)

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime([
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:02",
                "2020-01-01 00:00:04"
            ]),
            "description": [768, 769, 770]  # start, left, right
        }).reset_index(drop=True)

        # Mock do Raw
        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw.return_value = fake_raw

        result = bciciv2b_raw(subject=1, session_list=["01T"])

        # Estrutura básica
        assert result["data_type"] == "raw"
        assert result["X"].shape == (3, n_times)
        assert result["y"].shape == (n_times,)

        assert result["sfreq"] == 250.
        assert isinstance(result["y_dict"], dict)

        # Checar canais
        assert len(result["ch_names"]) == 3

        # Garantir que labels foram atribuídos
        assert np.any(result["y"] > 0)

    @patch("bciflow.datasets.bciciv2b.mne.io.read_raw_gdf")
    def test_multi_session_concatenation(self, mock_read_raw):

        n_channels = 3
        n_times = 1000

        fake_data = np.random.randn(n_channels, n_times)
        fake_times = np.linspace(0, 5, n_times)

        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime(["2020-01-01 00:00:00"]),
            "description": [768]  # start-of-trial
        })

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = fake_data
        fake_raw.times = fake_times

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw.return_value = fake_raw

        result = bciciv2b_raw(subject=1, session_list=["01T", "02T"])

        # concatenação no tempo
        assert result["X"].shape == (3, n_times * 2)
        assert result["y"].shape == (n_times * 2,)

        assert result["data_type"] == "raw"
        assert result["sfreq"] == 250.
        assert len(result["ch_names"]) == 3
    
    @patch("bciflow.datasets.bciciv2b.mne.io.read_raw_gdf")
    def test_ignore_unknown_event(self, mock_read_raw):
        n_times = 1000

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = np.random.randn(3, n_times)
        fake_raw.times = np.linspace(0, 5, n_times)

        # código que NÃO existe no event_conversion
        annotations_df = pd.DataFrame({
            "onset": pd.to_datetime(["2020-01-01 00:00:00"]),
            "description": [9999]
        })

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = annotations_df
        fake_raw.annotations = fake_annotations

        mock_read_raw.return_value = fake_raw

        result = bciciv2b_raw(subject=1, session_list=["01T"])

        assert np.all(result["y"] == 0)
    
    @patch("bciflow.datasets.bciciv2b.mne.io.read_raw_gdf")
    def test_session_list_none_uses_all_sessions(self, mock_read_raw):

        fake_raw = MagicMock()
        fake_raw.get_data.return_value = np.random.randn(3, 1000)
        fake_raw.times = np.linspace(0, 5, 1000)

        fake_annotations = MagicMock()
        fake_annotations.to_data_frame.return_value = pd.DataFrame({
            "onset": pd.to_datetime(["2020-01-01"]),
            "description": [768]
        })

        fake_raw.annotations = fake_annotations
        mock_read_raw.return_value = fake_raw

        bciciv2b_raw(subject=1, session_list=None)

        assert mock_read_raw.call_count == 10
    # ======================================================
    # SECTION 3 - Test Path Handling
    # ======================================================

    @patch("bciflow.datasets.bciciv2b.mne.io.read_raw_gdf")
    def test_path_normalization(self, mock_read_raw):

        n_channels = 3
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

        # sem barra no final
        bciciv2b_raw(subject=1, session_list=["01T"], path="my/custom/path")

        mock_read_raw.assert_called_with(
            "my/custom/path/B0101T.gdf",
            preload=True,
            verbose="ERROR"
        )