import numpy as np
import pytest
from unittest.mock import patch, call

from bciflow.datasets.cbcic_raw import cbcic_raw 

# ==========================================================
# Test Suite: cbcic_raw
# ==========================================================
# Cobertura:
#   1. Validação de tipos (subject, session_list, path)
#   2. Validação de valores permitidos (subject range, sessions válidas)
#   3. Uso de valores default (session_list=None → todas as sessões)
#   4. Normalização automática do path (adição de '/')
#   5. Mock do carregamento externo (scipy.io.loadmat)
#   6. Tratamento de formato dos dados (transpose condicional)
#   7. Conversão de trials em sinal contínuo (concatenação temporal)
#   8. Construção de eventos (offset acumulativo)
#   9. Mapeamento de labels (1/2 → 0/1 nos eventos)
#   10. Ignorar labels inválidos (não gera evento)
#   11. Concatenação de múltiplas sessões
#   12. Estrutura e integridade do dicionário retornado
# ==========================================================

class TestCBCIC_raw:
    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            cbcic_raw(subject="A")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            cbcic_raw(subject=11)

    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            cbcic_raw(session_list="T")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            cbcic_raw(session_list=["X"])

    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            cbcic_raw(path=432.2)


    # ======================================================
    # SECTION 2 — Full Execution Test (mocked)
    # ======================================================

    @patch("bciflow.datasets.cbcic_raw.loadmat")
    def test_full_execution_single_session(self, mock_loadmat):

        n_trials = 3
        n_samples = 1000
        n_channels = 12

        # shape esperado: (trials, samples, channels)
        fake_eeg = np.random.randn(n_trials, n_samples, n_channels)

        fake_labels = np.array([[1], [2], [1]])

        mock_loadmat.return_value = {
            "RawEEGData": fake_eeg,
            "Labels": fake_labels
        }

        result = cbcic_raw(subject=1, session_list=["T"])

        assert result["data_type"] == "raw"
        assert result["X"].shape == (n_trials * n_samples, n_channels)
        assert result["events"].shape[0] == n_trials
        assert set(result["events"][:, 2]) == {0, 1}
        assert result["sfreq"] == 512.
        assert len(result["ch_names"]) == 12

    @patch("bciflow.datasets.cbcic_raw.loadmat")
    def test_transpose_branch(self, mock_loadmat):

        n_trials = 2
        n_channels = 12
        n_samples = 500

        # aqui channels está no eixo 1 → ativa o transpose
        fake_eeg = np.random.randn(n_trials, n_channels, n_samples)

        fake_labels = np.array([[1], [2]])

        mock_loadmat.return_value = {
            "RawEEGData": fake_eeg,
            "Labels": fake_labels
        }

        result = cbcic_raw(subject=1, session_list=["T"])

        # depois do transpose → vira (samples, channels)
        assert result["X"].shape == (n_trials * n_samples, n_channels)
    
    @patch("bciflow.datasets.cbcic_raw.loadmat")
    def test_ignore_invalid_labels(self, mock_loadmat):

        n_trials = 3
        n_samples = 500
        n_channels = 12

        fake_eeg = np.random.randn(n_trials, n_samples, n_channels)

        fake_labels = np.array([[1], [3], [2]])

        mock_loadmat.return_value = {
            "RawEEGData": fake_eeg,
            "Labels": fake_labels
        }

        result = cbcic_raw(subject=1, session_list=["T"])
        assert result["events"].shape[0] == 2

    # ======================================================
    # SECTION 3 - Test Path Handling
    # ======================================================

    @patch("bciflow.datasets.cbcic_raw.loadmat")
    def test_path_normalization(self, mock_loadmat):

        n_trials = 2
        n_samples = 500
        n_channels = 12

        fake_eeg = np.random.randn(n_trials, n_samples, n_channels)
        fake_labels = np.array([[1], [2]])

        mock_loadmat.return_value = {
            "RawEEGData": fake_eeg,
            "Labels": fake_labels
        }

        cbcic_raw(subject=1, session_list=['T'], path="my/custom/path")
        expected_path = "my/custom/path/parsed_P01T.mat"
        mock_loadmat.assert_called_with(expected_path)

    @patch("bciflow.datasets.cbcic_raw.loadmat")
    def test_path_multiple_sessions(self, mock_loadmat):

        fake_eeg = np.random.randn(2, 500, 12)
        fake_labels = np.array([[1], [2]])

        mock_loadmat.return_value = {
            "RawEEGData": fake_eeg,
            "Labels": fake_labels
        }

        cbcic_raw(subject=1, session_list=['T', 'E'], path="my/custom/path")

        expected_calls = [
            call("my/custom/path/parsed_P01T.mat"),
            call("my/custom/path/parsed_P01E.mat"),
        ]

        mock_loadmat.assert_has_calls(expected_calls, any_order=False)
        assert mock_loadmat.call_count == 2 