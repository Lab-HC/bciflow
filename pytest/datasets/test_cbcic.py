import numpy as np
import pytest
from unittest.mock import patch

from bciflow.datasets.cbcic import cbcic 

# ==========================================================
# Test Suite: cbcic
# ==========================================================
# Cobertura:
#   1. Validação de tipos (subject, labels, session_list, path)
#   2. Validação de valores permitidos (subject range, labels válidos, sessions válidas)
#   3. Uso de valores default (session_list=None → todas as sessões)
#   4. Normalização automática do path (adição de '/')
#   5. Mock do carregamento externo (scipy.io.loadmat)
#   6. Reshape dos dados ([trials, channels, time] → [trials, 1, channels, time])
#   7. Concatenação de múltiplas sessões
#   8. Mapeamento de labels (1/2 → left/right → 0/1)
#   9. Filtragem de labels (subset de labels selecionados)
#   10. Tratamento de erro (arquivo inexistente → ValueError)
#   11. Estrutura e integridade do dicionário retornado
# ==========================================================

class TestCBCIC:


    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            cbcic(subject="A")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            cbcic(subject=11)

    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            cbcic(session_list="T")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            cbcic(session_list=["X"])

    def test_invalid_labels_type(self):
        with pytest.raises(ValueError):
            cbcic(labels="left-hand")

    def test_invalid_label_value(self):
        with pytest.raises(ValueError):
            cbcic(labels=["invalid"])

    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            cbcic(path=432.2)

    # ======================================================
    # SECTION — Full Execution Test (mocked)
    # ======================================================

    @patch("bciflow.datasets.cbcic.loadmat")
    def test_full_execution(self, mock_loadmat):

        n_trials = 4
        n_channels = 12
        n_times = 1000

        # Fake EEG data: (trials, channels, time)
        fake_data = np.random.randn(n_trials, n_channels, n_times)

        # Fake labels (1 = left, 2 = right)
        fake_labels = np.array([[1], [2], [1], [2]])

        mock_loadmat.return_value = {
            "RawEEGData": fake_data,
            "Labels": fake_labels
        }

        eeg = cbcic(subject=1, session_list=["T"])
        assert eeg["data_type"] == "epochs"
        assert eeg["X"].shape == (n_trials, 1, n_channels, n_times)
        assert eeg["y"].shape == (n_trials,)
        assert set(eeg["y"]) == {0, 1}
        assert eeg["sfreq"] == 512.
        assert eeg["tmin"] == 0.
        assert "left-hand" in eeg["y_dict"]
        assert "right-hand" in eeg["y_dict"]

    @patch("bciflow.datasets.cbcic.loadmat")
    def test_file_not_found(self, mock_loadmat):

        mock_loadmat.side_effect = FileNotFoundError

        with pytest.raises(ValueError):
            cbcic(subject=1, session_list=["T"])

    # ======================================================
    # SECTION 3 - Test Path Handling
    # ======================================================

    @patch("bciflow.datasets.cbcic.loadmat")
    def test_path_handling(self, mock_loadmat):

        # Fake retorno do loadmat
        mock_loadmat.return_value = {
            "RawEEGData": np.random.randn(2, 12, 1000),
            "Labels": np.array([[1], [2]])
        }

        custom_path = "my/custom/path"  # sem barra final

        cbcic(subject=1, session_list=["T"], path=custom_path)

        # caminho esperado (com / adicionado automaticamente)
        expected_path = "my/custom/path/parsed_P01T.mat"

        mock_loadmat.assert_called_with(expected_path)