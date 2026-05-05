import numpy as np
import pandas as pd
import pytest

from bciflow.modules.core.kfold import classification_kfold

# ==========================================================
# Test Suite: classification_kfold
# ==========================================================
#
# Cobertura:
#   1. Validação estrutural de target
#   2. Validação de parâmetros (window, folding)
#   3. Execução do pre-fold
#   4. Execução do pos-fold
#   5. Normalização float → list
#   6. Integração com StratifiedKFold
#   7. Estrutura final do DataFrame
#
# ==========================================================

class TestClassificationKfold:
    
    # ==========================================================
    # Fixtures
    # ==========================================================

    @pytest.fixture
    def dummy_target(self):
        return {
            "X": np.random.rand(20, 4, 100),
            "y": np.array([0, 1] * 10),
            "sfreq": 100,
            "y_dict": {"A": 0, "B": 1},
            "events": {},
            "ch_names": ["C1", "C2", "C3", "C4"],
            "tmin": 0.0
        }


    # ==========================================================
    # SECTION 1 — Target Validation
    # ==========================================================

    def test_target_not_dict(self):
        with pytest.raises(ValueError):
            classification_kfold(
                target="invalid",
                start_window=0.0,
                pos_folding={"clf": (lambda x: x, {})}
            )


    def test_missing_target_key(self, dummy_target):
        bad = dummy_target.copy()
        del bad["X"]

        with pytest.raises(ValueError):
            classification_kfold(
                target=bad,
                start_window=0.0,
                pos_folding={"clf": (lambda x: x, {})}
            )


    # ==========================================================
    # SECTION 2 — Window Validation
    # ==========================================================

    def test_invalid_start_window(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window="invalid",
                pos_folding={"clf": (lambda x: x, {})}
            )


    def test_invalid_start_window_list(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=[0.0, "bad"],
                pos_folding={"clf": (lambda x: x, {})}
            )


    def test_invalid_start_test_window(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                start_test_window="invalid",
                pos_folding={"clf": (lambda x: x, {})}
            )


    def test_invalid_start_test_window_list(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                start_test_window=[0.5, "bad"],
                pos_folding={"clf": (lambda x: x, {})}
            )


    # ==========================================================
    # SECTION 3 — pre_folding Validation
    # ==========================================================

    def test_invalid_prefolding_type(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pre_folding="invalid",
                pos_folding={"clf": (lambda x: x, {})}
            )


    def test_invalid_prefolding_structure(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pre_folding={"step": "wrong"},
                pos_folding={"clf": (lambda x: x, {})}
            )


    def test_invalid_prefolding_callable(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pre_folding={"step": (123, {})},
                pos_folding={"clf": (lambda x: x, {})}
            )


    def test_invalid_prefolding_params(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pre_folding={"step": (lambda x: x, "bad")},
                pos_folding={"clf": (lambda x: x, {})}
            )


    # ==========================================================
    # SECTION 4 — pos_folding Validation
    # ==========================================================

    def test_invalid_posfold_type(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pos_folding="invalid"
            )


    def test_missing_clf(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pos_folding={}
            )


    def test_invalid_posfold_structure(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pos_folding={"clf": "wrong"}
            )


    def test_invalid_posfold_callable(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pos_folding={"clf": (123, {})}
            )


    def test_invalid_posfold_params(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pos_folding={"clf": (lambda x: x, "bad")}
            )


    # ==========================================================
    # SECTION 5 — window_size Validation
    # ==========================================================

    def test_invalid_window_size_type(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pos_folding={"clf": (lambda x: x, {})},
                window_size="bad"
            )


    def test_invalid_window_size_value(self, dummy_target):
        with pytest.raises(ValueError):
            classification_kfold(
                dummy_target,
                start_window=0.0,
                pos_folding={"clf": (lambda x: x, {})},
                window_size=-1
            )


    # ==========================================================
    # SECTION 6 — Full Pipeline Execution
    # ==========================================================

    def test_full_pipeline_execution(self, dummy_target, monkeypatch):

        # --- Mock crop ---
        monkeypatch.setattr(
            "bciflow.modules.core.kfold.util.crop",
            lambda data, tmin, window_size, inplace: data
        )

        # --- Mock apply_prefold ---
        monkeypatch.setattr(
            "bciflow.modules.core.kfold.apply_prefold",
            lambda **kwargs: {0.0: dummy_target}
        )

        # --- Mock apply_posfold ---
        def fake_posfold(**kwargs):
            return [[
                kwargs["fold_id"],
                0.0,
                "A",
                0.5,
                0.5
            ]]

        monkeypatch.setattr(
            "bciflow.modules.core.kfold.apply_posfold",
            fake_posfold
        )

        class DummyClf:
            def fit(self, X, y): return self
            def predict_proba(self, X): return np.zeros((len(X), 2))

        df = classification_kfold(
            target=dummy_target,
            start_window=0.0,          # float → list branch
            start_test_window=None,    # None branch
            pre_folding=None,
            pos_folding={"clf": (DummyClf(), {})},
            window_size=1.0
        )

        assert isinstance(df, pd.DataFrame)
        assert "fold" in df.columns
        assert "tmin" in df.columns
        assert "true_label" in df.columns


    # ==========================================================
    # SECTION 7 — start_test_window Float Branch
    # ==========================================================

    def test_start_test_window_float_branch(self, dummy_target, monkeypatch):

        monkeypatch.setattr(
            "bciflow.modules.core.kfold.util.crop",
            lambda data, tmin, window_size, inplace: data
        )

        monkeypatch.setattr(
            "bciflow.modules.core.kfold.apply_prefold",
            lambda **kwargs: {0.5: dummy_target}
        )

        def fake_posfold(**kwargs):
            return [[
                kwargs["fold_id"],   # fold
                0.5,                 # tmin
                "A",                 # true_label
                0.5,                 # prob A
                0.5                  # prob B
            ]]

        monkeypatch.setattr(
            "bciflow.modules.core.kfold.apply_posfold",
            fake_posfold
        )

        class DummyClf:
            def fit(self, X, y): return self
            def predict_proba(self, X): return np.zeros((len(X), 2))

        classification_kfold(
            target=dummy_target,
            start_window=[0.0],
            start_test_window=0.5,  # força branch float
            pos_folding={"clf": (DummyClf(), {})},
            window_size=1.0
        )