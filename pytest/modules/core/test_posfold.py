import numpy as np
import pytest
from bciflow.modules.core.posfold import apply_posfold


# ==========================================================
# Test Suite: apply_posfold
# ==========================================================
#
# Cobertura:
#   1. Validação estrutural de entradas
#   2. Validação de parâmetros
#   3. Validação de pos_folding
#   4. Normalização de janelas
#   5. Execução de transforms (function e object)
#   6. Execução do classificador
#   7. Fallback quando predict_proba não existe
#   8. Estrutura final do results
#
# ==========================================================



class TestApplyPosfold:
    # ==========================================================
    # Fixtures
    # ==========================================================
    @pytest.fixture
    def dummy_target(self):
        return {
            "X": np.random.randn(10, 8, 256),
            "y": np.array([0, 1] * 5),
            "y_dict": {0: "A", 1: "B"}
        }

    @pytest.fixture
    def dummy_target_dict(self,dummy_target):
        return {
            0.0: dummy_target.copy(),
            0.5: dummy_target.copy()
        }

    @pytest.fixture
    def train_index(self):
        return [0, 1, 2, 3, 4]

    @pytest.fixture
    def test_index(self):
        return [5, 6, 7, 8, 9]


    # ==========================================================
    # SECTION 1 — Validação estrutural de entradas
    # ==========================================================

    def test_target_dict_must_be_dict(self, dummy_target, train_index, test_index):
        with pytest.raises(ValueError):
            apply_posfold("invalid", train_index, test_index,
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_target_dict_key_must_be_float(self, dummy_target_dict, train_index, test_index, dummy_target):
        bad = {"invalid": dummy_target_dict[0.0]}
        with pytest.raises(ValueError):
            apply_posfold(bad, train_index, test_index,
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_target_dict_value_must_be_dict(self, dummy_target):
        bad_target_dict = {0.0: "not_a_dict"}
        with pytest.raises(ValueError):
            apply_posfold(bad_target_dict, [0,1], [2],
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_target_must_be_dict(self, dummy_target_dict):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        "invalid", 1, [])


    def test_target_missing_key(self, dummy_target_dict, train_index, test_index):
        bad_target = {"X": np.random.randn(10, 4), "y": np.array([0]*10)}
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, train_index, test_index,
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        bad_target, 1, [])


    def test_target_y_dict_must_be_dict(self, dummy_target, dummy_target_dict):
        bad_target = dummy_target.copy()
        bad_target["y_dict"] = "invalid"
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        bad_target, 1, [])


    def test_start_test_window_float_normalization(self, dummy_target_dict,
                                                dummy_target,
                                                train_index,
                                                test_index,
                                                monkeypatch):
        """
        Cobre linha 201:
        start_test_window = [start_test_window]
        """

        def fake_get_trial(data, ids):
            return {"X": data["X"][ids], "y": data["y"][ids]}

        def fake_concatenate(list_data):
            return list_data[0]

        def fake_find_key(d, value):
            return d[value]

        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.get_trial",
            fake_get_trial
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.concatenate",
            fake_concatenate
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.find_key_with_value",
            fake_find_key
        )

        class DummyClf:
            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                return np.ones((len(X), 2)) * 0.5

        results = apply_posfold(
            dummy_target_dict,
            train_index,
            test_index,
            start_window=[0.0],
            start_test_window=0.5,  # float → deve virar [0.5]
            pos_folding={"clf": (DummyClf(), {})},
            target=dummy_target,
            fold_id=1,
            results=[]
        )

        assert len(results) > 0

    def test_start_test_window_none(self, dummy_target_dict,
                                    dummy_target,
                                    train_index,
                                    test_index,
                                    monkeypatch):

        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.get_trial",
            lambda data, ids: data
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.concatenate",
            lambda x: x[0]
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.find_key_with_value",
            lambda d, v: "A"
        )

        class DummyClf:
            def fit(self, X, y): return self
            def predict_proba(self, X): return np.zeros((len(X), 2))

        results = apply_posfold(
            dummy_target_dict,
            train_index,
            test_index,
            0.0,          # float → força normalização
            None,         # cobre branch start_test_window is None
            {"clf": (DummyClf(), {})},
            dummy_target,
            1,
            []
        )

        assert len(results) > 0
    # ==========================================================
    # SECTION 2 — Validação de parâmetros
    # ==========================================================

    def test_train_index_must_be_iterable(self, dummy_target_dict, dummy_target, test_index):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, 123, test_index,
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_test_index_must_be_iterable(self, dummy_target, dummy_target_dict):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0,1], 123,
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_fold_id_invalid(self, dummy_target_dict, dummy_target, train_index, test_index):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, train_index, test_index,
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 0, [])


    def test_fold_id_must_be_int(self, dummy_target, dummy_target_dict):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, "invalid", [])


    def test_results_must_be_list(self, dummy_target_dict, dummy_target, train_index, test_index):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, train_index, test_index,
                        [0.0], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, "invalid")


    # ==========================================================
    # SECTION 3 — Validação de pos_folding
    # ==========================================================

    def test_pos_folding_must_be_dict(self, dummy_target_dict, dummy_target, train_index, test_index):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, train_index, test_index,
                        [0.0], [0.0],
                        "invalid",
                        dummy_target, 1, [])


    def test_pos_folding_must_contain_clf(self, dummy_target_dict, dummy_target, train_index, test_index):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, train_index, test_index,
                        [0.0], [0.0],
                        {},
                        dummy_target, 1, [])


    def test_pos_folding_invalid_tuple_structure(self, dummy_target_dict, dummy_target, train_index, test_index):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, train_index, test_index,
                        [0.0], [0.0],
                        {"clf": "wrong"},
                        dummy_target, 1, [])


    def test_pos_folding_params_must_be_dict(self, dummy_target, dummy_target_dict):
        def dummy(x): return x
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        [0.0], [0.0],
                        {"transform": (dummy, "invalid"),
                        "clf": (dummy, {})},
                        dummy_target, 1, [])


    def test_pos_folding_step_must_be_callable_or_fit(self, dummy_target, dummy_target_dict):
        class Invalid: pass
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        [0.0], [0.0],
                        {"transform": (Invalid(), {}),
                        "clf": (lambda x: x, {})},
                        dummy_target, 1, [])

    def test_posfold_function_branch(self, dummy_target_dict,
                                    dummy_target,
                                    train_index,
                                    test_index,
                                    monkeypatch):

        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.get_trial",
            lambda data, ids: data
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.concatenate",
            lambda x: x[0]
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.find_key_with_value",
            lambda d, v: "A"
        )

        def transform(data, scale=1):
            return data

        class DummyClf:
            def fit(self, X, y): return self
            def predict_proba(self, X): return np.zeros((len(X), 2))

        results = apply_posfold(
            dummy_target_dict,
            train_index,
            test_index,
            [0.0],
            [0.0],
            {"scale": (transform, {"scale": 2}),
            "clf": (DummyClf(), {})},
            dummy_target,
            1,
            []
        )

        assert len(results) > 0

    def test_classifier_without_predict_proba(self, dummy_target_dict,
                                            dummy_target,
                                            train_index,
                                            test_index,
                                            monkeypatch):

        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.get_trial",
            lambda data, ids: data
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.concatenate",
            lambda x: x[0]
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.find_key_with_value",
            lambda d, v: "A"
        )

        class DummyClf:
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))  # sem predict_proba

        results = apply_posfold(
            dummy_target_dict,
            train_index,
            test_index,
            [0.0],
            [0.0],
            {"clf": (DummyClf(), {})},
            dummy_target,
            1,
            []
        )

        assert len(results) > 0

    # ==========================================================
    # SECTION 4 — Normalização de janelas
    # ==========================================================

    def test_start_window_invalid_type(self, dummy_target, dummy_target_dict):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        "invalid", [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_start_window_list_invalid_value(self, dummy_target, dummy_target_dict):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        [0.0, "invalid"], [0.0],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_start_test_window_list_invalid_value(self, dummy_target, dummy_target_dict):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, [0], [1],
                        [0.0], [0.0, "invalid"],
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    def test_start_test_window_invalid_type_branch(self, dummy_target_dict,
                                                dummy_target,
                                                train_index,
                                                test_index):
        with pytest.raises(ValueError):
            apply_posfold(dummy_target_dict, train_index, test_index,
                        [0.0], {"invalid": 1},
                        {"clf": (lambda x: x, {})},
                        dummy_target, 1, [])


    # ==========================================================
    # SECTION 5 — Execução de transforms
    # ==========================================================

    def test_posfold_object_transform(self, dummy_target, dummy_target_dict, monkeypatch):

        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.get_trial",
            lambda data, ids: data
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.concatenate",
            lambda x: x[0]
        )
        monkeypatch.setattr(
            "bciflow.modules.core.posfold.util.find_key_with_value",
            lambda d, v: "A"
        )

        class DummyTransformer:
            def fit(self, data, scale=1): return self
            def fit_transform(self, data, scale=1): return data
            def transform(self, data): return data

        class DummyClf:
            def fit(self, X, y): return self
            def predict_proba(self, X): return np.zeros((len(X), 2))

        results = apply_posfold(
            dummy_target_dict, [0], [1],
            [0.0], [0.0],
            {"scale": (DummyTransformer(), {"scale": 2}),
            "clf": (DummyClf(), {})},
            dummy_target, 1, []
        )

        assert len(results) > 0