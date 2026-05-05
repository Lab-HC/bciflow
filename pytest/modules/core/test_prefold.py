import numpy as np
import pytest
from bciflow.modules.core.prefold import apply_prefold
# ==========================================================
# Test Suite: apply_prefold
# ==========================================================
#
# Cobertura:
#   1. Validação estrutural de entrada
#   2. Validação semântica de parâmetros
#   3. Normalização de tipos (float → list)
#   4. Execução de pre_folding com:
#        - função callable
#        - objeto com método transform
#        - função estilo transform
#   5. Múltiplas janelas
#   6. Casos degenerados (prefolding vazio)
#   7. Objetos inválidos
#
# ==========================================================

class TestApplyPrefold:

    # ==========================================================
    # Fixtures
    # ==========================================================

    @pytest.fixture
    def dummy_target(self):
        """
        Synthetic deterministic EEG-like dataset.

        Structure:
            X        : (trials, channels, samples)
            y        : labels
            sfreq    : sampling frequency
            y_dict   : label mapping
            events   : event indices
            ch_names : channel names
            tmin     : initial time
        """
        np.random.seed(42)

        return {
            "X": np.random.randn(10, 8, 256),
            "y": np.array([0, 1] * 5),
            "sfreq": 128,
            "y_dict": {0: "A", 1: "B"},
            "events": np.arange(10),
            "ch_names": [f"ch{i}" for i in range(8)],
            "tmin": 0.0
        }


    # ==========================================================
    # SECTION 1 — Structural Input Validation
    # ==========================================================

    def test_target_must_be_dict(self):
        """Target must be a dictionary."""
        with pytest.raises(ValueError):
            apply_prefold("not_a_dict", 0.0, None, 1.0, {})


    def test_missing_required_key(self, dummy_target):
        """Target must contain mandatory keys."""
        bad_target = dummy_target.copy()
        del bad_target["X"]

        with pytest.raises(ValueError):
            apply_prefold(bad_target, 0.0, None, 1.0, {})


    def test_invalid_prefolding_type(self, dummy_target):
        """pre_folding must be a dictionary."""
        with pytest.raises(ValueError):
            apply_prefold(dummy_target, 0.0, None, 1.0, "not_a_dict")


    def test_invalid_prefolding_entry_structure(self, dummy_target):
        """Each pre_folding entry must be a (callable, dict) tuple."""
        with pytest.raises(ValueError):
            apply_prefold(dummy_target, 0.0, None, 1.0, {"bp": "wrong_structure"})


    def test_prefolding_function_must_be_callable(self, dummy_target):
        """First element of tuple must be callable."""
        with pytest.raises(ValueError):
            apply_prefold(dummy_target, 0.0, None, 1.0,
                        {"bp": ("not_callable", {})})


    def test_prefolding_params_must_be_dict(self,dummy_target):
        """Second element of tuple must be a parameter dictionary."""
        def dummy_func(x):
            return x

        with pytest.raises(ValueError):
            apply_prefold(dummy_target, 0.0, None, 1.0,
                        {"bp": (dummy_func, "not_a_dict")})
    
    # ==========================================================
    # SECTION 2 — Parameter Type and Value Validation
    # ==========================================================

    def test_invalid_start_window_type(self, dummy_target):
        """start_window must be float or list of floats."""
        with pytest.raises(ValueError):
            apply_prefold(dummy_target, "invalid", None, 1.0, {})


    def test_invalid_start_test_window_type(self, dummy_target):
        """start_test_window must be float, list of floats, or None."""
        with pytest.raises(ValueError):
            apply_prefold(dummy_target, 0.0, "invalid", 1.0, {})


    def test_start_window_list_with_invalid_type(self, dummy_target):
        """All elements in start_window list must be float."""
        with pytest.raises(ValueError,
                        match="start_window list must contain only float values"):
            apply_prefold(dummy_target,
                        [0.0, "invalid"],
                        [0.0],
                        1.0,
                        {})


    def test_start_test_window_list_with_invalid_type(self, dummy_target):
        """All elements in start_test_window list must be float."""
        with pytest.raises(ValueError,
                        match="start_test_window list must contain only float values"):
            apply_prefold(dummy_target,
                        [0.0],
                        [0.0, "invalid"],
                        1.0,
                        {})


    def test_window_size_invalid_type(self, dummy_target):
        """window_size must be float."""
        with pytest.raises(ValueError,
                        match="window_size has to be a float type value"):
            apply_prefold(dummy_target,
                        [0.0],
                        [0.0],
                        "invalid",
                        {})


    def test_window_size_must_be_positive(self, dummy_target):
        """window_size must be strictly positive."""
        with pytest.raises(ValueError):
            apply_prefold(dummy_target, 0.0, None, -1.0, {})


    # ==========================================================
    # SECTION 3 — Window Normalization Behavior
    # ==========================================================

    def test_start_window_float_conversion(self, dummy_target, monkeypatch):
        """Float start_window must be internally converted to list."""

        def fake_apply(*args, **kwargs):
            return {"processed": True}

        monkeypatch.setattr(
            "bciflow.modules.core.prefold.util.apply_to_trials",
            fake_apply
        )

        result = apply_prefold(dummy_target,
                            0.0,
                            None,
                            1.0,
                            {})

        assert list(result.keys()) == [0.0]


    def test_start_test_window_float(self, dummy_target, monkeypatch):
        """Float start_test_window must be internally converted to list."""

        def fake_apply(*args, **kwargs):
            return {"processed": True}

        monkeypatch.setattr(
            "bciflow.modules.core.prefold.util.apply_to_trials",
            fake_apply
        )

        result = apply_prefold(dummy_target,
                            [0.0],
                            0.5,
                            1.0,
                            {})

        assert list(result.keys()) == [0.5]


    # ==========================================================
    # SECTION 4 — Functional Behavior (Monkeypatched Execution)
    # ==========================================================

    def test_prefolding_calls_apply_to_trials(self, dummy_target, monkeypatch):
        """Callable functions must be executed via apply_to_trials."""
        call_counter = {"count": 0}

        def fake_apply_to_trials(data, func, func_param, inplace):
            call_counter["count"] += 1
            return {"processed": True}

        monkeypatch.setattr(
            "bciflow.modules.core.prefold.util.apply_to_trials",
            fake_apply_to_trials
        )

        def dummy_func(x, low, high):
            return x

        result = apply_prefold(dummy_target,
                            0.0,
                            0.5,
                            1.0,
                            {"bp": (dummy_func, {"low": 8, "high": 30})})

        assert call_counter["count"] == 1
        assert 0.5 in result


    # ==========================================================
    # SECTION 5 — Object-Based Transform Execution
    # ==========================================================

    class DummyTransformer:
        """Object exposing a transform method."""
        def transform(self, x, scale=1):
            return x * scale


    def test_prefolding_object_transform_class(self, dummy_target, monkeypatch):
        """Objects with .transform must be accepted."""

        def fake_apply_to_trials(data, func, func_param, inplace):
            return {"used_transform": True}

        monkeypatch.setattr(
            "bciflow.modules.core.prefold.util.apply_to_trials",
            fake_apply_to_trials
        )

        transformer = self.DummyTransformer()

        result = apply_prefold(dummy_target,
                            0.0,
                            0.5,
                            1.0,
                            {"scale": (transformer, {"scale": 2})})

        assert 0.5 in result
        assert result[0.5]["used_transform"] is True


    def test_prefolding_object_transform_function(self, dummy_target, monkeypatch):
        """Functions behaving like transform should also be valid."""

        def fake_apply_to_trials(data, func, func_param, inplace):
            return {"ok": True}

        monkeypatch.setattr(
            "bciflow.modules.core.prefold.util.apply_to_trials",
            fake_apply_to_trials
        )

        def dummyTransform(x, scale=1):
            return x * scale

        result = apply_prefold(dummy_target,
                            0.0,
                            0.5,
                            1.0,
                            {"scale": (dummyTransform, {"scale": 2})})

        assert 0.5 in result


    # ==========================================================
    # SECTION 6 — Multiple Windows and Edge Cases
    # ==========================================================

    def test_multiple_test_windows(self, dummy_target, monkeypatch):
        """Multiple test windows must generate multiple outputs."""

        def fake_apply_to_trials(data, func, func_param, inplace):
            return {"processed": True}

        monkeypatch.setattr(
            "bciflow.modules.core.prefold.util.apply_to_trials",
            fake_apply_to_trials
        )

        result = apply_prefold(dummy_target,
                            [0.0],
                            [0.0, 0.5, 1.0],
                            1.0,
                            {})

        assert len(result.keys()) == 3
        assert set(result.keys()) == {0.0, 0.5, 1.0}


    def test_empty_prefolding_still_creates_windows(self, dummy_target):
        """Even without pre_folding, window structure must exist."""
        result = apply_prefold(dummy_target,
                            [0.0],
                            [0.0, 0.5],
                            1.0,
                            {})

        assert set(result.keys()) == {0.0, 0.5}


    # ==========================================================
    # SECTION 7 — Invalid Object Handling
    # ==========================================================

    def test_prefolding_invalid_object(self, dummy_target):
        """Objects without callable or transform must raise error."""

        class Invalid:
            pass

        with pytest.raises(ValueError):
            apply_prefold(dummy_target,
                        [0.0],
                        [0.0],
                        1.0,
                        {"invalid": (Invalid(), {})})
        
   