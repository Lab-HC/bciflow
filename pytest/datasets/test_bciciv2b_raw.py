import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.bciciv2b_raw import bciciv2b_raw

class TestBCICIV2B:


    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            bciciv2b_raw(subject="1")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            bciciv2b_raw(subject=10)