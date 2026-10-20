"""Ensure the external independent comparison actually rejects incorrect data."""
import numpy as np
import pytest

from scripts.independent_check import compare


@pytest.mark.parametrize(
    "field,calculated,published", [
        ("eta", 0.25, 0.30),
        ("A", 0.25, 0.30),
        ("C_angstrom_per_mdyn", 1.25, 1.30),
        ("high", 0.98, 0.95),
    ],
)
def test_independent_reproduction_gate_rejects_disagreement(field, calculated, published):
    with pytest.raises(AssertionError, match="independent="):
        compare({field: calculated}, {field: published}, {field: field},
                "synthetic_wrong_value", [])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_independent_reproduction_gate_rejects_nonfinite(value):
    with pytest.raises(AssertionError, match="nonfinite"):
        compare({"eta": value}, {"eta": 0.0}, {"eta": "eta"},
                "synthetic_nonfinite", [])


def test_independent_reproduction_gate_accepts_valid_floating_difference():
    observed = []
    compare({"eta": 0.6 + 1e-10}, {"eta": 0.6}, {"eta": "eta"},
            "synthetic_machine_roundoff", observed)
    assert len(observed) == 1
