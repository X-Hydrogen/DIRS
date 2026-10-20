from __future__ import annotations

import numpy as np
import pytest

from dirs.archive import load_hessian_npz, save_hessian_npz
from dirs.backends import HessianData


def _record() -> HessianData:
    hessian = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.1, 0.8, 0.2],
            [0.0, 0.2, 0.7],
        ],
        dtype=float,
    )
    return HessianData(
        symbols=("H",),
        positions_bohr=np.array([[0.1, -0.2, 0.3]]),
        masses_amu=np.array([1.00784]),
        hessian_hartree_per_bohr2=hessian,
        source_format="synthetic",
        energy_hartree=-0.5,
        multiplicity=2,
        max_hessian_asymmetry=0.0,
    )


def test_hessian_archive_round_trip(tmp_path) -> None:
    path = tmp_path / "record.npz"
    original = _record()
    save_hessian_npz(original, path)
    loaded = load_hessian_npz(path)
    assert loaded.symbols == original.symbols
    assert loaded.source_format == original.source_format
    assert loaded.energy_hartree == original.energy_hartree
    assert loaded.multiplicity == original.multiplicity
    assert np.allclose(loaded.positions_bohr, original.positions_bohr)
    assert np.allclose(loaded.masses_amu, original.masses_amu)
    assert np.allclose(
        loaded.hessian_hartree_per_bohr2,
        original.hessian_hartree_per_bohr2,
    )


def test_hessian_archive_preserves_missing_optional_metadata(tmp_path) -> None:
    original = _record()
    without_optional = HessianData(
        symbols=original.symbols,
        positions_bohr=original.positions_bohr,
        masses_amu=original.masses_amu,
        hessian_hartree_per_bohr2=original.hessian_hartree_per_bohr2,
        source_format=original.source_format,
    )
    path = tmp_path / "missing.npz"
    save_hessian_npz(without_optional, path)
    loaded = load_hessian_npz(path)
    assert loaded.energy_hartree is None
    assert loaded.multiplicity is None


def test_hessian_archive_rejects_unexpected_keys(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, unexpected=np.array([1.0]))
    with pytest.raises(ValueError, match="Unexpected Hessian archive keys"):
        load_hessian_npz(path)


def test_raw_asymmetry_survives_archive_round_trip(tmp_path) -> None:
    """Raw Hessian asymmetry must be preserved through save/load."""
    hessian = np.array(
        [
            [1.0, 0.10000001, 0.0],
            [0.1, 0.8, 0.2],
            [0.0, 0.2, 0.7],
        ],
        dtype=float,
    )
    raw_asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    # Archive stores symmetrized Hessian; raw asymmetry is metadata only
    hessian_sym = 0.5 * (hessian + hessian.T)
    record = HessianData(
        symbols=("H",),
        positions_bohr=np.array([[0.0, 0.0, 0.0]]),
        masses_amu=np.array([1.00784]),
        hessian_hartree_per_bohr2=hessian_sym,
        source_format="synthetic",
        max_hessian_asymmetry=raw_asymmetry,
    )
    path = tmp_path / "asymmetry.npz"
    save_hessian_npz(record, path)
    loaded = load_hessian_npz(path)
    # The raw asymmetry must survive: stored value must be >= the measured value
    assert loaded.max_hessian_asymmetry >= raw_asymmetry
    # The loaded Hessian is symmetrized, so its measured asymmetry should be ~0
    loaded_measured = float(np.max(np.abs(
        loaded.hessian_hartree_per_bohr2 - loaded.hessian_hartree_per_bohr2.T
    )))
    assert loaded_measured < 1e-15
