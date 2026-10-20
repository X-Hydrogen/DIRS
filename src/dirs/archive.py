"""Compact, pickle-free serialization of backend-neutral Hessian records."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends import HessianData

_REQUIRED_KEYS = {
    "symbols",
    "positions_bohr",
    "masses_amu",
    "hessian_hartree_per_bohr2",
    "source_format",
    "energy_hartree",
    "multiplicity",
    "max_hessian_asymmetry",
}


def save_hessian_npz(record: HessianData, path: str | Path) -> None:
    """Write a compressed Hessian record without Python object arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    energy = np.nan if record.energy_hartree is None else record.energy_hartree
    multiplicity = -1 if record.multiplicity is None else record.multiplicity
    np.savez_compressed(
        path,
        symbols=np.asarray(record.symbols, dtype=np.str_),
        positions_bohr=np.asarray(record.positions_bohr, dtype=np.float64),
        masses_amu=np.asarray(record.masses_amu, dtype=np.float64),
        hessian_hartree_per_bohr2=np.asarray(
            record.hessian_hartree_per_bohr2,
            dtype=np.float64,
        ),
        source_format=np.asarray(record.source_format, dtype=np.str_),
        energy_hartree=np.asarray(energy, dtype=np.float64),
        multiplicity=np.asarray(multiplicity, dtype=np.int64),
        max_hessian_asymmetry=np.asarray(
            record.max_hessian_asymmetry,
            dtype=np.float64,
        ),
    )


def load_hessian_npz(
    path: str | Path,
    *,
    symmetry_tolerance: float = 1.0e-10,
) -> HessianData:
    """Load and strictly validate a compact Hessian record."""
    if symmetry_tolerance <= 0.0:
        raise ValueError("symmetry_tolerance must be strictly positive.")
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != _REQUIRED_KEYS:
            missing = sorted(_REQUIRED_KEYS - keys)
            extra = sorted(keys - _REQUIRED_KEYS)
            raise ValueError(f"Unexpected Hessian archive keys; missing={missing}, extra={extra}.")
        symbols = tuple(str(value) for value in archive["symbols"].tolist())
        positions = np.asarray(archive["positions_bohr"], dtype=np.float64)
        masses = np.asarray(archive["masses_amu"], dtype=np.float64)
        hessian = np.asarray(
            archive["hessian_hartree_per_bohr2"],
            dtype=np.float64,
        )
        source_value = np.asarray(archive["source_format"])
        if source_value.ndim != 0:
            raise ValueError("source_format must be a scalar string.")
        source_format = str(source_value.item())
        energy_value = np.asarray(archive["energy_hartree"], dtype=np.float64)
        multiplicity_value = np.asarray(archive["multiplicity"], dtype=np.int64)
        asymmetry_value = np.asarray(
            archive["max_hessian_asymmetry"],
            dtype=np.float64,
        )
        if energy_value.ndim != 0 or multiplicity_value.ndim != 0 or asymmetry_value.ndim != 0:
            raise ValueError("Scalar metadata arrays must be zero-dimensional.")
        energy_raw = float(energy_value.item())
        multiplicity_raw = int(multiplicity_value.item())
        stored_asymmetry = float(asymmetry_value.item())

    n_atoms = len(symbols)
    if n_atoms == 0:
        raise ValueError("Archive must contain at least one atom.")
    if positions.shape != (n_atoms, 3):
        raise ValueError("Archived positions have an invalid shape.")
    if masses.shape != (n_atoms,) or np.any(masses <= 0.0):
        raise ValueError("Archived masses have an invalid shape or value.")
    if hessian.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError("Archived Hessian has an invalid shape.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(masses)):
        raise ValueError("Archived geometry or masses contain non-finite values.")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("Archived Hessian contains non-finite values.")
    measured_asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    scale = max(float(np.max(np.abs(hessian))), 1.0)
    if measured_asymmetry > symmetry_tolerance * scale:
        raise ValueError("Archived Hessian is not symmetric within tolerance.")
    if stored_asymmetry < 0.0 or not np.isfinite(stored_asymmetry):
        raise ValueError("Archived asymmetry metadata is invalid.")
    if not source_format:
        raise ValueError("Archived source_format is empty.")
    energy = None if np.isnan(energy_raw) else energy_raw
    if not np.isnan(energy_raw) and not np.isfinite(energy_raw):
        raise ValueError("Archived energy is invalid.")
    multiplicity = None if multiplicity_raw == -1 else multiplicity_raw
    if multiplicity is not None and multiplicity < 1:
        raise ValueError("Archived multiplicity is invalid.")

    return HessianData(
        symbols=symbols,
        positions_bohr=positions.copy(),
        masses_amu=masses.copy(),
        hessian_hartree_per_bohr2=0.5 * (hessian + hessian.T),
        source_format=source_format,
        energy_hartree=energy,
        multiplicity=multiplicity,
        max_hessian_asymmetry=max(stored_asymmetry, measured_asymmetry),
    )
