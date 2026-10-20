"""Strict Hessian data adapters for ORCA and PySCF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .units import amu_to_electron_masses

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HessianData:
    """Backend-neutral Cartesian Hessian record."""

    symbols: tuple[str, ...]
    positions_bohr: FloatArray
    masses_amu: FloatArray
    hessian_hartree_per_bohr2: FloatArray
    source_format: str
    energy_hartree: float | None = None
    multiplicity: int | None = None
    max_hessian_asymmetry: float = 0.0

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def masses_electron(self) -> FloatArray:
        return amu_to_electron_masses(self.masses_amu)

    def molecular_inputs(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return copies in the Stage 0B atomic-unit convention."""
        return (
            self.positions_bohr.copy(),
            self.masses_electron.copy(),
            self.hessian_hartree_per_bohr2.copy(),
        )


def _validate_record(
    symbols: Iterable[str],
    positions_bohr: FloatArray,
    masses_amu: FloatArray,
    hessian: FloatArray,
    *,
    source_format: str,
    energy_hartree: float | None = None,
    multiplicity: int | None = None,
    symmetry_tolerance: float = 1.0e-10,
    max_asymmetry: float | None = None,
) -> HessianData:
    symbols_tuple = tuple(str(symbol) for symbol in symbols)
    positions = np.asarray(positions_bohr, dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)
    hessian_array = np.asarray(hessian, dtype=np.float64)
    n_atoms = len(symbols_tuple)

    if n_atoms == 0:
        raise ValueError("At least one atom is required.")
    if positions.shape != (n_atoms, 3):
        raise ValueError("positions_bohr must have shape (n_atoms, 3).")
    if masses.shape != (n_atoms,):
        raise ValueError("masses_amu must have shape (n_atoms,).")
    if hessian_array.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError("Cartesian Hessian shape must be (3*n_atoms, 3*n_atoms).")
    if not all(
        symbol and not any(character.isspace() for character in symbol)
        for symbol in symbols_tuple
    ):
        raise ValueError("Each atomic symbol must be a nonempty token.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_bohr contains a non-finite value.")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_amu must be finite and strictly positive.")
    if not np.all(np.isfinite(hessian_array)):
        raise ValueError("Hessian contains a non-finite value.")
    if symmetry_tolerance <= 0.0:
        raise ValueError("symmetry_tolerance must be strictly positive.")

    # Measure raw asymmetry from the supplied Hessian (before any symmetrization)
    measured_asymmetry = float(np.max(np.abs(hessian_array - hessian_array.T)))
    # Use caller-supplied raw asymmetry if provided (captured before symmetrization)
    stored_asymmetry = max_asymmetry if max_asymmetry is not None else measured_asymmetry
    scale = max(float(np.max(np.abs(hessian_array))), 1.0)
    if measured_asymmetry > symmetry_tolerance * scale:
        raise ValueError(
            f"Cartesian Hessian is not symmetric: max asymmetry {measured_asymmetry:.6e}."
        )
    hessian_array = 0.5 * (hessian_array + hessian_array.T)

    if multiplicity is not None and multiplicity < 1:
        raise ValueError("multiplicity must be a positive integer.")

    return HessianData(
        symbols=symbols_tuple,
        positions_bohr=positions.copy(),
        masses_amu=masses.copy(),
        hessian_hartree_per_bohr2=hessian_array,
        source_format=source_format,
        energy_hartree=energy_hartree,
        multiplicity=multiplicity,
        max_hessian_asymmetry=max(stored_asymmetry, measured_asymmetry),
    )


def _unique_section(lines: list[str], name: str) -> int:
    matches = [index for index, line in enumerate(lines) if line.strip() == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} section; found {len(matches)}.")
    return matches[0]


def _optional_scalar(lines: list[str], name: str, converter: Any) -> Any | None:
    matches = [index for index, line in enumerate(lines) if line.strip() == name]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected at most one {name} section; found {len(matches)}.")
    index = matches[0] + 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise ValueError(f"Missing value after {name}.")
    tokens = lines[index].split()
    if len(tokens) != 1:
        raise ValueError(f"Expected one scalar after {name}.")
    return converter(tokens[0])


def _parse_orca_block_matrix(lines: list[str], start: int, dimension: int) -> FloatArray:
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    line_index = start
    next_column = 0

    while next_column < dimension:
        while line_index < len(lines) and not lines[line_index].strip():
            line_index += 1
        if line_index >= len(lines):
            raise ValueError("Unexpected end of file in ORCA Hessian matrix.")
        header_tokens = lines[line_index].split()
        try:
            columns = [int(token) for token in header_tokens]
        except ValueError as exc:
            raise ValueError("Invalid ORCA Hessian column-header line.") from exc
        if not columns:
            raise ValueError("Empty ORCA Hessian column-header line.")
        expected = list(range(next_column, next_column + len(columns)))
        if columns != expected:
            raise ValueError(
                f"ORCA Hessian columns must be contiguous; expected {expected}, got {columns}."
            )
        if columns[-1] >= dimension:
            raise ValueError("ORCA Hessian column index exceeds declared dimension.")
        line_index += 1

        for expected_row in range(dimension):
            if line_index >= len(lines):
                raise ValueError("Unexpected end of file in ORCA Hessian rows.")
            tokens = lines[line_index].split()
            if len(tokens) != 1 + len(columns):
                raise ValueError(
                    "ORCA Hessian row width does not match the active column block."
                )
            try:
                row_index = int(tokens[0])
                values = [
                    float(token.replace("D", "E").replace("d", "e"))
                    for token in tokens[1:]
                ]
            except ValueError as exc:
                raise ValueError("Invalid numeric token in ORCA Hessian matrix.") from exc
            if row_index != expected_row:
                raise ValueError(
                    f"Expected ORCA Hessian row {expected_row}, got {row_index}."
                )
            matrix[row_index, columns] = values
            line_index += 1
        next_column += len(columns)

    return matrix


def parse_orca_hess_text(
    text: str,
    *,
    symmetry_tolerance: float = 1.0e-10,
) -> HessianData:
    """Parse the required `$hessian` and `$atoms` sections of an ORCA file.

    ORCA stores the Cartesian Hessian in Hartree/bohr^2 and the `$atoms`
    coordinates in bohr with masses in atomic mass units.
    """
    lines = text.splitlines()
    hessian_section = _unique_section(lines, "$hessian")
    atoms_section = _unique_section(lines, "$atoms")

    if hessian_section + 1 >= len(lines):
        raise ValueError("Missing dimension after $hessian.")
    try:
        dimension = int(lines[hessian_section + 1].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("Invalid dimension after $hessian.") from exc
    if dimension <= 0 or dimension % 3 != 0:
        raise ValueError("ORCA Hessian dimension must be a positive multiple of three.")
    hessian = _parse_orca_block_matrix(lines, hessian_section + 2, dimension)

    if atoms_section + 1 >= len(lines):
        raise ValueError("Missing atom count after $atoms.")
    try:
        n_atoms = int(lines[atoms_section + 1].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("Invalid atom count after $atoms.") from exc
    if n_atoms * 3 != dimension:
        raise ValueError("$atoms count is inconsistent with the Hessian dimension.")

    symbols: list[str] = []
    masses: list[float] = []
    positions: list[list[float]] = []
    line_index = atoms_section + 2
    for atom_index in range(n_atoms):
        if line_index >= len(lines):
            raise ValueError("Unexpected end of file in $atoms section.")
        tokens = lines[line_index].split()
        if len(tokens) != 5:
            raise ValueError(f"ORCA atom row {atom_index} must contain five tokens.")
        symbol = tokens[0]
        try:
            mass = float(tokens[1].replace("D", "E").replace("d", "e"))
            coordinates = [
                float(token.replace("D", "E").replace("d", "e"))
                for token in tokens[2:5]
            ]
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric token in ORCA atom row {atom_index}."
            ) from exc
        symbols.append(symbol)
        masses.append(mass)
        positions.append(coordinates)
        line_index += 1

    energy = _optional_scalar(lines, "$act_energy", float)
    multiplicity = _optional_scalar(lines, "$multiplicity", int)
    return _validate_record(
        symbols,
        np.asarray(positions, dtype=np.float64),
        np.asarray(masses, dtype=np.float64),
        hessian,
        source_format="orca-hess",
        energy_hartree=energy,
        multiplicity=multiplicity,
        symmetry_tolerance=symmetry_tolerance,
    )


def parse_orca_hess(
    path: str | Path,
    *,
    symmetry_tolerance: float = 1.0e-10,
) -> HessianData:
    """Read and parse an ORCA `.hess` ASCII file."""
    path = Path(path)
    return parse_orca_hess_text(
        path.read_text(encoding="utf-8", errors="strict"),
        symmetry_tolerance=symmetry_tolerance,
    )


def pyscf_hessian_to_cartesian(hessian_blocks: FloatArray) -> FloatArray:
    """Flatten PySCF `(A,B,dR_A,dR_B)` Hessian blocks to `(3N,3N)`.

    PySCF's source examples use
    ``hessian.transpose(0, 2, 1, 3).reshape(3*N, 3*N)``.
    """
    blocks = np.asarray(hessian_blocks, dtype=np.float64)
    if (
        blocks.ndim != 4
        or blocks.shape[0] != blocks.shape[1]
        or blocks.shape[2:] != (3, 3)
    ):
        raise ValueError("PySCF Hessian must have shape (n_atoms, n_atoms, 3, 3).")
    n_atoms = blocks.shape[0]
    return blocks.transpose(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)


def hessian_data_from_pyscf_arrays(
    symbols: Iterable[str],
    positions_bohr: FloatArray,
    masses_amu: FloatArray,
    hessian_blocks: FloatArray,
    *,
    energy_hartree: float | None = None,
    multiplicity: int | None = None,
    symmetry_tolerance: float = 1.0e-10,
) -> HessianData:
    """Build backend-neutral data from explicit PySCF arrays."""
    cart = pyscf_hessian_to_cartesian(hessian_blocks)
    # Measure raw numerical asymmetry before symmetrization.
    # PySCF analytic RHF Hessians may carry backend numerical asymmetry.
    # The exact RHF Hessian is symmetric; symmetrization is applied for
    # downstream analysis but the raw asymmetry is preserved as metadata.
    raw_asymmetry = float(np.max(np.abs(cart - cart.T)))
    cart_sym = 0.5 * (cart + cart.T)
    return _validate_record(
        symbols,
        positions_bohr,
        masses_amu,
        cart_sym,
        source_format="pyscf",
        energy_hartree=energy_hartree,
        multiplicity=multiplicity,
        symmetry_tolerance=symmetry_tolerance,
        max_asymmetry=raw_asymmetry,
    )


def hessian_data_from_pyscf_molecule(
    molecule: Any,
    hessian_blocks: FloatArray,
    *,
    energy_hartree: float | None = None,
    multiplicity: int | None = None,
    symmetry_tolerance: float = 1.0e-10,
) -> HessianData:
    """Build data from a PySCF Mole-like object without importing PySCF."""
    n_atoms = int(molecule.natm)
    symbols = [molecule.atom_symbol(index) for index in range(n_atoms)]
    positions = np.asarray(molecule.atom_coords(unit="Bohr"), dtype=np.float64)
    masses = np.asarray(molecule.atom_mass_list(), dtype=np.float64)
    return hessian_data_from_pyscf_arrays(
        symbols,
        positions,
        masses,
        hessian_blocks,
        energy_hartree=energy_hartree,
        multiplicity=multiplicity,
        symmetry_tolerance=symmetry_tolerance,
    )
