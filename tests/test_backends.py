from __future__ import annotations

import numpy as np
import pytest

from dirs.backends import (
    hessian_data_from_pyscf_arrays,
    parse_orca_hess_text,
    pyscf_hessian_to_cartesian,
)


def _write_block_matrix(matrix: np.ndarray, columns_per_block: int = 5) -> str:
    lines: list[str] = []
    for start in range(0, matrix.shape[1], columns_per_block):
        columns = list(
            range(start, min(start + columns_per_block, matrix.shape[1]))
        )
        lines.append(" ".join(str(column) for column in columns))
        for row in range(matrix.shape[0]):
            values = " ".join(
                f"{matrix[row, column]:.16E}" for column in columns
            )
            lines.append(f"{row} {values}")
    return "\n".join(lines)


def _orca_text(hessian: np.ndarray, *, atom_count: int = 2) -> str:
    atoms = [
        "H 1.00784 0.0 0.0 -0.7",
        "H 1.00784 0.0 0.0 0.7",
    ][:atom_count]
    return (
        "$orca_hessian_file\n\n"
        "$act_energy\n -1.100000\n\n"
        "$multiplicity\n 1\n\n"
        "$hessian\n"
        f"{hessian.shape[0]}\n"
        f"{_write_block_matrix(hessian)}\n\n"
        "$atoms\n"
        f"{atom_count}\n"
        + "\n".join(atoms)
        + "\n"
    )


def _symmetric_hessian() -> np.ndarray:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(6, 6))
    return 0.5 * (matrix + matrix.T)


def test_orca_parser_reads_block_matrix_and_atoms() -> None:
    hessian = _symmetric_hessian()
    record = parse_orca_hess_text(_orca_text(hessian))
    assert record.symbols == ("H", "H")
    assert record.positions_bohr.shape == (2, 3)
    assert np.allclose(record.masses_amu, [1.00784, 1.00784])
    assert np.allclose(record.hessian_hartree_per_bohr2, hessian)
    assert record.energy_hartree == pytest.approx(-1.1)
    assert record.multiplicity == 1
    assert record.source_format == "orca-hess"


def test_orca_parser_rejects_atom_dimension_mismatch() -> None:
    hessian = _symmetric_hessian()
    text = _orca_text(hessian).replace("$atoms\n2\n", "$atoms\n1\n")
    with pytest.raises(ValueError, match="inconsistent"):
        parse_orca_hess_text(text)


def test_orca_parser_rejects_noncontiguous_column_header() -> None:
    hessian = _symmetric_hessian()
    text = _orca_text(hessian).replace(
        "0 1 2 3 4\n",
        "0 1 3 4 5\n",
        1,
    )
    with pytest.raises(ValueError, match="contiguous"):
        parse_orca_hess_text(text)


def test_orca_parser_rejects_material_asymmetry() -> None:
    hessian = _symmetric_hessian()
    hessian[0, 1] += 1.0e-4
    with pytest.raises(ValueError, match="not symmetric"):
        parse_orca_hess_text(_orca_text(hessian))


def test_pyscf_block_flattening_uses_atom_cartesian_order() -> None:
    blocks = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3)
    cartesian = pyscf_hessian_to_cartesian(blocks)
    for atom_a in range(2):
        for atom_b in range(2):
            for axis_a in range(3):
                for axis_b in range(3):
                    assert (
                        cartesian[
                            3 * atom_a + axis_a,
                            3 * atom_b + axis_b,
                        ]
                        == blocks[atom_a, atom_b, axis_a, axis_b]
                    )


def test_orca_and_pyscf_adapters_match_same_synthetic_data() -> None:
    hessian = _symmetric_hessian()
    orca = parse_orca_hess_text(_orca_text(hessian))
    blocks = hessian.reshape(2, 3, 2, 3).transpose(0, 2, 1, 3)
    pyscf = hessian_data_from_pyscf_arrays(
        ["H", "H"],
        orca.positions_bohr,
        orca.masses_amu,
        blocks,
        energy_hartree=-1.1,
        multiplicity=1,
    )
    assert np.allclose(
        pyscf.hessian_hartree_per_bohr2,
        orca.hessian_hartree_per_bohr2,
    )
    assert np.allclose(pyscf.positions_bohr, orca.positions_bohr)
    assert np.allclose(pyscf.masses_electron, orca.masses_electron)
