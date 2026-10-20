"""Mass-independent Cartesian rigid-invariance enforcement for molecular Hessians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .rigid import mass_weighted_rigid_basis

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RigidInvarianceResult:
    """A Cartesian Hessian with geometric rigid-motion contamination removed."""

    hessian: FloatArray
    projector: FloatArray
    rigid_basis: FloatArray
    vibrational_basis: FloatArray
    rigid_rank: int
    is_linear: bool
    raw_rigid_block_norm: float
    raw_rigid_vibrational_coupling_norm: float
    correction_max_abs: float
    correction_frobenius: float
    correction_relative_frobenius: float


def enforce_cartesian_rigid_invariance(
    positions: FloatArray,
    hessian: FloatArray,
    *,
    linearity_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-10,
) -> RigidInvarianceResult:
    """Project a Cartesian Hessian onto the geometric vibrational subspace.

    The translational and rotational displacement span is a geometric property
    of the Cartesian structure and must not depend on isotope masses.  It is
    therefore constructed with equal auxiliary weights.  The returned Hessian
    is ``P K P`` with ``P`` the Euclidean projector orthogonal to that span.

    This operation removes finite-precision violations of the molecular
    acoustic and rotational sum rules before physical mass weighting.  It does
    not alter the molecular geometry or introduce a model-dependent mode
    deletion.  For an exactly invariant Hessian the correction is zero up to
    roundoff.
    """
    positions = np.asarray(positions, dtype=np.float64)
    hessian = np.asarray(hessian, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3).")
    n_cart = positions.size
    if hessian.shape != (n_cart, n_cart):
        raise ValueError("hessian must have shape (3*n_atoms, 3*n_atoms).")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(hessian)):
        raise ValueError("positions and hessian must contain only finite values.")
    if linearity_tolerance <= 0.0:
        raise ValueError("linearity_tolerance must be strictly positive.")
    if symmetry_tolerance <= 0.0:
        raise ValueError("symmetry_tolerance must be strictly positive.")

    asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    scale = max(float(np.max(np.abs(hessian))), 1.0)
    if asymmetry > symmetry_tolerance * scale:
        raise ValueError(
            "hessian must be symmetric within the declared numerical tolerance."
        )
    hessian = 0.5 * (hessian + hessian.T)

    geometric = mass_weighted_rigid_basis(
        positions,
        np.ones(positions.shape[0], dtype=np.float64),
        linearity_tolerance=linearity_tolerance,
    )
    q = geometric.rigid_basis
    z = geometric.vibrational_basis
    projector = geometric.projector

    raw_rigid_block_norm = float(np.linalg.norm(q.T @ hessian @ q))
    raw_rigid_vibrational_coupling_norm = float(np.linalg.norm(q.T @ hessian @ z))

    corrected = projector @ hessian @ projector
    corrected = 0.5 * (corrected + corrected.T)
    correction = corrected - hessian
    correction_frobenius = float(np.linalg.norm(correction))
    hessian_frobenius = float(np.linalg.norm(hessian))

    return RigidInvarianceResult(
        hessian=corrected,
        projector=projector,
        rigid_basis=q,
        vibrational_basis=z,
        rigid_rank=q.shape[1],
        is_linear=geometric.is_linear,
        raw_rigid_block_norm=raw_rigid_block_norm,
        raw_rigid_vibrational_coupling_norm=raw_rigid_vibrational_coupling_norm,
        correction_max_abs=float(np.max(np.abs(correction))),
        correction_frobenius=correction_frobenius,
        correction_relative_frobenius=(
            correction_frobenius / hessian_frobenius
            if hessian_frobenius > 0.0
            else correction_frobenius
        ),
    )
