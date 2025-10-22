"""
This module implements functions specifically for the use with sPHNNs
For now it only implements the check if an ISPHS is 0-GAS
"""

import jax

from .derivative_models import ISPHS
from .function_models import FICNN, LyapunovNN
from .function_models.matrices import *

valid_structure_matrices = (SkewSymmetricMatrix,
                            ConstantSkewSymmetricMatrix, 
                            ConstantSkewSymmetricMatrix,
                            SymplecticMatrix)

valid_spsd_matrices = (SPSDMatrix,
                       ConstantSPSDMatrix)

valid_spd_matrices = (SPDMatrix,
                      ConstantSPDMatrix)

def is_positive_definite(matrix, epsilon=1e-6):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > epsilon)

def _check_valid_isphs(isphs):
    assert isinstance(isphs, ISPHS), 'The model must be an ISPHS instance'
    assert isinstance(isphs.hamiltonian, LyapunovNN), "The ISPHS' Hamiltonian is not represented with a LyapunovNN instance"
    assert isinstance(isphs.hamiltonian.ficnn, FICNN), "LyapunovNN's convex NN is not of known type"
    assert isinstance(isphs.poisson_matrix, valid_structure_matrices), "(Poisson) structure matrix is not of known type"
    assert isinstance(isphs.resistive_matrix, valid_spd_matrices+valid_spsd_matrices), "Resisitive/Dissipation matrix is not of known type"


def is_zero_gas_guarantee_valid(isphs:eqx.Module, epsilon:float=1e-6):
    
    _check_valid_isphs(isphs)

    # Check the positive definiteness of the Hamiltonian at the minimum
    H_hessian = jax.hessian(isphs.hamiltonian)(isphs.hamiltonian.minimum)
    H_hessian_pd = is_positive_definite(H_hessian, epsilon)

    # Check if the resistive/dissipation matrix is positive definite
    R_globally_pd = False
    if isinstance(isphs.resistive_matrix, SPDMatrix):
        # I assume here that the SPDMatrix implementation is correct 
        R_globally_pd = True
    elif isinstance(isphs.resistive_matrix, (ConstantSPDMatrix, ConstantSPSDMatrix)):
        # If the matrix is constant, I can simply check
        R = isphs.resistive_matrix(None)
        R_globally_pd = is_positive_definite(R, epsilon)

    zero_gas = R_globally_pd and H_hessian_pd
    return zero_gas

def get_eigenvals(isphs):

    _check_valid_isphs(isphs)

    # Get eigenvalues of the Hamiltonian's Hessian at the minimum
    H_hessian = jax.hessian(isphs.hamiltonian)(isphs.hamiltonian.minimum)
    H_hessian_eigvals = np.linalg.eigvals(H_hessian)

    # Check if the resistive/dissipation matrix is positive definite
    R_eigvals = None
    if isinstance(isphs.resistive_matrix, (ConstantSPDMatrix, ConstantSPSDMatrix)):
        # If the matrix is constant, I can simply check
        R = isphs.resistive_matrix(None)
        R_eigvals = np.linalg.eigvals(R)

    return H_hessian_eigvals, R_eigvals
