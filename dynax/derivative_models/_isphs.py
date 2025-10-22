"""
This file implements learnable Input-State port-Hamiltonian Systems
"""


from jaxtyping import Array, Scalar
import equinox as eqx
import jax
from jax import numpy as jnp

class ISPHS(eqx.Module):
    hamiltonian: eqx.Module
    poisson_matrix: eqx.Module
    resistive_matrix: eqx.Module
    input_matrix: eqx.Module

    def __init__(self,
                 hamiltonian: eqx.Module,
                 poisson_matrix: eqx.Module,
                 resistive_matrix: eqx.Module = None,
                 input_matrix: eqx.Module = None):
        
        self.hamiltonian = hamiltonian
        self.poisson_matrix = poisson_matrix
        self.resistive_matrix = resistive_matrix
        self.input_matrix = input_matrix

    def __call__(self, t:Scalar, x:Array, u:Array|None) -> Array:

        S = self.poisson_matrix(x)

        if self.resistive_matrix is not None:
            R = self.resistive_matrix(x)
            S -= R
        
        x_t = S @ jax.grad(self.hamiltonian)(x)

        if u is not None:
            g = self.input_matrix(x)
            x_t += g @ u

        return x_t
    
    def get_conservative_dynamics(self, t:Scalar, x:Array, u:Array|None) -> Array:

        S = self.poisson_matrix(x)
        x_t = S @ jax.grad(self.hamiltonian)(x)

        if u is not None:
            g = self.input_matrix(x)
            x_t += g @ u

        return x_t
    
    def get_dissipative_dynamics(self, t:Scalar, x:Array, u:Array|None) -> Array:

        if self.resistive_matrix is not None:
            R = self.resistive_matrix(x)
        else:
            R = jnp.zeros(2*(x.size,))
        x_t = -R @ jax.grad(self.hamiltonian)(x)

        if u is not None:
            g = self.input_matrix(x)
            x_t += g @ u

        return x_t