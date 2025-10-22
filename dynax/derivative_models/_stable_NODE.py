"""
This file implements stable neural ODEs
from this paper https://proceedings.neurips.cc/paper_files/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf
by Kolter et al.
"""

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.nn as jnn

from jaxtyping import Scalar, Array

class StableNODE(eqx.Module):
    dynamics_func: eqx.Module
    lyapunov_func: eqx.Module
    alpha: float

    def __init__(self,
                 dynamics_func: eqx.Module,
                 lyapunov_func: eqx.Module,
                 alpha: float = 0.
                 ):
        
        self.dynamics_func = dynamics_func
        self.lyapunov_func = lyapunov_func
        self.alpha = alpha

    def __call__(self, t:Scalar|None, x:Array, u:Array|None):

        f_hat = self.dynamics_func(x)
        V, grad_V = jax.value_and_grad(self.lyapunov_func)(x)

        corr_dir = jnn.relu(jnp.inner(grad_V, f_hat) + jax.lax.stop_gradient(self.alpha)*V)
        norm_grad_V = jnp.inner(grad_V, grad_V)

        # Avoid devision by zero in a jit tracable way
        # jnp.array(0.) is vital. Just 0.0 leads to float comparisons that can result in the divisor being zero!
        projection = jnp.where(norm_grad_V==jnp.array(0.), jnp.array(0.), (corr_dir / norm_grad_V) * grad_V) 

        f = f_hat - projection
        return f
