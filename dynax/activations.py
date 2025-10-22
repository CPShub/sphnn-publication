"""
Custom activation functions
"""

import jax
import jax.numpy as jnp
from jaxtyping import Scalar


@jax.jit
def smoothed_relu(x: Scalar, d: float = 0.5) -> Scalar:

    condlist = [x <= 0.,
                jnp.logical_and(0. < x, x < d),
                d <= x]
    funclist = [0., 
                lambda x: 0.5/d*x**2, 
                lambda x: x-0.5*d]
    
    return jnp.piecewise(x, condlist, funclist)
