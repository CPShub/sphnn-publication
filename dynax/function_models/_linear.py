"""
This file implements a linear transformation layer. 
This implementations adresses the limitaitons of the equinox implementation:
- Custom parameter initialization

The code is an adapted version of eqx.nn.Linear from equinox.
"""

import equinox as eqx

import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import he_normal, zeros

from jaxtyping import Array, PRNGKeyArray
from typing import (
    Literal,
    Union,
)

class Linear(eqx.Module):
    """
    Custom implementation of a linear layer
    """
    weight: Array
    bias: Array
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        weight_initializer = he_normal(),
        bias_initializer = zeros,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        
        wkey, bkey = jr.split(key, 2)
        in_size_ = 1 if in_size == "scalar" else in_size
        out_size_ = 1 if out_size == "scalar" else out_size

        self.weight = weight_initializer(wkey, (out_size_, in_size_), dtype=dtype)
        if use_bias:
            self.bias = bias_initializer(bkey, (out_size_,), dtype=dtype)
        else:
            self.bias = None

        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = use_bias

    def __call__(self, x: Array) -> Array:

        if self.in_size == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_size == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x
    
