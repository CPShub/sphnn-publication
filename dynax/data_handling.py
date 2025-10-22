"""
This Module implements data related operations such as normalization
"""

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Scalar, Array

class Normalizer:
    """Shifts and scales the data such that the nomalization
    maps a value of shift to 0 and a value of shift+scale to 1.
    """

    scale: Scalar|Array
    shift: Scalar|Array

    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale

    def normalize(self, data):
        return (data - self.shift) / self.scale
    
    def denormalize(self, data):
        return data * self.scale + self.shift
    

class NormalizationWrapper(eqx.Module):
    model: eqx.Module
    t_normalizer: Normalizer
    y_normalizer: Normalizer
    u_normalizer: Normalizer

    def __init__(self, model: eqx.Module, t_normalizer: Normalizer, y_normalizer: Normalizer, u_normalizer: Normalizer):
        self.model = model
        self.t_normalizer = t_normalizer
        self.y_normalizer = y_normalizer
        self.u_normalizer = u_normalizer

    def __call__(self, ts:Array, y0:Array, us:Array|None) -> Array:
        ts = self.t_normalizer.normalize(ts)
        y0 = self.y_normalizer.normalize(y0)
        us = self.u_normalizer.normalize(us)

        ys_pred = self.model(ts, y0, us)

        return self.y_normalizer.denormalize(ys_pred)

