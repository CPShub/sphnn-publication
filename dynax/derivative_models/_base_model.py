"""
This implements the basic form of a derivative model.
This can be used
a) As a reference for implementing other derivative models.
b) To define neural ODEs, by passing a normal MLP as the submodel.
"""

import equinox as eqx

import jax.numpy as jnp

from jaxtyping import Array, Scalar



class BaseModel(eqx.Module):
    submodel: eqx.Module
    state_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    time_dependent: bool = eqx.field(static=True)

    def __init__(self, 
                 submodel:eqx.Module,
                 state_size:int, 
                 input_size:int=0, 
                 time_dependent:bool=False):

        self.submodel = submodel
        self.state_size = state_size
        self.input_size = input_size
        self.time_dependent = time_dependent

    def __call__(self, t:Scalar, y:Array, u:Array|None) -> Array:

        if u is None:
            u = jnp.empty(shape=(0,))

        assert u.shape[0] == self.input_size, f'Expected an input size of {self.input_size} but got {u.shape[0]}.'

        if self.time_dependent:
            t = jnp.expand_dims(t, axis=0)
            nn_input = jnp.concat([t, y, u], axis=0)
        else:
            nn_input = jnp.concat([y, u], axis=0)
            
        return self.submodel(nn_input)