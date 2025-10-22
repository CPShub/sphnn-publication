
import jax

import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array

## Abstract Base Class ##

class Constraint(eqx.Module):
    def get(self):
        raise NotImplementedError('The constraint class should implement a `get` function.')

## Constraints ##

class NonNegative(Constraint):
    array: Array

    def get(self):
        return jnp.maximum(self.array, 0.)
    
class Symmetric(Constraint):
    array: Array

    def get(self):
        return 0.5 * (self.array + self.array.T)
    
class SkewSymmetric(Constraint):
    array: Array

    def get(self):
        return 0.5 * (self.array - self.array.T)


## Method(s) to apply constraints ##

is_constraint = lambda x: isinstance(x, Constraint)

def _apply_constraint(x):
    if is_constraint(x):
        return x.get()
    else:
        return x
    
def resolve_constraints(model:eqx.Module) -> eqx.Module:
    return jax.tree.map(_apply_constraint, model, is_leaf=is_constraint)