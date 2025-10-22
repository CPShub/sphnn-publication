

from optax import losses
from jax import numpy as jnp


def mse(pred, target, model):
    return jnp.mean((pred - target) ** 2)

def huber(pred, target, model):
    return jnp.mean(losses.huber_loss(pred, target))