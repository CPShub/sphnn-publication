# import sys

# sys.path.append("C:/Users/roth/Documents/Git Repositories/PANNs_dynamic")

import jax
import jax.numpy as jnp
from jax import random as jr
from jax import nn as jnn
import equinox as eqx

from dynax.function_models import (
    LyapunovNN,
    OnsagerNetPotential,
    ConstantSkewSymmetricMatrix,
    ConstantMatrix,
    ConstantSPDMatrix,
    MLP,
    FICNN,
)
from dynax.integration_models import ODESolver
from dynax.derivative_models import BaseModel, ISPHS


ACTIVATIONS = dict(
    softplus=jax.nn.softplus,
    tanh=jax.nn.tanh,
    relu=jax.nn.relu,
)

INITIALIZERS = dict(
    he_uniform=jax.nn.initializers.he_uniform(),
    glorot_uniform=jax.nn.initializers.glorot_uniform(),
    zeros=jax.nn.initializers.zeros,
)


def define_sPHNN_model(hyperparams: dict, *, key):
    j_key, r_key, g_key, h_key = jr.split(key, 4)
    num_aug = hyperparams["num_aug"]

    state_size, input_size = 2 + num_aug, 1

    J = ConstantSkewSymmetricMatrix(state_size, key=j_key)
    R = ConstantSPDMatrix(state_size, initializer=jnn.initializers.zeros, key=r_key)
    g = ConstantMatrix(
        (state_size, input_size), initialize=jnn.initializers.zeros, key=g_key
    )

    ficnn = FICNN(
        state_size,
        "scalar",
        width=hyperparams["ficnn_width"],
        depth=hyperparams["ficnn_depth"],
        activation=jnn.softplus,
        w_initializer=INITIALIZERS[hyperparams["weight_initialization"]],
        b_initializer=INITIALIZERS[hyperparams["bias_initialization"]],
        key=h_key,
    )
    H = LyapunovNN(ficnn, minimum=jnp.zeros(state_size))
    sphnn_ = ISPHS(H, J, R, g)
    sphnn_ode = ODESolver(sphnn_, augmentation=num_aug, augmentation_learnable=False)

    return sphnn_ode


def define_sPHNN_LM_model(hyperparams: dict, *, key):
    j_key, r_key, g_key, h_key, minimum_key = jr.split(key, 5)
    num_aug = hyperparams["num_aug"]

    state_size, input_size = 2 + num_aug, 1

    J = ConstantSkewSymmetricMatrix(state_size, key=j_key)
    R = ConstantSPDMatrix(state_size, initializer=jnn.initializers.zeros, key=r_key)
    g = ConstantMatrix(
        (state_size, input_size), initialize=jnn.initializers.zeros, key=g_key
    )

    ficnn = FICNN(
        state_size,
        "scalar",
        width=hyperparams["ficnn_width"],
        depth=hyperparams["ficnn_depth"],
        activation=jnn.softplus,
        w_initializer=INITIALIZERS[hyperparams["weight_initialization"]],
        b_initializer=INITIALIZERS[hyperparams["bias_initialization"]],
        key=h_key,
    )
    _H = LyapunovNN(ficnn)
    initial_minimum = jax.nn.initializers.normal(
        hyperparams["minimum_initializer_std"]
    )(minimum_key, _H.minimum.shape)
    H = eqx.tree_at(lambda x: x.minimum, _H, initial_minimum)
    assert H.minimum_learnable, "Minimum is not learnable for sPHNN_LM"
    sphnn_ = ISPHS(H, J, R, g)
    sphnn_ode = ODESolver(sphnn_, augmentation=num_aug, augmentation_learnable=False)

    return sphnn_ode


def define_PHNN_model(hyperparams: dict, *, key):
    j_key, r_key, g_key, h_key = jr.split(key, 4)
    num_aug = hyperparams["num_aug"]

    state_size, input_size = 2 + num_aug, 1

    J = ConstantSkewSymmetricMatrix(state_size, key=j_key)
    R = ConstantSPDMatrix(state_size, initializer=jnn.initializers.zeros, key=r_key)
    g = ConstantMatrix(
        (state_size, input_size), initialize=jnn.initializers.zeros, key=g_key
    )

    H = MLP(
        in_size=state_size,
        out_size="scalar",
        width_size=hyperparams["mlp_width"],
        depth=hyperparams["mlp_depth"],
        weight_initializer=INITIALIZERS[hyperparams["weight_initialization"]],
        bias_initializer=INITIALIZERS[hyperparams["bias_initialization"]],
        activation=ACTIVATIONS[hyperparams["activation"]],
        key=h_key,
    )
    phnn_ = ISPHS(H, J, R, g)
    phnn_ode_ = ODESolver(phnn_, augmentation=num_aug, augmentation_learnable=True)

    return phnn_ode_


def define_NODE_model(hyperparams: dict, *, key):
    num_aug = hyperparams["num_aug"]
    state_size, input_size = 2 + num_aug, 1

    mlp = MLP(
        state_size + input_size,
        state_size,
        width_size=hyperparams["mlp_width"],
        depth=hyperparams["mlp_depth"],
        weight_initializer=INITIALIZERS[hyperparams["weight_initialization"]],
        bias_initializer=INITIALIZERS[hyperparams["bias_initialization"]],
        activation=ACTIVATIONS[hyperparams["activation"]],
        key=key,
    )
    div_model = BaseModel(mlp, state_size, input_size)
    node = ODESolver(
        div_model,
        augmentation=num_aug,
        augmentation_learnable=True,
        max_steps=8192,  # Hopefully all nodes can be integrated by increasing the max steps.
    )

    return node


def define_cPHNN_model(hyperparams: dict, *, key):
    j_key, r_key, g_key, h_key = jr.split(key, 4)

    num_aug = hyperparams["num_aug"]
    state_size, input_size = 2 + num_aug, 1

    J = ConstantSkewSymmetricMatrix(state_size, key=j_key)
    R = ConstantSPDMatrix(state_size, initializer=jnn.initializers.zeros, key=r_key)
    g = ConstantMatrix(
        (state_size, input_size), initialize=jnn.initializers.zeros, key=g_key
    )

    H = OnsagerNetPotential(
        state_size,
        width_size=hyperparams["mlp_width"],
        depth=hyperparams["mlp_depth"],
        activation=ACTIVATIONS[hyperparams["activation"]],
        weight_initializer=INITIALIZERS[hyperparams["weight_initialization"]],
        bias_initializer=INITIALIZERS[hyperparams["bias_initialization"]],
        beta=hyperparams["beta"],
        beta_learnable=hyperparams["beta_learnable"],
        key=h_key,
    )

    cphnn_ = ISPHS(H, J, R, g)
    cphnn_ode_ = ODESolver(cphnn_, augmentation=num_aug, augmentation_learnable=True)

    return cphnn_ode_


def get_model(model_type, hyperparameters, key):
    match model_type:
        case "sPHNN":
            return define_sPHNN_model(hyperparameters, key=key)
        case "sPHNN-LM":
            return define_sPHNN_LM_model(hyperparameters, key=key)
        case "PHNN":
            return define_PHNN_model(hyperparameters, key=key)
        case "NODE":
            return define_NODE_model(hyperparameters, key=key)
        case "cPHNN":
            return define_cPHNN_model(hyperparameters, key=key)
        case _:
            raise ValueError(f"Model type {model_type} not recognized.")
