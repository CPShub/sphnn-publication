"""
This file implements a neural network models for differnet matrix valued functions.
"""

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

import numpy as np

from jaxtyping import Array, PRNGKeyArray
from typing import Union, Literal, Callable

from ..constraints import SkewSymmetric

#### Arbitrary Matrix Function ####

class ConstantMatrix(eqx.Module):
    A: Array

    def __init__(self, 
                 shape: tuple[int,...], 
                 initialize: jnn.initializers.Initializer|Literal['zero', 'random'] = 'random',
                 *, 
                 key: PRNGKeyArray):
        if initialize == 'random':
            initializer = jnn.initializers.glorot_uniform()
            # self.A = initializer(key, shape)
        elif initialize == 'zero':
            # self.A = jnp.zeros(shape)
            initializer = jnn.initializers.zeros
        elif isinstance(initialize, jnn.initializers.Initializer):
            initializer = initialize
        else:
            raise ValueError(f'Ivalid literal "{initialize}" for argument initialize.')

        self.A = initializer(key, shape)

    def __call__(self, x: Array) -> Array:
        return self.A

class MatrixFunction(eqx.Module):
    mlp: eqx.nn.MLP
    shape: int

    def __init__(self,
                 in_size: int,
                 shape: tuple[int,...] = None,
                 width: int = 16,
                 depth: int = 2,
                 activation: Callable=jnn.sparse_plus,
                 *,
                 key: PRNGKeyArray):
        
        shape = (in_size, in_size) if shape is None else shape
        self.shape = shape
        k = np.prod(shape)

        self.mlp = eqx.nn.MLP(in_size, k, 
                              width_size=width, depth=depth, 
                              activation=activation,
                              key=key)
        
    def __call__(self, x: Array) -> Array:
        a = self.mlp(x)
        return jnp.reshape(a, self.shape)
    
    
#### Skew Symmetry ####

class SkewSymmetricMatrix(eqx.Module):
    """
    *Skew Symmetric Matrix*
    This implements a matrix-valued function f: R^m -> R^(nxn), 
    where the output is a skew-symmetric matrix. The parameterization
    is done by a neural network f: R^m -> R^k with k=n(n-1)/2. It works by 
    initially constructing a R^(nxnxk) tensor, that maps from the k-dimensional
    component vector to the space of skew-symmetric matrices. When called the
    input is fed to the NN and its ouput is multiplied by the tensor. 
    """

    mlp: eqx.nn.MLP
    tensor: Array  # Not learnable transform tensor from the vector-space of components to the space of skew-symmetric matrices

    def __init__(self,
                 in_size: int,
                 matrix_size: int = None,
                 width: int = 16,
                 depth: int = 2,
                 activation: Callable=jnn.sparse_plus,
                 *,
                 key: PRNGKeyArray):
        
        matrix_size = in_size if matrix_size is None else matrix_size

        # Construct the tensor
        n = matrix_size
        k = n*(n-1)//2
        l = []
        for i in range(n-1):
            for j in range(i+1, n):
                s = np.zeros((n, n), dtype='int32')
                s[i, j] = 1
                s[j, i] = -1
                l.append(s)
        self.tensor = jnp.stack(l, axis=-1)

        self.mlp = eqx.nn.MLP(in_size, k, 
                              width_size=width, depth=depth, 
                              activation=activation,
                              key=key)
        
    def __call__(self, x: Array) -> Array:
        a = self.mlp(x)
        return jax.lax.stop_gradient(self.tensor) @ a

class ConstantSkewSymmetricMatrix(eqx.Module):
    """
    *Constant Skew Symmetric Matrix*
    This implements a constant, learnable skew-symmetric matrix
    by applying the skew symmetry constraint on an array.
    """

    A: Array    # Holds the skew-symmetric matrix directly

    def __init__(self, 
                 size: int, 
                 initialize: jnn.initializers.Initializer|Literal['symplectic', 'random'] = 'random',
                 *, 
                 key: PRNGKeyArray):
        
        if isinstance(initialize, jnn.initializers.Initializer):
            A = initialize(key, (size, size))
        elif initialize == 'random':
            initializer = jnn.initializers.glorot_uniform() # TODO: UNIFY intilizer
            A = initializer(key, (size, size))
        elif initialize == 'symplectic':
            assert size%2 == 0, 'For symplectic initialization the matrix size must be an even integer.'
            n_dof = size // 2
            I, O = jnp.eye(n_dof), jnp.zeros((n_dof, n_dof))
            omega = jnp.block([[ O, I], 
                               [-I, O]])
            A = 0.5*omega
        else:
            raise ValueError(f'Ivalid literal "{initialize}" for argument initialize.')

        self.A = SkewSymmetric(A-A.T)   # Make sure A is skew-symmetric initially

    def __call__(self, x: Array) -> Array:
        return self.A

class SymplecticMatrix(eqx.Module):

    J: Array

    def __init__(self, size: int):
        assert size%2==0, 'The size must be an even integer!'
        n_dof = size//2
        I, O = jnp.eye(n_dof, dtype='int'), jnp.zeros((n_dof, n_dof), dtype='int')
        self.J = jnp.block([[ O, I], 
                            [-I, O]])

    def __call__(self, x: Array) -> Array:
        return self.J


#### Symmetric Positive Semi-Definite ####

class SPSDMatrix(eqx.Module):
    """
    *Symmetric Positive Semi-Definite Matrix*
    This implements a matrix-valued function f: R^m -> R^(nxn), 
    where the output is a symmetric positive semi-definite matrix.
    The parameterization is done by a neural network f: R^m -> R^k 
    with k=n(n+1)/2. It works by initially constructing a {0,1}^(n x n x k) tensor, 
    that maps from the k-dimensional component vector to the space 
    of lower-triangular matrices. When called the input is fed to the 
    NN and its ouput is multiplied by the tensor. Then the Cholesky 
    decomposition is used to construct the symmetric positive 
    semi-definite matrix.
    """

    mlp: eqx.nn.MLP
    tensor: Array  # Not learnable transform tensor from the vector-space of components to the space of lower triangular matrices
 
    def __init__(self,
                 in_size: int,
                 matrix_size: int = None,
                 width: int = 16,
                 depth: int = 2,
                 activation: Callable=jnn.sparse_plus,
                 *,
                 key: PRNGKeyArray):
        
        matrix_size = in_size if matrix_size is None else matrix_size

        # Construct the tensor
        n = matrix_size
        k = n*(n+1)//2
        l = []
        for i in range(n):
            for j in range(i+1):
                s = np.zeros((n, n), dtype='int32')
                s[i, j] = 1
                l.append(s)
        self.tensor = jnp.stack(l, axis=-1)

        self.mlp = eqx.nn.MLP(in_size, k, 
                              width_size=width, depth=depth, 
                              activation=activation,
                              key=key)
        
    def __call__(self, x: Array) -> Array:
        a = self.mlp(x)
        L = jax.lax.stop_gradient(self.tensor) @ a
        # To ensure nonegative diagonals, take the absolute value of the diagonal elements
        L_lower = jnp.tril(L, -1)
        L_diasq  = jnp.diag(jnp.abs(jnp.diag(L)))
        L = L_lower + L_diasq
        return L @ L.T

class ConstantSPSDMatrix(eqx.Module):
    """
    *Constant symmetric positive semi-definite matrix*
    """
    L: Array

    def __init__(self, 
                 size: int, 
                 initializer: jnn.initializers.Initializer = jnn.initializers.glorot_uniform(), 
                 *, key:PRNGKeyArray):
        
        self.L = initializer(key, (size, size))

    def __call__(self, x: Array) -> Array:
        # To ensure nonegative diagonals, square the diagonal elements
        L_lower = jnp.tril(self.L, -1)
        L_diasq  = jnp.diag(jnp.abs(jnp.diag(self.L)))
        L = L_lower + L_diasq
        return L @ L.T


#### Symmetric Positive Definite ####

class SPDMatrix(eqx.Module):
    """
    *Symmetric Positive DEFINITE Matrix*
    This implements a matrix-valued function f: R^m -> R^(nxn), 
    where the output is a symmetric positive definite matrix.
    The parameterization is done by a neural network f: R^m -> R^k 
    with k=n(n+1)/2. It works by initially constructing a {0,1}^(n x n x k) tensor, 
    that maps from the k-dimensional component vector to the space 
    of lower-triangular matrices. When called the input is fed to the 
    NN and its ouput is multiplied by the tensor. Then the Cholesky 
    decomposition is used to construct the symmetric positive 
    semi-definite matrix. The diagonal elements are passed throught softplus
    to ensure positivity.
    """

    mlp: eqx.nn.MLP
    tensor: Array  # Not learnable transform tensor from the vector-space of components to the space of lower triangular matrices
 
    def __init__(self,
                 in_size: int,
                 matrix_size: int = None,
                 width: int = 16,
                 depth: int = 2,
                 activation: Callable=jnn.sparse_plus,
                 *,
                 key: PRNGKeyArray):
        
        matrix_size = in_size if matrix_size is None else matrix_size

        # Construct the tensor
        n = matrix_size
        k = n*(n+1)//2
        l = []
        for i in range(n):
            for j in range(i+1):
                s = np.zeros((n, n), dtype='int32')
                s[i, j] = 1
                l.append(s)
        self.tensor = jnp.stack(l, axis=-1)

        self.mlp = eqx.nn.MLP(in_size, k, 
                              width_size=width, depth=depth, 
                              activation=activation,
                              key=key)
        
    def __call__(self, x: Array) -> Array:
        a = self.mlp(x)
        L = jax.lax.stop_gradient(self.tensor) @ a
        # To ensure positive diagonals, pass diagonal through softplus
        L_lower = jnp.tril(L, -1)
        L_diag  = jnp.diag(jnn.softplus(jnp.diag(L)))
        L = L_lower + L_diag
        return L @ L.T

class ConstantSPDMatrix(eqx.Module):
    L: Array

    def __init__(self, 
                 size: int, 
                 initializer: jnn.initializers.Initializer = jnn.initializers.glorot_uniform(), 
                 *, key:PRNGKeyArray):
        
        self.L = initializer(key, (size, size))

    def __call__(self, x: Array) -> Array:
        # To ensure positive diagonals, pass diagonal through softplus
        L_lower = jnp.tril(self.L, -1)
        L_diag  = jnp.diag(jnn.softplus(jnp.diag(self.L)))
        L = L_lower + L_diag
        return L @ L.T

