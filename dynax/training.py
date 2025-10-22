"""
This module implements JAX + Equinox + Opatx based training loops.
In particular derivative fitting and trajectory fitting for neural ODE type models.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable, Generator

import equinox as eqx
import diffrax
import optax

import numpy as np

from pathlib import Path
from datetime import timedelta

from .constraints import resolve_constraints
from .losses import mse

def _dataloader(data: PyTree, batch_size:int, *, key:PRNGKeyArray, batch_mask: PyTree=None) -> Generator[PyTree, None, None]:
    """Returns a generator object, that produces a randomly chosen subset of the data (random choice without replacement).
    The data may be an arbitrary tuple-pytree of arrays and none, e.g. `(array_x, (array_y, array_z))`.
    All the arrays should have the same shape along the first axis (the batch axis, `array.shape[0]`).
    The generator will yield a random subset of size batch_size and return it in the same pytree format.
    None values will be unchanged. If a batch mask is passed, then one can specify arrays that don't have a batch dimension.
    For the above mentioned data example, the batch mask `(True, (False, True))` will not slice the array_y along the batch dimension and instead return it unchanged every time.

    Args:
        data (PyTree): A tuple-pytree with data arrays or None values as leafs. All data arrays should generally have the same size along the first dimension, except if a batch mask is supplied.
        batch_size (int): Number of examples in a batch.
        key (PRNGKeyArray): PRNGKey for generating the random sampling.
        batch_mask (PyTree, optional): A tuple-pytree with booleans as leafs and the same structure as the data pytree. The booleans indicate if the corresponding array in the data pytree has a batch dimension. If False, then the corresponding data array will be passed unchanged every time. Defaults to None.

    Returns:
        Generator: Generator that yields a random batch of data every time.

    Yields:
        Pytree: Pytree with the same structure as data.
    """


    # Generate an all true batch mask if None was passed
    if batch_mask is None:
        batch_mask = jax.tree.map(lambda x: x is not None, data)

    # Split the pytree according to the batch mask
    batched_data, unbatched_data = eqx.partition(data, batch_mask)

    # Check that all batch_mask data has the same batch dimension
    arrays_with_batch_dimension = jax.tree.leaves(batched_data)
    if len(arrays_with_batch_dimension) == 0:
        raise ValueError('At least one array should have a batch dimension.')
    dataset_size = arrays_with_batch_dimension[0].shape[0]
    if not all(array.shape[0] == dataset_size for array in arrays_with_batch_dimension[1:]):
        raise ValueError('All arrays with a batch axis should have the same shape along that axis.')

    # Convert to Numpy arrays. Numpy's slicing is much faster than jax's, so for fast model training steps this actually makes a huge difference!
    batched_data = jax.tree.map(lambda x: np.array(x), batched_data) 

    batch_size = min(batch_size, dataset_size)  # Reduce batch size if the dataset has less examples than batch size
    indices = jnp.arange(dataset_size)

    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start, end = 0, batch_size
        while end <= dataset_size:
            batch_indices = perm[start:end]
            bs = jax.tree.map(lambda x: x[batch_indices], batched_data)
            yield eqx.combine(bs, unbatched_data)
            start = end
            end = start + batch_size


def differentiate_trajectory(ts, ys):
    def _diff(ts, ys):
        coeffs = diffrax.backward_hermite_coefficients(ts, ys)
        interp = diffrax.CubicInterpolation(ts, coeffs)
        return jax.vmap(jax.grad(interp.evaluate))(ts)
    return jax.vmap(jax.vmap(_diff, (None, -1), -1), (None, 0))(ts, ys)

def make_derivative_data(ts: Array, ys: Array, us: Array|None=None):
    """
    Calculates an approximate time derivative of trajectory data.

    Args:
        ts (Array): Array of time stamps, shape=(n_time,)
        ys (Array): Array of trajectory data, shape=(n_trajectories, n_time, n_dims)

    Returns:
        (y, y_t): Tuple of training data y -> dy/dt
                    with y.shape=(n_examples, n_dims), y_t.shape=(n_examples, n_dims)
                    and n_examples = n_trajectories * n_time
    """
    y_ts = differentiate_trajectory(ts, ys)
    if us is not None:
        return ys.reshape(-1, ys.shape[-1]), y_ts.reshape(-1, ys.shape[-1]), us.reshape(-1, us.shape[-1])
    else:
        return ys.reshape(-1, ys.shape[-1]), y_ts.reshape(-1, ys.shape[-1])


def concatenate_histories(*histories: dict[Array]):
    keys = histories[0].keys()
    assert all(h.keys() == keys for h in histories)
    history = histories[0]
    for h in histories[1:]:
        for key in keys:
            if key in ['loss', 'val_loss']:
                history[key] = np.concatenate([history[key], h[key]])
            elif key in ['training_time']:
                history[key] += h[key]
            else:
                assert history[key] == h[key], f'The value of key {key} is not the same for all histories.'
    return history

def save_history(file, history: dict[Array], overwrite=False):
    filepath = Path(file)

    if filepath.suffix == '':
        filepath = Path(str(filepath) + '.npz')
    elif filepath.suffix != '.npz':
        raise ValueError('Please don\'t specify a suffix.')
    
    if filepath.exists() and not overwrite:
        raise FileExistsError('File already exists. To overwrite use ```overwrite=True```')
    
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Make sure the folder exists
    jnp.savez(filepath, **history)

def load_history(file):
    return dict(jnp.load(file))


def fit(model: eqx.Module,
        x: Array|tuple[Array, ...],
        y: Array|tuple[Array, ...],
        *,
        validation_data: tuple[Array|tuple[Array, ...], Array|tuple[Array, ...]]|None = None,
        batch_size: int = 32,
        batch_mask: PyTree[bool]|None = None,
        steps: int = 1000,
        log_loss_every: int = 100,
        loss_fn: Callable = mse,
        optimizer: optax.GradientTransformation = optax.adabelief(1e-3),
        callback: Callable|None = None,
        key: PRNGKeyArray,
        ) -> tuple[eqx.Module, dict]:

    """Trains a model using an optimizer from optax.

    Args:
        model (eqx.Module): The model instance which should be trained. It may contain instances of Constraint classes.
        x (Array|tuple[Array, ...]): Input data. It could be a jax.numpy.array or a pytree of jax.numpy.array instances.
        y (Array|tuple[Array, ...]): Target data. It could be a jax.numpy.array or a pytree of jax.numpy.array instances.
        validation_data (tuple, optional): A tuple of inputs and targets used for validation during training. Defaults to None.
        batch_size (int, optional): Number of samples per gradient update. Defaults to 32.
        batch_mask (PyTree, optional): A tuple-pytree with booleans as leafs and the same structure as the pytree ```(x, y)```. The booleans indicate if the corresponding array in the data pytree has a batch dimension. If False, then the corresponding data array will be passed unchanged every time. None indicates that all arrays have a batch dimension. Defaults to None.
        steps (int, optional): Number of gradient updates to apply. Defaults to 1000.
        log_loss_every (int, optional): Idicates how many steps need to be taken in order to conduct a new loss evaluation. A loss evaluation consists of calculating the trainin and validation losses *over the etire datasets* and storing them in the history dictionary. Defaults to 10.
        loss_fn (Callable): A function with call signature ```(prediction, target, model) -> float``` that computes the loss. Defaults to mse.
        optimizer (optax.GradientTransformation): Any optax gradient transform to calculate the updates for the model. Defaults to optax.adabelief(1e-3).
        callback (Callable|None): A function that is called after every gradient update. The call signature is ```(model, step) -> None```.
        key (PRNGKeyArray): A PRNGKey to randomize the individual batches.
        
    Returns:
        model, history (tuple[eqx.Module, dict]): Returns a tuple of the trained model and a history dictionary containing the loss history.
    """

    # Determine the batch dimension for each leaf in the input pytree according to the batch mask
    if batch_mask is None:
        model_in_axes = 0
    else:
        model_in_axes = (jax.tree.map(lambda x: 0 if x else None, batch_mask[0]),)

    # Define a function to calculate the loss. This is jit compiled to speed up the loss evaluation for the loss history.
    @eqx.filter_jit
    def get_loss(model, x, y):
        model = resolve_constraints(model)
        y_pred = jax.vmap(model, in_axes=model_in_axes)(x)
        return loss_fn(y_pred, y, model)
    
    grad_loss = eqx.filter_grad(get_loss) # Get the gradient function

    @eqx.filter_jit
    def make_step(x, y, flat_model, optimizer, flat_opt_state):
        # Use the unflatten trick to speed up training, see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

        # Compute and apply the parameter updates
        grads = grad_loss(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)

        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state = jax.tree_util.tree_leaves(opt_state)

        return flat_model, flat_opt_state

    # Initialize the history dict
    history = {'log_loss_every': log_loss_every,
               'loss': [],}
    if validation_data is not None:
        vx, vy = validation_data
        history['val_loss'] = []

    val_loss = None

    # Initialize the optimizer and 'tell it' to optimize with respect to all inexact arrays in the model
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Use the unflatten trick to speed up training, see https://docs.kidger.site/equinox/tricks/
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

    # Loop over all training steps
    start_time = time.time()
    for step, (xi, yi) in zip(range(1, steps+1), 
                              _dataloader((x, y), batch_size, batch_mask=batch_mask, key=key)):
        flat_model, flat_opt_state = make_step(xi, yi, flat_model, optimizer, flat_opt_state)   # Make the step

        if callback is not None:
            model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            callback(model, step)

        # Log the losses
        if (step % log_loss_every) == 0 or step == steps - 1:
            model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            train_loss = get_loss(model, x, y)
            history['loss'].append(train_loss)
            if validation_data is not None:
                val_loss = get_loss(model, vx, vy)
                history['val_loss'].append(val_loss)
                print(f"Step: {step}, Loss: {train_loss:.3e}, Validation loss: {val_loss:.3e}")
            else:
                print(f"Step: {step}, Loss: {train_loss:.3e}")

    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

    training_time = time.time() - start_time
    print(f'Training took: {timedelta(seconds=training_time)}')

    history['training_time'] = training_time
    history = {k: np.array(v) for k,v in history.items()}

    return model, history


## Special callbacks

def save_every(n, filename, overwrite: bool=False):
    filepath = Path(filename)
    folderpath = filepath.parents[0]
    filestem = filepath.stem    # Filename without (optional) extension

    if filepath.suffix not in ['', '.eqx']:
        raise ValueError('Please don\'t specify a suffix.')
    
    filename = folderpath / filepath.stem   # remove suffix

    if folderpath.exists():    # Check if the parent folder exists
        # Check if we are in danger of overwriting!
        potential_confilicts = list(folderpath.glob(rf'{filestem}_*.eqx'))
        if len(potential_confilicts) != 0 and not overwrite:
            raise FileExistsError('There are files at the specified location which could be overwritten. To ignore this error or to overwrite files set ```overwrite=True```.')
    else:
        # Create the folder
        folderpath.mkdir(parents=True, exist_ok=True)

    def save_callback(model, step):
        if step % n == 0:
            file = folderpath / f'{filestem}_{step}'
            save_weights(file, model)
    return save_callback


## Load and save weights

def save_weights(file, model, overwrite=False):
    filepath = Path(file)
    folderpath = filepath.parents[0]
    if filepath.exists() and not overwrite:
        raise FileExistsError('File already exists. To overwrite set ```overwrite=True```')
    if not folderpath.exists():    # Check if the parent folder exists
        folderpath.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(file, model)

def load_weights(file, model):
    return eqx.tree_deserialise_leaves(file, model)


## Special derived training loops for derivative and trajectory fitting

class DerivativeWrapper(eqx.Module):
    submodel: eqx.Module

    def __init__(self, submodel):
        self.submodel = submodel

    def __call__(self, x):
        t, y, u = x
        return self.submodel(t, y, u)
    
class TrajectoryWrapper(eqx.Module):
    submodel: eqx.Module

    def __init__(self, submodel):
        self.submodel = submodel

    def __call__(self, x):
        t, y0, u = x
        return self.submodel(t, y0, u)


def fit_derivative(model: PyTree,
                   ys: Array,
                   y_ts: Array,
                   ts: Array = None,
                   us: Array = None,
                   *,
                   key: PRNGKeyArray,
                   **kwargs):

    model_ = DerivativeWrapper(model)
    model_, history = fit(model_, (ts, ys, us), y_ts, 
                          key=key, **kwargs)
    model = model_.submodel

    return model, history

def fit_trajectory(model: PyTree,
                   ts: Array,
                   ys: Array,
                   us: Array = None,
                   *,
                   key: PRNGKeyArray,
                   **kwargs):

    model_ = TrajectoryWrapper(model)
    model_, history = fit(model_, (ts, ys[:,0], us), ys, 
                        key=key, batch_mask=((False, True, True), True), **kwargs)
    model = model_.submodel

    return model, history

