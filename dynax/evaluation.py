"""
Collection of tools foro the evaluation of multiple instances
"""
from typing import Callable, TypeVar, cast, TypeAlias
import numpy as np

import warnings

import jax 
import jax.numpy as jnp

Array: TypeAlias  = jax.Array | np.ndarray
T = TypeVar('T', bound=Array|None)

def rmse(prediction: Array, target: Array, axis: int|tuple=-1):
    """Root Mean Squared Error along axis.

    Args:
        pred (Array): Array of predictions
        target (Array): Array of targets
        axis (int|tuple), defaults to -1: Axis of the arrays along which to calculate the RMSE.

    Returns:
        RMSEs (Array): Array of RMSEs
    """
    return jnp.sqrt(jnp.mean(jnp.square(prediction - target), axis=axis))

def get_statistics(array: Array):
    """Get statistics such as interquartile mean and many more along the first axis of the array.

    Args:
        array (Array): Input array of data.

    Returns:
        dict: Dictionary of statistics including the oridinal array.
    """
    first_quartile, median, third_quartile = np.quantile(array, [0.25, 0.5, 0.75], axis=0)
    # Compute the interquartile mean
    interquartile_mask = np.logical_and(first_quartile <= array, array <= third_quartile)
    with warnings.catch_warnings(action="ignore"):
        # This ignores the warning from numpy when trying to compute the mean where no ground truth data is available
        interquartile_mean = np.nanmean(np.where(interquartile_mask, array, np.nan), axis=0)
    statistics = dict(
        values = array,
        max = np.max(array, axis=0),
        min = np.min(array, axis=0),
        mean = np.mean(array, axis=0),
        first_quartile = first_quartile, 
        median = median, 
        interquartile_mean = interquartile_mean,
        third_quartile = third_quartile,
        std = np.std(array, axis=0),
    )
    return statistics

def make_batched(array: T) -> T:
    """Adds a third dimension as first dimension if the array is 2D.
    If array is already 3D it is returned unchanged.
    For other dimensions an Value Error is thrown.

    Args:
        array (Array | None): Input Array

    Raises:
        ValueError: If the array is not 2D or 3D.

    Returns:
        Array | None: 3D version of array or None if input is None.
    """
    if array is None:
        return array
    if array.ndim == 2:
        return cast(T, array[None, :, :])
    elif array.ndim == 3:
        return array
    else:
        raise ValueError('Array does not have 2 or 3 dimensions.')

def get_prediction_statistics(ts: Array, ys: Array, us: Array, results:dict, error_metric:Callable=rmse, exclude_model_types=[], model_key='model'):

    # Add batch dimension if only a single trajectory is passed
    ys = make_batched(ys)
    us = make_batched(us)

    true_data = dict(
        ts = ts, ys = ys, us = us
    )
    prediction_data = {}
    for model_type, model_results in results.items():
        if model_type in exclude_model_types:
            continue

        ys_preds = []
        errors = []
        for n, instance_result in enumerate(model_results):
            model = instance_result[model_key]
            try:
                ys_pred = jax.vmap(model, in_axes=(None, 0, 0))(ts, ys[:,0], us)
                ys_preds.append(ys_pred)
                errors.append(error_metric(ys_pred, ys))
            except Exception as e:
                print(f'{model_type}, instance {n} faild to integrate. {e}')
            
        ys_preds = np.stack(ys_preds, axis=0)
        errors = np.stack(errors, axis=0)
        print(f'# of valid {model_type} predictions: {ys_preds.shape[0]}')
        prediction_data[model_type] = dict(
            ys      = get_statistics(ys_preds),
            errors  = get_statistics(errors),
        )
    return prediction_data, true_data