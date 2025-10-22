
from pathlib import Path
import optax
import numpy as np
import jax
import jax.numpy as jnp

from dynax import training
from dynax.losses import mse
from .dataloader import DataSet


def get_rmse(model, ts, ys, us):
    ys_pred = jax.vmap(model, in_axes=(None, 0, 0))(ts, ys[:,0], us)
    return jnp.sqrt(jnp.mean(jnp.square(ys_pred - ys)))


def train_model(
        model, 
        train_data: DataSet,
        vali_data: DataSet,
        training_hyperparams: dict,
        weights_dir:Path,
        *, key
    ):

    try:
        model = training.load_weights(weights_dir/Path('weights.eqx'), model)
        history = training.load_history(weights_dir/Path('history.npz'))

    except FileNotFoundError:
        model, history = training.fit_trajectory(
            model, 
            train_data.ts, train_data.ys, train_data.us,
            validation_data = ((vali_data.ts, vali_data.ys[:,0], vali_data.us), vali_data.ys),
            loss_fn         = mse,
            batch_size      = training_hyperparams['batch_size'],
            steps           = training_hyperparams['steps'],
            optimizer       = optax.adam(training_hyperparams['learning_rate']),
            log_loss_every  = 100,
            key             = key
            )
        
        training.save_weights(weights_dir/Path('weights'), model)
        training.save_history(weights_dir/Path('history'), history)

    return model, history


def evaluate_model(
        model, 
        metrics_dir: Path,
        **datasets: DataSet
    ):    

    # Load RMSE data if available, otherwise compute it
    try:
        error_measures = np.load(metrics_dir/Path('error_measures.npz'))
        error_measures = dict(error_measures)
    except FileNotFoundError:
        print('Couldnt find error measures')
        error_measures = dict()

    # Check if the saved metrics contain all requested metrics
    save_necessary = False
    for key, data in datasets.items():
        metric = "rmse_" + key
        if metric not in error_measures:
            error_measures[metric] = get_rmse(model, data.ts, data.ys, data.us)
            save_necessary = True

    if save_necessary:
        np.savez(metrics_dir/Path('error_measures.npz'), **error_measures)

    return error_measures