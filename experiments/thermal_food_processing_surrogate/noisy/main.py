"""This file runs experiments for the noisy thermal food processing surrogate problem."""

import jax.random as jr

import json
from pathlib import Path
from multiprocessing import Pool

from dynax.training import resolve_constraints
from dynax.data_handling import NormalizationWrapper

from noisy_chicken_src import dataloader
from noisy_chicken_src.models import get_model
from noisy_chicken_src.training import train_model, evaluate_model


def setup_data():
    # Load the data only once for each worker
    global data, t_normalizer, y_normalizer, u_normalizer
    data, t_normalizer, y_normalizer, u_normalizer = dataloader.prepare_data()


def run(params: dict, return_results=False):
    # Unpack the params
    model_type = params["model_type"]
    num_aug = params["num_aug"]
    num_train_trajectories = params["num_train_trajectories"]
    noise_amplitude = params["noise_amplitude"]
    instance_id = params["instance_id"]
    directory = params["directory"]

    print(
        f"{model_type:<15}, num_aug={num_aug:>2}, num_dat={num_train_trajectories:>2}, noise_amplitude={noise_amplitude:>4}, instance: {instance_id:>2}"
    )

    key = jr.key(instance_id)
    model_key, loader_key = jr.split(key, 2)

    # Get the model
    _model = get_model(model_type, params["config"]["model"]["hyperparams"], model_key)

    # Apply the noise to the training data
    # Static key -> noise will be the same for all instances
    data["train_norm_noisy"] = data["train_norm"].add_noise(
        noise_amplitude, key=jr.key(0)
    )
    data["train_noisy"] = data["train_norm_noisy"].denormalize(
        t_normalizer, y_normalizer, u_normalizer
    )

    # Train the model
    _model, history = train_model(
        _model,
        data["train_norm_noisy"],
        data["test_norm"],
        training_hyperparams=params["config"]["training"],
        weights_dir=directory,
        key=loader_key,
    )

    model = resolve_constraints(_model)  # type: ignore
    wrapped_model = NormalizationWrapper(
        model, t_normalizer, y_normalizer, u_normalizer
    )

    error_measures = evaluate_model(
        wrapped_model,
        metrics_dir=directory,
        train=data["train"],
        train_noisy=data["train_noisy"],
        test=data["test"],
        delayed_test=data["test_delayed"],
        long_data=data["long"],
    )

    # Create results dictionary
    run_record = dict(
        **params,
        wrapped_model = wrapped_model,
        history=history,
        **error_measures,
    )

    if return_results:
        return run_record


def get_results(experiment_dir: Path, parallel=False):
    with open(experiment_dir / "hyperparameters.json", "r") as f:
        runparams = json.load(f)

    # Construct the param_list
    param_list = []
    model_types = set(runparams.keys()) - {"meta", "training"}
    for model_type in model_types:
        for num_train_trajectories in runparams["meta"]["nums_train_trajectories"]:
            for num_aug in runparams["meta"]["augmentations"]:
                for noise_amplitude in runparams["meta"]["noise_amplitude"]:
                    for instance_id in range(runparams["meta"]["num_instances"]):
                        directory = (
                            experiment_dir
                            / model_type
                            / f"augment_{num_aug}"
                            / f"dat_{num_train_trajectories}"
                            / f"noise_{noise_amplitude}"
                            / f"instance_{instance_id}"
                        )

                        params = dict(
                            model_type=model_type,
                            num_aug=num_aug,
                            num_train_trajectories=num_train_trajectories,
                            noise_amplitude=noise_amplitude,
                            instance_id=instance_id,
                            directory=directory,
                            config=dict(
                                model=dict(
                                    type=model_type,
                                    hyperparams=runparams[model_type]
                                    | {"num_aug": num_aug},
                                    instance_id=instance_id,
                                ),
                                training=runparams["training"],
                                data=dict(
                                    num_train_trajectories=num_train_trajectories,
                                    noise_amplitude=noise_amplitude,
                                ),
                                directory=directory,
                            ),
                        )

                        param_list.append(params)

    if parallel:
        with Pool(15, initializer=setup_data) as pool:
            pool.map(run, param_list)
    else:
        setup_data()
        results = []
        for params in param_list:
            results.append(run(params, return_results=True))

        return results


if __name__ == "__main__":
    SAVE_DIR = Path(R"icml_experiments\noisy_chicken_data\results\run_0")
    get_results(SAVE_DIR, parallel=True)
