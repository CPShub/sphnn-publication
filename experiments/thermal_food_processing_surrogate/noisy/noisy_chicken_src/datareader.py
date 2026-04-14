import numpy as np
from pathlib import Path
from typing import Literal

def make_tuple(x):
    return (x,) if isinstance(x, int) else tuple(x)

def load_chicken(trajectory_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a subset of the chicken data.

    Args:
        trajectory_ids (list[int]): Identifier of the trajectories.
        
    Returns:
        tuple[np.array, np.array, np.array]: Arrays of the timestamps, measured and oven temperature.
            ts - shape=(num_time,)
            ys - shape=(num_trajectories, num_time, state_size)
            us - shape=(num_trajectories, num_time)
    """

    # Load only from the data.npz file
    train_ids = [745, 795]
    test_ids  = [313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528] # Test group AP15

    data = np.load('../../../data/thermal_food_processing_surrogate/data.npz')
    if trajectory_ids == train_ids:
        return data['ts_train'], data['ys_train'][:2], data['us_train'][:2]
    elif trajectory_ids == test_ids:
        return data['ts_vali'],  data['ys_vali'],  data['us_vali'] 
    else:
        raise ValueError(f"Can only load trajectory id-lists {train_ids} or {test_ids}")
