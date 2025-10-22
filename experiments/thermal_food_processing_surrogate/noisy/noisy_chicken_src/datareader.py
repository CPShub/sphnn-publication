import numpy as np
from pathlib import Path
from typing import Literal

def make_tuple(x):
    return (x,) if isinstance(x, int) else tuple(x)

def load_chicken(trajectory_ids: int|tuple[int]|list[int]|Literal['all'],
                 make_same_shape: bool=True,
                 data_dir: Path=Path(R'C:\Users\roth\Documents\Git Repositories\PANNs_dynamic\data\chicken_data\train-s31-onlyTaTb'),
                 ignore_errors: bool=True,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a subset of the chicken data.

    Args:
        trajectory_ids (int | tuple[int] | Literal["all"]): Identifier of the trajectories.
        make_same_shape (bool, optional): If true, removes the last data point of trajectories with lenth 281.
        data_dir (Path): Path to the folder containing the data.
        ignore_errors (bool, optional): If the search for an identifier does not yield a unique result (no or more than one) then ignore that identifier.
        
    Returns:
        tuple[np.array, np.array, np.array]: Arrays of the timestamps, measured and oven temperature.
            ts - shape=(num_time,)
            ys - shape=(num_trajectories, num_time, state_size)
            us - shape=(num_trajectories, num_time)
    """

    trajectory_ids = range(272, 1087) if trajectory_ids=='all' else trajectory_ids

    trajectory_ids = make_tuple(trajectory_ids)

    trajectory_ids = tuple(str(x) for x in trajectory_ids)

    folders = []
    for id in trajectory_ids:
        folder = list(data_dir.glob(fr'{id}*'))
        if len(folder) == 0:
            if ignore_errors:
                print(f'No file for trajectory id {id} found.')
            else:
                raise FileNotFoundError(f'No file for trajectory id {id} found.')
        elif len(folder) > 1:
            if ignore_errors:
                print(f'Multiple files for trajectory id {id} found.')
            else:
                raise FileNotFoundError(f'Multiple for trajectory id {id} found.')
        folders += folder

    if len(folders) != len(trajectory_ids):
        print('Warning: Number of requested trajectories will not match the number of returned trajectories')

    trajectories = []
    excitations = []
    ts = None

    def set_ts(x):
        nonlocal ts
        if ts is None:
            ts = x
        else:
            assert np.array_equal(x, ts), 'timestamps do not match'

    for folder in folders:
        files = [x for x in folder.glob('*')]

        # Load the excitation timeseries
        exc_paths = [x for x in files if str(x.name).endswith('_exc.csv')]
        assert len(exc_paths) == 1, 'There are no or multiple exc files.'
        exc = np.loadtxt(exc_paths[0], skiprows=1, delimiter=',')

        time_exc, excitation = exc.T
        excitation = np.expand_dims(excitation, axis=-1)  # Add a excitation vector axis (Even though the excitation is a scalar, this will make it simpler for trianing.)

        # Load the output timeseries
        out_paths = [x for x in files if str(x.name).endswith('_out.csv')]
        assert len(out_paths) == 1, 'There are no or multiple out files.'
        out = np.loadtxt(out_paths[0], skiprows=1, delimiter=',')

        time_out, temperatures = np.split(out, [1], axis=1)
        time_out = np.squeeze(time_out)

        assert np.array_equal(time_exc, time_out), f'Timestamps from excitations and outputs of trajectory {folder} do not match.'

        if time_exc.shape == (281,) and make_same_shape:
            time_exc, temperatures, excitation = time_exc[:-1], temperatures[:-1], excitation[:-1]

        set_ts(time_exc)
        trajectories.append(temperatures)
        excitations.append(excitation)


    ys = np.stack(trajectories, axis=0)
    us = np.stack(excitations, axis=0)
    return ts, ys, us


def load_long_chicken(data_dir: Path=Path(R'C:\Users\roth\Documents\Git Repositories\PANNs_dynamic\data\chicken_data\train-s31-onlyTaTb')):
    """There is a single trajectory in the data set that is significantly
    longer than the others. This function loads that trajectory.

    Args:
        data_dir: Directory at which the data is stored.

    Returns:
        time, state, exciation.
    """
    long_dir = data_dir/'303-long'

    exc_paths = [f for f in long_dir.glob('*_exc.csv')]
    assert len(exc_paths) == 1, 'There are no or multiple exc files.'
    exc = np.loadtxt(exc_paths[0], skiprows=1, delimiter=',')

    time_exc, exc = exc.T
    exc = np.expand_dims(exc, axis=-1)

    out_paths = [f for f in long_dir.glob('*_out.csv')]
    assert len(exc_paths) == 1, 'There are no or multiple out files.'
    out = np.loadtxt(out_paths[0], skiprows=1, delimiter=',')

    time_out, T_A, T_B = out.T
    out = np.stack([T_A, T_A], axis=-1)

    assert np.array_equal(time_exc, time_out), 'Different timesteps in ..._exc.csv and ..._out.csv'

    return time_exc, out, exc
