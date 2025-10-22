
from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from dynax.data_handling import Normalizer
from .datareader import load_chicken, load_long_chicken

fair_test_groups = {
        "AP15":     [313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528,],
        "sinAP15":  [835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917, 919, 927,],
        "MS15":     [793, 796, 808, 818, 819, 820, 822, 825, 826, 829, 1042, 1045, 1047, 1051, 1044,],
    }

fair_test_groups["more_AP15"] = list(
        set(range(313, 528)) - set(fair_test_groups["AP15"])
    )


@dataclass
class DataSet:
    ts: np.ndarray
    ys: np.ndarray
    us: np.ndarray

    def __post_init__(self):
        self.ts = np.array(self.ts)
        self.ys = np.array(self.ys)
        self.us = np.array(self.us)

    def __getitem__(self, slice_index):
        return DataSet(self.ts, self.ys[slice_index], self.us[slice_index])
    
    def normalize(self, t_normalizer, y_normalizer, u_normalizer):
        """Normalize the data using the provided normalizers."""
        ts = t_normalizer.normalize(self.ts)
        ys = y_normalizer.normalize(self.ys)
        us = u_normalizer.normalize(self.us)
        return DataSet(ts, ys, us)

    def denormalize(self, t_normalizer, y_normalizer, u_normalizer):
        """Normalize the data using the provided normalizers."""
        ts = t_normalizer.denormalize(self.ts)
        ys = y_normalizer.denormalize(self.ys)
        us = u_normalizer.denormalize(self.us)
        return DataSet(ts, ys, us)
    
    def add_noise(self, noise_amplitude, key):
        """Add noise to the input data."""
        y_key, u_key = jr.split(key)
        ys = self.ys + noise_amplitude * jr.normal(y_key, shape=self.ys.shape)
        us = self.us + noise_amplitude * jr.normal(u_key, shape=self.us.shape)
        return DataSet(self.ts, ys, us)


def prepare_data(
        train_ids = [745, 795], 
        test_ids = fair_test_groups["AP15"]
    ):

    # Stitch together the training data
    ts_train, ys_train, us_train = load_chicken(train_ids)
    ts_vali, ys_vali, us_vali = load_chicken(test_ids)
    ts_long, y_long, u_long = load_long_chicken()

    # ### Compute the normalized data ###
    # The models are trained on normalized data and then the trained models are wrapped
    # by a normalizer
    # Build the normalizers, NOTE: I do not scale the components of y individually.
    ts_shift = 0.0
    ts_scale = jnp.std(ts_train - ts_shift)
    t_normalizer = Normalizer(ts_shift, ts_scale)

    ys_shift = 279.15
    ys_scale = jnp.std(ys_train - ys_shift)
    y_normalizer = Normalizer(ys_shift, ys_scale)

    us_shift = 279.15
    us_scale = jnp.std(us_train - us_shift)
    u_normalizer = Normalizer(us_shift, us_scale)

    # ### Create delayed test data ###
    n_shift = 200  # Number of samples that the trajectories are delayed by
    assert np.all(ts_vali == 5 * np.arange(ts_vali.size)), (
        "The assumption that ts_vali is equidistant with a sample period of 5s is not ture. This means that the code generating the delayed data will produce the right test data and should be changed."
    )
    ts_test_delayed = 5 * np.arange(ts_vali.size + n_shift)
    assert np.all(ys_vali[:, 0] == ys_shift), (
        f"All trajectories should start in the equilibrium position T={ys_shift=}"
    )
    ys_test_delayed = np.pad(ys_vali, ((0, 0), (n_shift, 0), (0, 0)), mode="edge")
    assert np.all(us_vali[:, 0] == us_shift), (
        f"All excitations should start out in the neutral value T={us_shift=}"
    )
    us_test_delayed = np.pad(us_vali, ((0, 0), (n_shift, 0), (0, 0)), mode="edge")

    # ### Put data into data dict ###
    data = dict(
        train_norm   = DataSet(ts_train, ys_train, us_train).normalize(t_normalizer, y_normalizer, u_normalizer),
        test_norm    = DataSet(ts_vali, ys_vali, us_vali).normalize(t_normalizer, y_normalizer, u_normalizer),
        train        = DataSet(ts_train, ys_train, us_train),
        test         = DataSet(ts_vali, ys_vali, us_vali),
        test_delayed = DataSet(ts_test_delayed, ys_test_delayed, us_test_delayed),
        long         = DataSet(ts_long, y_long[None, :], u_long[None, :]),
    )

    return data, t_normalizer, y_normalizer, u_normalizer