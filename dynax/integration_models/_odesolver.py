"""
Models of the form
`ys = ODESolve(ts, y0, us; f(t, y, u))`
"""


import equinox as eqx
import diffrax
from diffrax import backward_hermite_coefficients, CubicInterpolation

import jax
import jax.numpy as jnp

from jaxtyping import Array


class ODESolver(eqx.Module):
    derivative_model: eqx.Module
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    max_steps: int
    augmentation: Array # Array of initial values in the augmented dimension, its size determines the number of added dimensions.
    augmentation_learnable: bool = eqx.field(static=True)

    def __init__(self, 
                 derivative_model: eqx.Module, 
                 *, 
                 augmentation: int=0, 
                 augmentation_learnable: bool=False,
                 solver: diffrax.AbstractSolver=diffrax.Tsit5(),
                 stepsize_controller: diffrax.AbstractStepSizeController=diffrax.PIDController(rtol=1e-6, atol=1e-6),
                 max_steps: int=4096,):
        """Numerically solve the ODE `dy_dt = derivative_model(t, y, u(t))`.

        Args:
            derivative_model (eqx.Module): Function or model that takes the arguments (t, y, u).
            augmentation (int, optional): Number of dimensions that get added to y0 before integrating. Defaults to 0.
            augmentation_learnable (bool, optional): If true, then the initial values of the augmented dimensions are optimized during trainig.
            solver (diffrax.AbstractSolver, optional): Diffrax solver instance. Defaults to diffrax.Tsit5().
            stepsize_controller (diffrax.AbstractStepSizeController, optional): Diffrac stepsize_controller instance. Defaults to diffrax.PIDController(rtol=1e-6, atol=1e-6).
            max_steps (int, optional): Maximum number of steps for the integrator. Defaults to 4096.
        """
        super().__init__()

        self.derivative_model = derivative_model
        self.augmentation_learnable = augmentation_learnable
        self.augmentation = jnp.zeros(augmentation)

        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.max_steps = max_steps

    def __call__(self, ts:Array, y0:Array, us:Array|None) -> Array:
        """Solve the ODE defined by `dy_dt = derivative_model(t, y, u(t))`.

        Args:
            ts (Array): Array of the timestamps. `shape=(num_time, )`
            y0 (Array): Array of the initial state. `shape=(state_size, )`
            us (Array | None): Array of the inputs or None if no inpus are supplied. `shape=(num_time, input_size)`

        Returns:
            Array: Solution vector. `shape=(num_time, state_size)`
        """

        ys = self.get_augmented_trajectory(ts, y0, us)
        # Remove the augmentation dimension and return
        state_size = ys.shape[1] - self.augmentation.size
        return ys[:, :state_size]

    def get_augmented_trajectory(self, ts:Array, y0:Array, us:Array|None) -> Array:
        """Solve the ODE defined by `dy_dt = derivative_model(t, y, u(t))`.

        Args:
            ts (Array): Array of the timestamps. `shape=(num_time, )`
            y0 (Array): Array of the initial state. `shape=(state_size, )`
            us (Array | None): Array of the inputs or None if no inpus are supplied. `shape=(num_time, input_size)`

        Returns:
            Array: Solution vector. `shape=(num_time, state_size + aug_size)`
        """

        # Add the augmentation dimensions to the inital state

        if self.augmentation_learnable:
            augmentation = self.augmentation
        else:
            augmentation = jax.lax.stop_gradient(self.augmentation)

        y0 = jnp.concat([y0, augmentation])

        # If inputs are supplied, then interpolate them
        if us is not None:
            coeffs = backward_hermite_coefficients(ts, us)
            u_interp = CubicInterpolation(ts, coeffs)
        else:
            u_interp = None

        # Define the funtion to be intergrated
        def func(t, y, u_interp):
            if u_interp is None:
                u = None
            else:
                u = u_interp.evaluate(t)
            
            return self.derivative_model(t, y, u)

        # Solve the ODE using diffrax
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            self.solver,
            t0=ts[0], t1=ts[-1], 
            dt0=ts[1] - ts[0], y0=y0,
            args=u_interp,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
        )

        ys = solution.ys
        return ys





