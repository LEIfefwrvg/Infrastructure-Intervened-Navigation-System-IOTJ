"""Occupancy-aware global path planning in the spatial domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .travel_time import TravelTimeField


Array = np.ndarray


@dataclass(frozen=True)
class ObstacleProfile:
    """Predicted obstacle center along the planner's longitudinal stations."""

    s: Array
    y: Array
    safe_squared_distance: float = 4.0

    def __post_init__(self) -> None:
        s = np.asarray(self.s, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if s.ndim != 1 or y.shape != s.shape or len(s) < 2:
            raise ValueError("Obstacle s and y must be equal one-dimensional arrays")
        if np.any(np.diff(s) <= 0):
            raise ValueError("Obstacle longitudinal stations must be increasing")
        if self.safe_squared_distance <= 0:
            raise ValueError("safe_squared_distance must be positive")
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "y", y)

    def lateral_at(self, stations: Array) -> Array:
        return np.interp(stations, self.s, self.y)

    def active_at(self, stations: Array) -> Array:
        """Return the longitudinal interval where this prediction is valid."""

        values = np.asarray(stations, dtype=float)
        return (values >= self.s[0]) & (values <= self.s[-1])


@dataclass(frozen=True)
class GlobalPlannerConfig:
    horizon: int = 60
    ds: float = 1.0
    wheelbase: float = 2.6
    steering_min: float = np.deg2rad(-28.0)
    steering_max: float = np.deg2rad(28.0)
    steering_weight: float = 1.0
    travel_time_weight: float = 100.0
    cbf_gamma: float = 0.5
    max_iterations: int = 200
    feasibility_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.horizon < 2 or self.ds <= 0 or self.wheelbase <= 0:
            raise ValueError("horizon, ds, and wheelbase must be positive")
        if self.steering_min >= self.steering_max:
            raise ValueError("steering_min must be smaller than steering_max")


@dataclass(frozen=True)
class GlobalPath:
    s: Array
    y: Array
    heading: Array
    steering: Array
    curvature: Array
    objective: float
    success: bool
    message: str
    travel_time: Array = field(repr=False)


class SpatialGlobalPlanner:
    """Direct-shooting SQP approximation of equations (2)-(20) in the paper."""

    def __init__(self, travel_time: TravelTimeField, config: GlobalPlannerConfig | None = None):
        self.travel_time = travel_time
        self.config = config or GlobalPlannerConfig()

    def _curvature(self, curvature: Array | float) -> Array:
        values = np.broadcast_to(np.asarray(curvature, dtype=float), (self.config.horizon,))
        return values.copy()

    def _rollout(self, steering: Array, curvature: Array) -> tuple[Array, Array, Array]:
        cfg = self.config
        s = np.arange(cfg.horizon + 1, dtype=float) * cfg.ds
        y = np.zeros(cfg.horizon + 1, dtype=float)
        heading = np.zeros(cfg.horizon + 1, dtype=float)
        for k in range(cfg.horizon):
            y[k + 1] = y[k] + heading[k] * cfg.ds
            heading[k + 1] = heading[k] + (
                steering[k] / cfg.wheelbase - curvature[k]
            ) * cfg.ds
        return s, y, heading

    def _objective(self, steering: Array, curvature: Array) -> float:
        s, y, _ = self._rollout(steering, curvature)
        cell_time = self.travel_time.lookup(s[:-1], y[:-1])
        cfg = self.config
        return float(
            np.sum(cfg.steering_weight * steering**2 + cfg.travel_time_weight * cell_time)
        )

    def _cbf_constraints(
        self, steering: Array, curvature: Array, obstacles: Sequence[ObstacleProfile]
    ) -> Array:
        if not obstacles:
            return np.ones(1, dtype=float)
        s, y, _ = self._rollout(steering, curvature)
        constraints: list[Array] = []
        for obstacle in obstacles:
            obstacle_y = obstacle.lateral_at(s)
            energy = (y - obstacle_y) ** 2 - obstacle.safe_squared_distance
            spatial_cbf = np.diff(energy) / self.config.ds + self.config.cbf_gamma * energy[:-1]
            active = obstacle.active_at(s)
            transition_active = active[:-1] & active[1:]
            if np.any(active):
                constraints.append(energy[active])
            if np.any(transition_active):
                constraints.append(spatial_cbf[transition_active])
        return np.concatenate(constraints) if constraints else np.ones(1, dtype=float)

    def _candidate_seeds(self, initial: Array | None) -> list[Array]:
        cfg = self.config
        if initial is not None:
            seed = np.asarray(initial, dtype=float)
            if seed.shape != (cfg.horizon,):
                raise ValueError(f"initial_steering must have shape ({cfg.horizon},)")
            return [np.clip(seed, cfg.steering_min, cfg.steering_max)]
        zeros = np.zeros(cfg.horizon, dtype=float)
        bend = np.sin(np.linspace(0.0, 2.0 * np.pi, cfg.horizon, endpoint=False))
        amplitude = 0.35 * min(abs(cfg.steering_min), abs(cfg.steering_max))
        return [zeros, amplitude * bend, -amplitude * bend]

    def plan(
        self,
        *,
        curvature: Array | float = 0.0,
        obstacles: Sequence[ObstacleProfile] = (),
        initial_steering: Array | None = None,
    ) -> GlobalPath:
        cfg = self.config
        curvature_array = self._curvature(curvature)
        bounds = [(cfg.steering_min, cfg.steering_max)] * cfg.horizon
        best: OptimizeResult | None = None
        best_violation = np.inf
        best_rank: tuple[bool, bool, float, float] | None = None

        for seed in self._candidate_seeds(initial_steering):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds during a minimize step",
                    category=RuntimeWarning,
                )
                result = minimize(
                    self._objective,
                    seed,
                    args=(curvature_array,),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": self._cbf_constraints,
                            "args": (curvature_array, obstacles),
                        }
                    ],
                    options={"maxiter": cfg.max_iterations, "ftol": 1e-7, "disp": False},
                )
            violation = max(0.0, -float(np.min(self._cbf_constraints(result.x, curvature_array, obstacles))))
            rank = (
                violation > cfg.feasibility_tolerance,
                not bool(result.success),
                violation,
                float(result.fun),
            )
            if best_rank is None or rank < best_rank:
                best = result
                best_violation = violation
                best_rank = rank

        assert best is not None
        s, y, heading = self._rollout(best.x, curvature_array)
        cell_time = self.travel_time.lookup(s[:-1], y[:-1])
        feasible = best_violation <= cfg.feasibility_tolerance
        return GlobalPath(
            s=s,
            y=y,
            heading=heading,
            steering=np.asarray(best.x, dtype=float),
            curvature=curvature_array,
            objective=float(best.fun),
            success=bool(best.success and feasible),
            message=str(best.message) if feasible else f"CBF violation: {best_violation:.3g}",
            travel_time=np.asarray(cell_time, dtype=float),
        )
