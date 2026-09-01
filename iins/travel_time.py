"""Grid-level speed and cross-grid travel-time estimation.

The paper uses a learned long-term latent speed predictor. A learned predictor can replace
``fit`` later without changing the planner-facing ``TravelTimeField`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


Array = np.ndarray


@dataclass(frozen=True)
class GridSpec:
    """Uniform road grid in longitudinal/lateral vehicle-fixed coordinates."""

    s_min: float
    s_max: float
    y_min: float
    y_max: float
    ds: float
    dy: float

    def __post_init__(self) -> None:
        if self.s_max <= self.s_min or self.y_max <= self.y_min:
            raise ValueError("Grid bounds must have positive extent")
        if self.ds <= 0 or self.dy <= 0:
            raise ValueError("Grid resolution must be positive")

    @property
    def s_centers(self) -> Array:
        count = int(np.ceil((self.s_max - self.s_min) / self.ds))
        return self.s_min + (np.arange(count) + 0.5) * self.ds

    @property
    def y_centers(self) -> Array:
        count = int(np.ceil((self.y_max - self.y_min) / self.dy))
        return self.y_min + (np.arange(count) + 0.5) * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.s_centers), len(self.y_centers)

    def cell_index(self, s: float, y: float) -> tuple[int, int] | None:
        i = int(np.floor((s - self.s_min) / self.ds))
        j = int(np.floor((y - self.y_min) / self.dy))
        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
            return i, j
        return None


@dataclass(frozen=True)
class TravelTimeField:
    """Planner-facing road cost predicted for every grid cell."""

    grid: GridSpec
    velocity: Array
    occupancy: Array
    longitudinal_time: Array

    def __post_init__(self) -> None:
        expected = self.grid.shape
        if self.velocity.shape != expected + (2,):
            raise ValueError(f"velocity must have shape {expected + (2,)}")
        if self.occupancy.shape != expected:
            raise ValueError(f"occupancy must have shape {expected}")
        if self.longitudinal_time.shape != expected:
            raise ValueError(f"longitudinal_time must have shape {expected}")
        interpolator = RegularGridInterpolator(
            (self.grid.s_centers, self.grid.y_centers),
            self.longitudinal_time,
            bounds_error=False,
            fill_value=None,
        )
        object.__setattr__(self, "_interpolator", interpolator)

    def lookup(self, s: Array | float, y: Array | float) -> Array:
        """Interpolate T(s, y), the time required for one longitudinal cell."""

        s_array, y_array = np.broadcast_arrays(
            np.asarray(s, dtype=float), np.asarray(y, dtype=float)
        )
        # Clamp queries to the represented road area. Unbounded extrapolation
        # can otherwise create negative travel-time costs.
        points = np.column_stack(
            (
                np.clip(s_array.ravel(), self.grid.s_centers[0], self.grid.s_centers[-1]),
                np.clip(y_array.ravel(), self.grid.y_centers[0], self.grid.y_centers[-1]),
            )
        )
        values = np.asarray(self._interpolator(points), dtype=float)
        return np.maximum(values, 1e-6).reshape(s_array.shape)

    def transition_time(
        self, s: float, y: float, next_s: float, next_y: float, min_speed: float = 0.5
    ) -> float:
        """Estimate time along an arbitrary transition using the local velocity."""

        velocity_interp = RegularGridInterpolator(
            (self.grid.s_centers, self.grid.y_centers),
            self.velocity,
            bounds_error=False,
            fill_value=None,
        )
        displacement = np.array([next_s - s, next_y - y], dtype=float)
        distance = float(np.linalg.norm(displacement))
        if distance == 0:
            return 0.0
        direction = displacement / distance
        query = [
            np.clip(s, self.grid.s_centers[0], self.grid.s_centers[-1]),
            np.clip(y, self.grid.y_centers[0], self.grid.y_centers[-1]),
        ]
        local_velocity = np.asarray(velocity_interp([query])[0], dtype=float)
        projected_speed = max(float(local_velocity @ direction), min_speed)
        return distance / projected_speed


class GridTravelTimeEstimator:
    """Estimate the paper's occupancy-flow tensor from sampled trajectories.

    Each trajectory is an ``(n, 3)`` array with columns ``[time, s, y]``.
    Segment velocities are accumulated in the cell containing each segment's
    midpoint.  Unobserved cells retain a configurable free-flow prior.
    """

    def __init__(
        self,
        grid: GridSpec,
        *,
        prior_speed: float = 12.0,
        prior_weight: float = 1.0,
        min_speed: float = 0.5,
        max_speed: float = 45.0,
    ) -> None:
        if not 0 < min_speed <= prior_speed <= max_speed:
            raise ValueError("Expected 0 < min_speed <= prior_speed <= max_speed")
        if prior_weight < 0:
            raise ValueError("prior_weight cannot be negative")
        self.grid = grid
        self.prior_speed = prior_speed
        self.prior_weight = prior_weight
        self.min_speed = min_speed
        self.max_speed = max_speed

    def fit(self, trajectories: Iterable[Array]) -> TravelTimeField:
        shape = self.grid.shape
        weighted_velocity = np.zeros(shape + (2,), dtype=float)
        weighted_velocity[..., 0] = self.prior_speed * self.prior_weight
        weights = np.full(shape, self.prior_weight, dtype=float)
        observations = np.zeros(shape, dtype=float)

        for trajectory in trajectories:
            samples = np.asarray(trajectory, dtype=float)
            if samples.ndim != 2 or samples.shape[1] != 3:
                raise ValueError("Each trajectory must have shape (n, 3): [time, s, y]")
            if len(samples) < 2:
                continue
            dt = np.diff(samples[:, 0])
            displacement = np.diff(samples[:, 1:3], axis=0)
            midpoints = 0.5 * (samples[:-1, 1:3] + samples[1:, 1:3])
            for duration, delta, midpoint in zip(dt, displacement, midpoints):
                if duration <= 0:
                    continue
                index = self.grid.cell_index(float(midpoint[0]), float(midpoint[1]))
                if index is None:
                    continue
                velocity = delta / duration
                magnitude = float(np.linalg.norm(velocity))
                if magnitude > self.max_speed:
                    velocity *= self.max_speed / magnitude
                weighted_velocity[index] += velocity
                weights[index] += 1.0
                observations[index] += 1.0

        velocity = weighted_velocity / np.maximum(weights[..., None], 1e-12)
        occupancy = observations / max(float(observations.max()), 1.0)
        forward_speed = np.maximum(velocity[..., 0], self.min_speed)
        longitudinal_time = self.grid.ds / forward_speed
        return TravelTimeField(self.grid, velocity, occupancy, longitudinal_time)
