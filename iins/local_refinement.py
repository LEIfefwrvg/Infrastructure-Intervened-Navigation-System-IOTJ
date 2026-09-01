"""Longitudinal-lateral coupled spatial-domain MPC path refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .global_planner import GlobalPath


Array = np.ndarray


@dataclass(frozen=True)
class LocalRefinementConfig:
    horizon: int = 25
    wheelbase: float = 2.6
    speed_limit: float = 20.0
    minimum_speed: float = 0.5
    steering_min: float = np.deg2rad(-28.0)
    steering_max: float = np.deg2rad(28.0)
    steering_rate_limit: float = np.deg2rad(25.0)
    moderation_limit: float = 0.03
    moderation_rate_limit: float = 0.05
    state_weights: tuple[float, float, float, float] = (0.2, 80.0, 30.0, 50.0)
    control_weights: tuple[float, float] = (4.0, 8.0)

    def __post_init__(self) -> None:
        if self.horizon < 2 or self.wheelbase <= 0:
            raise ValueError("horizon and wheelbase must be positive")
        if not 0 < self.minimum_speed <= self.speed_limit:
            raise ValueError("Expected 0 < minimum_speed <= speed_limit")


@dataclass(frozen=True)
class LocalPath:
    s: Array
    arrival_time: Array
    y: Array
    speed: Array
    slowness: Array
    heading: Array
    moderation: Array
    steering: Array
    desired_state: Array


class SpatialMPCRefiner:
    """Finite-horizon affine LQR with the paper's spatial MPC state.

    State: ``[arrival_time, lateral_position, slowness, heading]``.
    Control: ``[moderation, front_wheel_angle]``.
    """

    def __init__(self, config: LocalRefinementConfig | None = None):
        self.config = config or LocalRefinementConfig()

    def _model(self, ds: float, curvature: float) -> tuple[Array, Array, Array]:
        cfg = self.config
        a = np.eye(4, dtype=float)
        a[0, 2] = ds
        a[1, 3] = ds
        b = np.zeros((4, 2), dtype=float)
        b[2, 0] = ds
        b[3, 1] = ds / cfg.wheelbase
        c = np.array([0.0, 0.0, 0.0, -curvature * ds], dtype=float)
        return a, b, c

    @staticmethod
    def _augmented_cost(q: Array, desired: Array) -> Array:
        cost = np.zeros((5, 5), dtype=float)
        cost[:4, :4] = q
        cost[:4, 4] = -q @ desired
        cost[4, :4] = cost[:4, 4]
        cost[4, 4] = float(desired @ q @ desired)
        return cost

    def refine(
        self,
        global_path: GlobalPath,
        *,
        desired_speed: float | Array,
        initial_state: Array | None = None,
        initial_steering: float = 0.0,
        initial_moderation: float = 0.0,
    ) -> LocalPath:
        cfg = self.config
        horizon = min(cfg.horizon, len(global_path.steering))
        if horizon < 2:
            raise ValueError("Global path is too short for local refinement")
        s = np.asarray(global_path.s[: horizon + 1], dtype=float)
        ds = np.diff(s)
        if np.any(ds <= 0):
            raise ValueError("Global path stations must be increasing")

        speed = np.broadcast_to(np.asarray(desired_speed, dtype=float), (horizon + 1,)).copy()
        speed = np.clip(speed, cfg.minimum_speed, cfg.speed_limit)
        desired = np.column_stack(
            (
                np.r_[0.0, np.cumsum(ds / speed[:-1])],
                global_path.y[: horizon + 1],
                1.0 / speed,
                global_path.heading[: horizon + 1],
            )
        )
        if initial_state is None:
            state0 = desired[0].copy()
        else:
            state0 = np.asarray(initial_state, dtype=float)
            if state0.shape != (4,):
                raise ValueError("initial_state must have shape (4,)")

        q = np.diag(np.asarray(cfg.state_weights, dtype=float))
        r = np.diag(np.asarray(cfg.control_weights, dtype=float))
        gains: list[Array] = [np.empty((2, 5)) for _ in range(horizon)]
        p = 2.0 * self._augmented_cost(q, desired[horizon])

        models: list[tuple[Array, Array, Array]] = []
        for k in range(horizon):
            models.append(self._model(float(ds[k]), float(global_path.curvature[k])))

        for k in range(horizon - 1, -1, -1):
            a, b, c = models[k]
            a_aug = np.eye(5, dtype=float)
            a_aug[:4, :4] = a
            a_aug[:4, 4] = c
            b_aug = np.zeros((5, 2), dtype=float)
            b_aug[:4] = b
            hessian = r + b_aug.T @ p @ b_aug
            gain = -np.linalg.solve(hessian, b_aug.T @ p @ a_aug)
            gains[k] = gain
            closed_loop = a_aug + b_aug @ gain
            p = self._augmented_cost(q, desired[k]) + gain.T @ r @ gain + closed_loop.T @ p @ closed_loop

        states = np.zeros((horizon + 1, 4), dtype=float)
        controls = np.zeros((horizon, 2), dtype=float)
        states[0] = state0
        previous_steering = float(initial_steering)
        previous_moderation = float(initial_moderation)
        min_slowness = 1.0 / cfg.speed_limit
        max_slowness = 1.0 / cfg.minimum_speed

        for k, (a, b, c) in enumerate(models):
            augmented_state = np.r_[states[k], 1.0]
            moderation, steering = gains[k] @ augmented_state

            moderation = float(np.clip(moderation, -cfg.moderation_limit, cfg.moderation_limit))
            moderation_delta = cfg.moderation_rate_limit * ds[k]
            moderation = float(
                np.clip(
                    moderation,
                    previous_moderation - moderation_delta,
                    previous_moderation + moderation_delta,
                )
            )
            moderation = max(moderation, (min_slowness - states[k, 2]) / ds[k])
            moderation = min(moderation, (max_slowness - states[k, 2]) / ds[k])

            dt = max(ds[k] * max(states[k, 2], min_slowness), 1e-3)
            steering_delta = cfg.steering_rate_limit * dt
            steering = float(
                np.clip(steering, previous_steering - steering_delta, previous_steering + steering_delta)
            )
            steering = float(np.clip(steering, cfg.steering_min, cfg.steering_max))

            controls[k] = moderation, steering
            states[k + 1] = a @ states[k] + b @ controls[k] + c
            states[k + 1, 2] = np.clip(states[k + 1, 2], min_slowness, max_slowness)
            previous_moderation = moderation
            previous_steering = steering

        return LocalPath(
            s=s,
            arrival_time=states[:, 0],
            y=states[:, 1],
            speed=1.0 / states[:, 2],
            slowness=states[:, 2],
            heading=states[:, 3],
            moderation=controls[:, 0],
            steering=controls[:, 1],
            desired_state=desired,
        )

