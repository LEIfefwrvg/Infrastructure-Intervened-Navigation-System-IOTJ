"""Occupancy-aware global path planning in the spatial domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import time
from functools import wraps

import numpy as np

from .travel_time import TravelTimeField, estimate_ego_arrival_times


Array = np.ndarray
DEFAULT_GLOBAL_CONTROL_STEP_M = 3.0


@dataclass(frozen=True)
class ObstacleProfile:
    """Predicted obstacle center along the planner's longitudinal stations."""

    s: Array
    y: Array
    safe_squared_distance: float = 4.0
    vehicle_id: str | None = None

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
    solver: str = "psd_dispatch"
    dispatch_sqp_time_limit: float = .08
    dispatch_bnb_time_limit: float = .015
    dispatch_bnb_node_limit: int = 64
    dispatch_bnb_relative_gap: float = .01
    horizon: int = 60
    ds: float = 1.0
    control_step: float = DEFAULT_GLOBAL_CONTROL_STEP_M
    wheelbase: float = 2.6
    steering_min: float = np.deg2rad(-28.0)
    steering_max: float = np.deg2rad(28.0)
    steering_weight: float = 1.0
    heading_weight: float = 10.0
    travel_time_weight: float = 100.0
    cbf_gamma: float = 0.5
    cbf_arrival_tolerance: float = 1.0  # seconds; same absolute clock for all ETAs
    max_iterations: int = 200
    feasibility_tolerance: float = 1e-4
    coarse_factor: int = 5
    variable_steps: bool = True

    def __post_init__(self) -> None:
        if self.solver != "psd_dispatch":
            raise ValueError("Only psd_dispatch is supported")
        positive = (self.ds, self.wheelbase, self.control_step,
                    self.dispatch_sqp_time_limit, self.dispatch_bnb_time_limit)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("Spatial scales and time budgets must be finite and positive")
        if not np.isfinite(self.cbf_arrival_tolerance) or self.cbf_arrival_tolerance < 0:
            raise ValueError("CBF arrival tolerance must be finite and nonnegative")
        counts = (self.horizon - 1, self.dispatch_bnb_node_limit, self.max_iterations)
        if not np.isfinite(counts).all() or min(counts) < 1:
            raise ValueError("horizon must be >= 2; iteration and node limits must be >= 1")
        if not np.isfinite(self.dispatch_bnb_relative_gap) or self.dispatch_bnb_relative_gap < 0:
            raise ValueError("Relative gap must be finite and nonnegative")
        if not np.isfinite((self.steering_min, self.steering_max)).all() or self.steering_min >= self.steering_max:
            raise ValueError("Steering bounds must be finite and increasing")



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
    wheelbase: float = 2.6


def variable_step_indices(ego_times, obstacle_times, coarse_factor=5):
    """Equations (16)-(19); K is the number of full-grid nodes.

    Preserve [0,...,k*], then coarse nodes including K-1. If a vehicle has
    no finite nonnegative time difference, keep the full horizon conservatively.
    """
    ego_times=np.asarray(ego_times,dtype=float)
    if ego_times.ndim!=1 or len(ego_times)<2 or np.isnan(ego_times).any() or np.isneginf(ego_times).any():
        raise ValueError('Ego arrivals must be a one-dimensional array of times or +inf')
    n=len(ego_times); stars=[]
    if not np.isfinite(ego_times).all():return np.arange(n),n-1
    if np.any(np.diff(ego_times)<0):raise ValueError('Ego arrivals must be nondecreasing')
    if coarse_factor < 1: raise ValueError('coarse_factor must be >= 1')
    for times in obstacle_times:
        times=np.asarray(times,dtype=float)
        if times.shape!=ego_times.shape: raise ValueError('Arrival grid shape mismatch')
        diff=ego_times-times
        valid=np.isfinite(diff)&(diff>=0)
        if not valid.any(): return np.arange(n), n-1
        stars.append(int(np.argmin(np.where(valid,diff,np.inf))))
    star=max(stars,default=0)
    nodes=np.unique(np.r_[np.arange(star+1),np.arange(star+coarse_factor,n,coarse_factor),n-1]).astype(int)
    return nodes,star


def control_blocks(stations, step):
    """Map intervals to held controls without removing state/constraint nodes.

    Blocks end at the first existing knot at least ``step`` metres away;
    the final block may be shorter. Curvature retains every original interval.
    """
    stations=np.asarray(stations,float)
    if stations.ndim!=1 or len(stations)<2 or not np.isfinite(stations).all() or np.any(np.diff(stations)<=0):
        raise ValueError("Expected finite increasing stations")
    if not np.isfinite(step) or step<=0:raise ValueError("Positive finite control step required")
    groups=[];group=0;start=stations[0]
    for k in range(len(stations)-1):
        if k and stations[k]-start>=step-1e-10:
            group+=1;start=stations[k]
        groups.append(group)
    return np.eye(group+1)[groups]


def _timed_plan(method):
    @wraps(method)
    def measured(self,*args,**kwargs):
        started=time.perf_counter()
        try:return method(self,*args,**kwargs)
        finally:self.last_call_time_s=time.perf_counter()-started
    return measured


class SpatialGlobalPlanner:
    """Spatial planner using PSD dispatch to SQP or limited spatial B&B."""

    def __init__(self, travel_time: TravelTimeField, config: GlobalPlannerConfig | None = None):
        self.travel_time = travel_time
        self.config = config or GlobalPlannerConfig()
        self.stations=np.arange(self.config.horizon+1)*self.config.ds
        self.initial_y=0.; self.initial_heading=0.
        self.k_star=self.config.horizon
        self.control_indices=np.arange(self.config.horizon+1)

    def _curvature(self, curvature):
        values=np.broadcast_to(np.asarray(curvature,dtype=float),(self.config.horizon,))
        return values[self.control_indices[:-1]].copy()

    def _rollout(self, steering, curvature):
        ds = np.diff(self.stations)
        rate = np.asarray(steering) / self.config.wheelbase - curvature
        heading = np.r_[self.initial_heading, self.initial_heading + np.cumsum(rate * ds)]
        y = np.r_[self.initial_y, self.initial_y + np.cumsum(heading[:-1] * ds)]
        return self.stations, y, heading

    def _objective(self, steering: Array, curvature: Array) -> float:
        s, y, heading = self._rollout(steering, curvature)
        cell_time = self.travel_time.lookup(s[:-1], y[:-1])
        cfg = self.config
        return float(
            np.sum(cfg.steering_weight * steering**2 * np.diff(s) + cfg.heading_weight * heading[:-1]**2 * np.diff(s) + cfg.travel_time_weight * cell_time * np.diff(s)/self.travel_time.grid.ds)
        )

    def _obstacle_cbf_active(self, obstacle):
        """Profiles passed to the solver already contain only ETA-active intervals."""
        return obstacle.active_at(self.stations)

    def _obstacle_arrival_mask(self, obstacle, vehicles, stations, ego_times):
        """Match by ID, or uniquely by geometry for legacy callers; unknown stays active.

        The mask is fixed from the reference-path ETA for this planning call.
        Missing/nonfinite ETAs do not establish temporal separation.
        """
        if obstacle.vehicle_id is not None:
            matches = [v for v in vehicles if v.vehicle_id == obstacle.vehicle_id]
        else:
            matches = []
            for vehicle in vehicles:
                covered = ((obstacle.s >= vehicle.stations[0]) &
                           (obstacle.s <= vehicle.stations[-1]))
                if covered.all() and np.allclose(
                    obstacle.y, np.interp(obstacle.s, vehicle.stations, vehicle.lateral),
                    rtol=0., atol=1e-6,
                ):
                    matches.append(vehicle)
        valid = obstacle.active_at(stations)
        if len(matches) != 1:
            return valid
        vehicle = matches[0]
        other = np.interp(stations, vehicle.stations, vehicle.times,
                          left=np.inf, right=np.inf)
        known = np.isfinite(ego_times) & np.isfinite(other)
        close = np.ones(len(stations), dtype=bool)
        close[known] = (np.abs(ego_times[known] - other[known]) <=
                        self.config.cbf_arrival_tolerance)
        return valid & close

    def _arrival_filtered_obstacles(self, obstacles, masks):
        """Split active interval runs so SQP and B&B use identical CBF rows.

        A spatial CBF uses two endpoints: enable it if either endpoint is
        temporally close (or unknown), provided both lie in the obstacle domain.
        Splitting prevents a gap between two conflicts from acquiring extra rows.
        """
        result = []
        for obstacle, full_mask in zip(obstacles, masks):
            close = full_mask[self.control_indices]
            valid = obstacle.active_at(self.stations)
            active = valid[:-1] & valid[1:] & (close[:-1] | close[1:])
            edges = np.diff(np.r_[False, active, False].astype(int))
            for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                stations = self.stations[first:stop + 1]
                result.append(ObstacleProfile(stations, obstacle.lateral_at(stations),
                                              obstacle.safe_squared_distance, obstacle.vehicle_id))
        return tuple(result)

    def _cbf_constraints(
        self, steering: Array, curvature: Array, obstacles: Sequence[ObstacleProfile]
    ) -> Array:
        s, y, _ = self._rollout(steering, curvature)
        lower,upper=self.travel_time.bounds(s)
        constraints: list[Array] = [y-lower, upper-y]
        for obstacle in obstacles:
            obstacle_y = obstacle.lateral_at(s)
            energy = (y - obstacle_y) ** 2 - obstacle.safe_squared_distance
            spatial_cbf = np.diff(energy) / np.diff(s) + self.config.cbf_gamma * energy[:-1]
            active = self._obstacle_cbf_active(obstacle)
            transition_active = active[:-1] & active[1:]
            if np.any(transition_active):
                constraints.append(spatial_cbf[transition_active])
        return np.concatenate(constraints) if constraints else np.ones(1, dtype=float)

    def _lateral_jacobian(self,query):
        # Euler steering affects heading at the next node, then lateral position.
        distance=np.maximum(np.asarray(query)[:,None]-self.stations[1:],0.)
        return distance*np.diff(self.stations)[None,:]/self.config.wheelbase

    def _objective_jac(self,steering,curvature):
        s,y,heading=self._rollout(steering,curvature); ds=np.diff(s); eps=1e-5
        heading_jac=np.minimum(np.maximum(s[:-1,None]-s[None,:-1],0.),ds[None,:])/self.config.wheelbase
        slope=(self.travel_time.lookup(s[:-1],y[:-1]+eps)-self.travel_time.lookup(s[:-1],y[:-1]-eps))/(2*eps)
        return 2*self.config.steering_weight*steering*ds + heading_jac.T@(2*self.config.heading_weight*heading[:-1]*ds) + self._lateral_jacobian(s[:-1]).T@(self.config.travel_time_weight*slope*ds/self.travel_time.grid.ds)

    def _cbf_jac(self,steering,curvature,obstacles):
        s,y,_=self._rollout(steering,curvature); jac=self._lateral_jacobian(s)
        rows=[jac,-jac]
        for obstacle in obstacles:
            derivative=2*(y-obstacle.lateral_at(s))[:,None]*jac
            active=self._obstacle_cbf_active(obstacle); transitions=active[:-1]&active[1:]
            if transitions.any(): rows.append((np.diff(derivative,axis=0)/np.diff(s)[:,None]+self.config.cbf_gamma*derivative[:-1])[transitions])
        return np.concatenate(rows)

    @_timed_plan
    def plan(
        self,
        *,
        curvature: Array | float = 0.0,
        obstacles: Sequence[ObstacleProfile] = (),
        initial_steering: Array | None = None,
        ego_arrival_times: Array | None = None,
        vehicle_arrivals=(),
        reference_y: Array | None = None,
        arrival_epoch: float | None = None,
        start_s: float = 0.,
        initial_y: float = 0.,
        initial_heading: float = 0.,
        previous_path: GlobalPath | None = None,
    ) -> GlobalPath:
        cfg = self.config
        self.initial_y=initial_y; self.initial_heading=initial_heading
        full_s=start_s+np.arange(cfg.horizon+1)*cfg.ds
        self.last_ego_arrival_times=None;self.last_arrival_reference_y=None
        self.arrival_time_source='disabled'
        obstacles=tuple(obstacles); vehicle_arrivals=tuple(vehicle_arrivals)
        if cfg.variable_steps or obstacles:
            if ego_arrival_times is None:
                if reference_y is not None:
                    reference=np.asarray(reference_y,dtype=float)
                    self.arrival_time_source='travel-time field / explicit reference'
                elif previous_path is not None and previous_path.success:
                    reference=np.interp(full_s,previous_path.s,previous_path.y)
                    before=full_s<previous_path.s[0];after=full_s>previous_path.s[-1]
                    reference[before]=previous_path.y[0]+(full_s[before]-previous_path.s[0])*previous_path.heading[0]
                    reference[after]=previous_path.y[-1]+(full_s[after]-previous_path.s[-1])*previous_path.heading[-1]
                    reference+=initial_y-reference[0]
                    self.arrival_time_source='travel-time field / previous path (reanchored, heading-extended)'
                else:
                    reference=initial_y+(full_s-full_s[0])*initial_heading
                    self.arrival_time_source='travel-time field / current-heading reference'
                if reference.shape!=full_s.shape or not np.isclose(reference[0],initial_y):
                    raise ValueError('Arrival reference must cover the full grid and start at initial_y')
                ego_arrival_times=estimate_ego_arrival_times(self.travel_time,full_s,reference,epoch=arrival_epoch)
                self.last_arrival_reference_y=reference.copy()
            else:
                self.arrival_time_source='caller-supplied override'
            ego_arrival_times=np.asarray(ego_arrival_times,dtype=float)
            if ego_arrival_times.shape!=full_s.shape:raise ValueError('Ego arrivals must match full global grid')
            self.last_ego_arrival_times=ego_arrival_times.copy()
            arrivals=[]
            for vehicle in vehicle_arrivals:
                if not np.allclose(vehicle.stations,full_s): raise ValueError('Arrival stations must match full global grid')
                arrivals.append(vehicle.times)
            if cfg.variable_steps:
                self.control_indices,self.k_star=variable_step_indices(ego_arrival_times,arrivals,cfg.coarse_factor)
            else:
                self.control_indices=np.arange(cfg.horizon+1); self.k_star=cfg.horizon
        else:
            self.control_indices=np.arange(cfg.horizon+1); self.k_star=cfg.horizon
        masks=[self._obstacle_arrival_mask(o,vehicle_arrivals,full_s,ego_arrival_times)
               for o in obstacles]
        # Retain each conflict and its adjacent full-grid nodes, even beyond k_star.
        for mask in masks:
            nodes=np.flatnonzero(mask)
            neighbors=np.clip(np.r_[nodes-1,nodes,nodes+1],0,cfg.horizon)
            self.control_indices=np.union1d(self.control_indices,neighbors).astype(int)
        self.stations=full_s[self.control_indices]
        self.cbf_obstacles=self._arrival_filtered_obstacles(obstacles,masks)
        curvature_array = self._curvature(curvature)
        expansion=control_blocks(self.stations,cfg.control_step)
        self.control_expansion=expansion
        from .psd_dispatch import plan_psd_dispatch
        return plan_psd_dispatch(self,curvature_array,self.cbf_obstacles,initial_steering,previous_path,expansion)

    def _global_path(self,steering,curvature,success,message,objective):
        s,y,heading=self._rollout(steering,curvature)
        cell_time=self.travel_time.lookup(s[:-1],y[:-1])
        return GlobalPath(s,y,heading,np.asarray(steering),curvature,float(objective),success,message,
                          np.asarray(cell_time)*np.diff(s)/self.travel_time.grid.ds,self.config.wheelbase)
