"""Occupancy-aware global path planning in the spatial domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import warnings
import time
from functools import wraps

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .travel_time import TravelTimeField
from .warm_start import remap_controls
from .spatial_dynamics import spatial_rollout
from .sequential_qp import solve_sequential_qp


Array = np.ndarray
DEFAULT_GLOBAL_CONTROL_STEP_M = 3.0


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
    solver: str = "scp"
    scp_max_iterations: int = 80
    scp_qp_max_iterations: int = 4000
    scp_max_seeds: int = 3
    scp_trust_radius: float = 1.0
    scp_penalty: float = 10.0
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
    max_iterations: int = 200
    feasibility_tolerance: float = 1e-4
    coarse_factor: int = 5
    variable_steps: bool = True

    def __post_init__(self) -> None:
        if self.solver not in ("scp", "slsqp"):
            raise ValueError("solver must be scp or slsqp")
        if min(self.scp_max_iterations,self.scp_qp_max_iterations,self.scp_max_seeds)<1 or not np.isfinite([self.scp_trust_radius,self.scp_penalty]).all() or min(self.scp_trust_radius,self.scp_penalty)<=0:
            raise ValueError("Invalid SCP settings")
        if not np.isfinite(self.control_step) or self.control_step <= 0:
            raise ValueError("control_step must be finite and positive")
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
    wheelbase: float = 2.6


def variable_step_indices(ego_times, obstacle_times, coarse_factor=5):
    """Equations (16)-(19); K is the number of full-grid nodes.

    Preserve [0,...,k*], then coarse nodes including K-1. If a vehicle has
    no finite nonnegative time difference, keep the full horizon conservatively.
    """
    ego_times=np.asarray(ego_times,dtype=float)
    n=len(ego_times); stars=[]
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
    """Spatial planner using sequential convexification, with optional legacy SLSQP."""

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
        y = np.r_[self.initial_y, self.initial_y + np.cumsum(heading[:-1] * ds + .5 * rate * ds**2)]
        return self.stations, y, heading

    def _objective(self, steering: Array, curvature: Array) -> float:
        s, y, heading = self._rollout(steering, curvature)
        cell_time = self.travel_time.lookup(s[:-1], y[:-1])
        cfg = self.config
        return float(
            np.sum(cfg.steering_weight * steering**2 * np.diff(s) + cfg.heading_weight * heading[:-1]**2 * np.diff(s) + cfg.travel_time_weight * cell_time * np.diff(s)/self.travel_time.grid.ds)
        )

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
            active = obstacle.active_at(s)
            transition_active = active[:-1] & active[1:]
            if np.any(active):
                constraints.append(energy[active])
            if np.any(transition_active):
                constraints.append(spatial_cbf[transition_active])
        return np.concatenate(constraints) if constraints else np.ones(1, dtype=float)

    def _lateral_jacobian(self,query):
        d1=np.maximum(np.asarray(query)[:,None]-self.stations[:-1],0.)
        d2=np.maximum(np.asarray(query)[:,None]-self.stations[1:],0.)
        return (d1*d1-d2*d2)/(2*self.config.wheelbase)

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
            active=obstacle.active_at(s); transitions=active[:-1]&active[1:]
            if active.any(): rows.append(derivative[active])
            if transitions.any(): rows.append((np.diff(derivative,axis=0)/np.diff(s)[:,None]+self.config.cbf_gamma*derivative[:-1])[transitions])
        return np.concatenate(rows)

    def _candidate_seeds(self, initial: Array | None) -> list[Array]:
        cfg = self.config
        if initial is not None:
            seed = np.asarray(initial, dtype=float)
            if seed.shape != ((len(self.stations)-1),):
                raise ValueError(f"initial_steering must have shape ({(len(self.stations)-1)},)")
            return [np.clip(seed, cfg.steering_min, cfg.steering_max)]
        zeros = np.zeros((len(self.stations)-1), dtype=float)
        bend = np.sin(np.linspace(0.0, 2.0 * np.pi, (len(self.stations)-1), endpoint=False))
        amplitude = 0.35 * min(abs(cfg.steering_min), abs(cfg.steering_max))
        return [zeros, amplitude * bend, -amplitude * bend]

    @_timed_plan
    def plan(
        self,
        *,
        curvature: Array | float = 0.0,
        obstacles: Sequence[ObstacleProfile] = (),
        initial_steering: Array | None = None,
        ego_arrival_times: Array | None = None,
        vehicle_arrivals=(),
        start_s: float = 0.,
        initial_y: float = 0.,
        initial_heading: float = 0.,
        previous_path: GlobalPath | None = None,
    ) -> GlobalPath:
        cfg = self.config
        self.initial_y=initial_y; self.initial_heading=initial_heading
        full_s=start_s+np.arange(cfg.horizon+1)*cfg.ds
        if cfg.variable_steps and ego_arrival_times is not None:
            arrivals=[]
            for vehicle in vehicle_arrivals:
                if not np.allclose(vehicle.stations,full_s): raise ValueError('Arrival stations must match full global grid')
                arrivals.append(vehicle.times)
            self.control_indices,self.k_star=variable_step_indices(ego_arrival_times,arrivals,cfg.coarse_factor)
        else:
            self.control_indices=np.arange(cfg.horizon+1); self.k_star=cfg.horizon
        self.stations=full_s[self.control_indices]
        curvature_array = self._curvature(curvature)
        expansion=control_blocks(self.stations,cfg.control_step)
        reduced=expansion.shape[1] < expansion.shape[0]
        self.control_expansion=expansion
        bounds = [(cfg.steering_min, cfg.steering_max)] * expansion.shape[1]
        if reduced:
            lengths=np.diff(self.stations)
            projection=(expansion.T*lengths)/(expansion.T@lengths)[:,None]
            objective=lambda u,curv:self._objective(expansion@u,curv)
            objective_jac=lambda u,curv:expansion.T@self._objective_jac(expansion@u,curv)
            constraints=lambda u,curv,obs:self._cbf_constraints(expansion@u,curv,obs)
            constraints_jac=lambda u,curv,obs:self._cbf_jac(expansion@u,curv,obs)@expansion
        else:
            objective=self._objective;objective_jac=self._objective_jac
            constraints=self._cbf_constraints;constraints_jac=self._cbf_jac
        if cfg.solver == "scp":
            return self._plan_scp(curvature_array,obstacles,initial_steering,previous_path,expansion)
        best: OptimizeResult | None = None
        best_violation = np.inf
        best_rank: tuple[bool, bool, float, float] | None = None

        self.last_diagnostics=[]
        seeds=self._candidate_seeds(initial_steering)
        if initial_steering is None:
            seeds=[np.clip(seed+curvature_array*cfg.wheelbase,cfg.steering_min,cfg.steering_max) for seed in seeds]
        self.warm_start_used=False
        if previous_path is not None and previous_path.success:
            warm,self.warm_start_used=remap_controls(previous_path.s,previous_path.steering,self.stations,curvature_array*cfg.wheelbase)
            if self.warm_start_used:seeds.insert(0,np.clip(warm,cfg.steering_min,cfg.steering_max))
        for seed_index,seed in enumerate(seeds):
            if reduced:seed=projection@seed
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds during a minimize step",
                    category=RuntimeWarning,
                )
                solve_started=time.perf_counter()
                result = minimize(
                    objective,
                    seed,
                    args=(curvature_array,),
                    jac=objective_jac,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": constraints,
                            "jac": constraints_jac,
                            "args": (curvature_array, obstacles),
                        }
                    ],
                    options={"maxiter": cfg.max_iterations, "ftol": 1e-7, "disp": False},
                )
            solve_elapsed=time.perf_counter()-solve_started
            if reduced:result.x=expansion@result.x
            violation = max(0.0, -float(np.min(self._cbf_constraints(result.x, curvature_array, obstacles))))
            self.last_diagnostics.append(dict(solve_time_s=solve_elapsed,iterations=int(getattr(result,"nit",0)),solver_success=bool(result.success),message=str(result.message),violation=violation,variables=len(seed),state_intervals=len(self.stations)-1,constraints=len(self._cbf_constraints(result.x,curvature_array,obstacles)),control_step_m=cfg.control_step))
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
            if self.warm_start_used and seed_index==0 and result.success and violation<=cfg.feasibility_tolerance:
                break

        assert best is not None
        s, y, heading = self._rollout(best.x, curvature_array)
        cell_time = self.travel_time.lookup(s[:-1], y[:-1])
        feasible = best_violation <= cfg.feasibility_tolerance
        return GlobalPath(
            s=s,
            wheelbase=cfg.wheelbase,
            y=y,
            heading=heading,
            steering=np.asarray(best.x, dtype=float),
            curvature=curvature_array,
            objective=float(best.fun),
            success=bool(best.success and feasible),
            message=str(best.message) if feasible else f"CBF violation: {best_violation:.3g}",
            travel_time=np.asarray(cell_time, dtype=float)*np.diff(s)/self.travel_time.grid.ds,
        )


    def _plan_scp(self,curvature,obstacles,initial,previous,expansion):
        cfg=self.config;ds=np.diff(self.stations);n=expansion.shape[1]
        projection=(expansion.T*ds)/(expansion.T@ds)[:,None]
        full_jac=self._lateral_jacobian(self.stations);gmap=full_jac@expansion
        bounds=np.tile([cfg.steering_min,cfg.steering_max],(n,1))
        self.last_diagnostics=[];self.warm_start_used=False;self.last_retry_diagnostics=[]
        lower,upper=self.travel_time.bounds(self.stations)
        lower=np.broadcast_to(lower,self.stations.shape);upper=np.broadcast_to(upper,self.stations.shape)
        immutable_violation=max(0.,float(lower[0]-self.initial_y),float(self.initial_y-upper[0]))
        for obstacle in obstacles:
            if obstacle.active_at(self.stations[:1])[0]:
                immutable_violation=max(immutable_violation,float(obstacle.safe_squared_distance-(self.initial_y-obstacle.lateral_at(self.stations[:1])[0])**2))
        if immutable_violation>cfg.feasibility_tolerance:
            u=np.clip(curvature*cfg.wheelbase,cfg.steering_min,cfg.steering_max)
            self.last_diagnostics.append(dict(solver='precheck',solve_time_s=0.,iterations=0,solver_success=False,
                message='Initial state violates an immutable constraint',violation=immutable_violation,
                variables=n,constraints=len(self._cbf_constraints(u,curvature,obstacles)),qp_solves=0,qp_events=[]))
            return self._global_path(u,curvature,False,'Initial state infeasible',self._objective(u,curvature))
        # Cache fixed geometry and interval maps only inside this planning call.
        base=self._rollout(np.zeros(len(ds)),curvature)[1]
        obstacle_data=[(o,o.lateral_at(self.stations),o.active_at(self.stations)) for o in obstacles]
        constraint_scale=[np.ones(2*len(self.stations))]
        for obstacle,_,active in obstacle_data:
            scale=max(1.,2*np.sqrt(obstacle.safe_squared_distance))
            if active.any():constraint_scale.append(np.full(np.count_nonzero(active),scale))
            transition=active[:-1]&active[1:]
            if transition.any():constraint_scale.append(np.full(np.count_nonzero(transition),scale))
        constraint_scale=np.concatenate(constraint_scale)
        def objective(u):return self._objective(expansion@u,curvature)
        def gradient(u):return expansion.T@self._objective_jac(expansion@u,curvature)
        def constraints(u):
            y=base+gmap@u;rows=[y-lower,upper-y]
            for obstacle,oy,active in obstacle_data:
                energy=(y-oy)**2-obstacle.safe_squared_distance;transition=active[:-1]&active[1:]
                if active.any():rows.append(energy[active])
                if transition.any():rows.append((np.diff(energy)/ds+cfg.cbf_gamma*energy[:-1])[transition])
            return np.concatenate(rows)
        def jacobian(u):
            y=base+gmap@u;rows=[gmap,-gmap]
            for obstacle,oy,active in obstacle_data:
                derivative=2*(y-oy)[:,None]*gmap;transition=active[:-1]&active[1:]
                if active.any():rows.append(derivative[active])
                if transition.any():rows.append((np.diff(derivative,axis=0)/ds[:,None]+cfg.cbf_gamma*derivative[:-1])[transition])
            return np.concatenate(rows)
        candidates=[]
        feed=curvature*cfg.wheelbase
        if previous is not None and previous.success:
            warm,self.warm_start_used=remap_controls(previous.s,previous.steering,self.stations,feed)
            if self.warm_start_used:candidates.append(('warm',projection@warm))
        if initial is not None:
            candidates.append(('explicit',projection@self._candidate_seeds(initial)[0]))
        else:
            candidates.append(('feedforward',projection@feed))
            length=self.stations[-1]-self.stations[0]
            bend=np.sin(2*np.pi*(self.stations[:-1]-self.stations[0])/length)
            amplitude=min(.35*min(abs(cfg.steering_min),abs(cfg.steering_max)),2*np.pi*cfg.wheelbase*3.5/length**2)
            candidates.extend((label,projection@(feed+sign*amplitude*bend)) for label,sign in (('left',-1),('right',1)))
        unique=[]
        for label,seed in candidates:
            seed=np.clip(seed,bounds[:,0],bounds[:,1])
            if any(np.max(np.abs(gmap@(seed-other)))<1e-6 for _,other in unique):
                self.last_retry_diagnostics.append(dict(seed=label,reason='duplicate predicted path'));continue
            if label=='feedforward' and np.any((constraints(seed)<-cfg.feasibility_tolerance)&(np.linalg.norm(jacobian(seed),axis=1)<1e-10)):
                self.last_retry_diagnostics.append(dict(seed=label,reason='violated constraint has zero local gradient'));continue
            unique.append((label,seed))
        # Prefer feasible / near-feasible seeds; retain the measured warm start first.
        unique.sort(key=lambda item:(item[0]!='warm',float(np.maximum(-constraints(item[1]),0.).sum())))
        self.last_retry_diagnostics.extend(dict(seed=name,reason='seed budget') for name,_ in unique[cfg.scp_max_seeds:])
        best=None;best_rank=None
        hessian=2*cfg.steering_weight*expansion.T@(ds[:,None]*expansion)
        hj=np.minimum(np.maximum(self.stations[:-1,None]-self.stations[None,:-1],0.),ds[None,:])/cfg.wheelbase@expansion
        hessian+=2*cfg.heading_weight*hj.T@(ds[:,None]*hj)
        attempted_candidates=unique[:cfg.scp_max_seeds]
        for index,(label,seed) in enumerate(attempted_candidates):
            started=time.perf_counter()
            result=solve_sequential_qp(objective,gradient,constraints,jacobian,seed,bounds,gmap,hessian,cfg,constraint_scale)
            elapsed=time.perf_counter()-started
            steering=expansion@result.x
            # Always check the original rollout/CBF, independently of the affine cache.
            raw=self._cbf_constraints(steering,curvature,obstacles)
            violation=max(0.,-float(np.min(raw)),float(np.max(cfg.steering_min-steering)),float(np.max(steering-cfg.steering_max))) if np.isfinite(raw).all() and np.isfinite(steering).all() else np.inf
            valid=bool(result.success and violation<=cfg.feasibility_tolerance)
            self.last_diagnostics.append(dict(solver='scp_checked_qp',seed=label,solve_time_s=elapsed,
                iterations=int(result.nit),solver_success=valid,converged=bool(result.converged),kkt_certificate=getattr(result,"kkt_certificate",None),
                message=str(result.message),violation=violation,variables=n,state_intervals=len(ds),
                constraints=len(raw),control_step_m=cfg.control_step,solution_source=result.solution_source,qp_solves=len(result.qp_events),qp_events=result.qp_events))
            rank=(not valid,violation if not valid else 0.,float(result.fun))
            if best_rank is None or rank<best_rank:
                best=(steering,valid,str(result.message),float(result.fun));best_rank=rank
            if valid:
                self.last_retry_diagnostics.extend(dict(seed=name,reason='feasible iterate already found') for name,_ in attempted_candidates[index+1:]);break
        assert best is not None
        return self._global_path(best[0],curvature,best[1],best[2],best[3])

    def _global_path(self,steering,curvature,success,message,objective):
        s,y,heading=self._rollout(steering,curvature)
        cell_time=self.travel_time.lookup(s[:-1],y[:-1])
        return GlobalPath(s,y,heading,np.asarray(steering),curvature,float(objective),success,message,
                          np.asarray(cell_time)*np.diff(s)/self.travel_time.grid.ds,self.config.wheelbase)
