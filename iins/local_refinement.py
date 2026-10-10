"""Longitudinal-lateral coupled spatial-domain MPC path refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import time
from scipy.optimize import minimize, OptimizeResult
from scipy import sparse
from .certified_qp import CertifiedQPSolver

import numpy as np

from .global_planner import GlobalPath
from .warm_start import remap_controls
from .spatial_dynamics import spatial_matrices,spatial_rollout


Array = np.ndarray


@dataclass(frozen=True)
class LocalRefinementConfig:
    solver: str = "osqp"
    max_deceleration: float | None = 6.0
    max_acceleration: float = 2.0
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
    safe_time_gap: float = 1.0
    spatial_step: float = 1.0
    max_branch_vehicles: int = 8
    feasibility_tolerance: float = 1e-4
    max_path_deviation: float = .35

    def __post_init__(self) -> None:
        if self.max_deceleration is not None and (not np.isfinite(self.max_deceleration) or self.max_deceleration<=0):
            raise ValueError("max_deceleration must be finite and positive")
        if not np.isfinite(self.max_acceleration) or self.max_acceleration <= 0:
            raise ValueError("max_acceleration must be finite and positive")
        if self.solver not in ("osqp", "slsqp"):
            raise ValueError("solver must be osqp or slsqp")
        if self.safe_time_gap<=0 or self.spatial_step<=0 or self.max_path_deviation<=0 or self.max_branch_vehicles<1:
            raise ValueError("Invalid safety/refinement parameters")
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
    success: bool = True
    message: str = ""
    arrival_gaps: dict = field(default_factory=dict)
    passing_order: dict = field(default_factory=dict)


class SpatialMPCRefiner:
    """Constrained finite-horizon spatial MPC, with enumerated passing orders.

    State: ``[arrival_time, lateral_position, slowness, heading]``.
    Control: ``[moderation, front_wheel_angle]``.
    """

    def __init__(self, config: LocalRefinementConfig | None = None):
        self.config = config or LocalRefinementConfig()

    def _model(self, ds: float, curvature: float) -> tuple[Array, Array, Array]:
        return spatial_matrices(ds,curvature,self.config.wheelbase)

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
        vehicle_arrivals=(),
        road_bounds=None,
        obstacles=(),
        previous_path: LocalPath | None = None,
    ) -> LocalPath:
        cfg = self.config
        if not np.isclose(global_path.wheelbase,cfg.wheelbase):
            raise ValueError("Global and local planners must share the same wheelbase")
        global_path = resample_path(global_path,cfg.spatial_step)
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

        # Arrival targets must allow the initial vehicle to accelerate to the
        # desired speed. An instantaneous-speed time reference causes overshoot
        # followed by braking merely to catch up an unreachable clock.
        reachable=np.minimum(speed,np.sqrt((1/state0[2])**2+2*cfg.max_acceleration*(s-s[0])))
        desired[:,0]=np.r_[0.,np.cumsum(2*ds/(reachable[:-1]+reachable[1:]))]
        desired[:,0] += state0[0]
        return self._constrained(global_path,s,desired,state0,initial_steering,
                                 initial_moderation,vehicle_arrivals,road_bounds,previous_path,obstacles)

    def _constrained(self,path,s,desired,state0,initial_steering,initial_moderation,arrivals,road_bounds=None,previous_path=None,obstacles=()):
        cfg=self.config; ds=np.diff(s); n=len(ds)
        base=np.empty((n+1,4)); jac=np.zeros((n+1,4,2*n)); base[0]=state0
        for k,step in enumerate(ds):
            a,b,c=self._model(step,path.curvature[k]); base[k+1]=a@base[k]+c
            jac[k+1]=a@jac[k]; jac[k+1,:,2*k:2*k+2]+=b
        q=np.asarray(cfg.state_weights); r=np.tile(cfg.control_weights,n)
        flat=jac.reshape(-1,2*n); weights=np.tile(q,n+1)
        target=(base-desired).ravel()
        hess=2*(flat.T@(weights[:,None]*flat)+np.diag(r))
        linear=2*flat.T@(weights*target)
        def objective(u): return .5*u@hess@u+linear@u
        def grad(u): return hess@u+linear
        rows=[]; offsets=[]
        if road_bounds is not None:
            lower,upper=[np.broadcast_to(np.asarray(x,float),(n+1,)) for x in road_bounds]
        for k in range(n+1):
            if road_bounds is not None:
                rows.extend([jac[k,1],-jac[k,1]])
                offsets.extend([base[k,1]-lower[k],upper[k]-base[k,1]])
            rows.extend([jac[k,2],-jac[k,2]])
            offsets.extend([base[k,2]-1/cfg.speed_limit,1/cfg.minimum_speed-base[k,2]])
            if arrivals:
                rows.extend([jac[k,1],-jac[k,1]])
                offsets.extend([base[k,1]-desired[k,1]+cfg.max_path_deviation,
                                desired[k,1]+cfg.max_path_deviation-base[k,1]])
        for k in range(n):
            for channel,limit,previous in ((0,cfg.moderation_rate_limit*ds[k],initial_moderation),(1,0,initial_steering)):
                delta=np.zeros(2*n); delta[2*k+channel]=1
                if k: delta[2*(k-1)+channel]=-1
                previous=previous if k==0 else 0.
                if channel==0:
                    rows.extend([delta,-delta]); offsets.extend([limit-previous,limit+previous])
                else:
                    rate=cfg.steering_rate_limit*ds[k]
                    rows.extend([rate*jac[k,2]+delta,rate*jac[k,2]-delta])
                    offsets.extend([rate*base[k,2]-previous,rate*base[k,2]+previous])
        # z(s) is affine within each held-control interval. Its minimum is
        # attained at an endpoint. Tangent(z**3) <= z**3 for positive z:
        # these affine endpoint bounds guarantee physical acceleration on the
        # entire interval, including the high-speed end of acceleration.
        zref=min(float(state0[2]),float(np.min(desired[:,2])))
        for k in range(n):
            for endpoint in (k,k+1):
                for limit,sign in ((cfg.max_acceleration,1.),(cfg.max_deceleration,-1.)):
                    if limit is None: continue
                    row=limit*3*zref**2*jac[endpoint,2].copy()
                    row[2*k]+=sign
                    rows.append(row)
                    offsets.append(limit*(3*zref**2*base[endpoint,2]-2*zref**3))
        # Preserve the global planner's obstacle side in a convex local corridor.
        # Stationary vehicles have no finite arrival-time profile, so time-gap
        # constraints alone cannot protect this geometric clearance.
        for obstacle in obstacles:
            active=obstacle.active_at(s)
            oy=obstacle.lateral_at(s)
            side=np.where(path.y[:n+1]>=oy,1.,-1.)
            for k in np.flatnonzero(active):
                rows.append(side[k]*jac[k,1])
                offsets.append(side[k]*(base[k,1]-oy[k])-np.sqrt(obstacle.safe_squared_distance))
        common=np.asarray(rows); common_b=np.asarray(offsets); conflicts=[]
        for arrival in arrivals:
            # Interpolate finite predictions only within their known station range.
            valid=np.isfinite(arrival.times)
            if not valid.any(): continue
            positions=arrival.stations[valid]
            mask=(s>=positions[0])&(s<=positions[-1])
            times=np.interp(s,positions,arrival.times[valid])
            # Conservatively enforce time separation for every reached station.
            # This also covers lateral deviations during the optimization.
            if mask.any(): conflicts.append((arrival.vehicle_id,mask,times))
        if len(conflicts)>cfg.max_branch_vehicles:
            raise ValueError('Too many crossing vehicles for exact enumeration; shorten local horizon')
        tail=np.zeros((n,2));tail[:,1]=path.curvature[:n]*cfg.wheelbase
        self.warm_start_used=False
        if previous_path is not None and previous_path.success:
            tail,self.warm_start_used=remap_controls(previous_path.s,np.column_stack([previous_path.moderation,previous_path.steering]),s,tail)
        tail[:,0]=np.clip(tail[:,0],-cfg.moderation_limit,cfg.moderation_limit)
        tail[:,1]=np.clip(tail[:,1],cfg.steering_min,cfg.steering_max)
        seed=tail.ravel() if self.warm_start_used else np.zeros(2*n)
        self.last_diagnostics=[]
        best=None; best_order=None; best_violation=np.inf
        for order in product((0,1),repeat=len(conflicts)):
            aa=[common]; bb=[common_b]
            for (_,mask,times),epsilon in zip(conflicts,order):
                sign=1 if epsilon else -1
                aa.append(sign*jac[mask,0]); bb.append(sign*(base[mask,0]-times[mask])-cfg.safe_time_gap)
            a=np.concatenate(aa); b=np.concatenate(bb)
            bounds=np.tile([[-cfg.moderation_limit,cfg.moderation_limit],
                            [cfg.steering_min,cfg.steering_max]],(n,1))
            seeds=[seed,np.zeros(2*n)] if self.warm_start_used and cfg.solver == 'slsqp' else [seed]
            for candidate in seeds:
                solve_started=time.perf_counter()
                if cfg.solver == 'osqp':
                    matrix=sparse.vstack([sparse.csc_matrix(a),sparse.eye(2*n)],format='csc')
                    qp=CertifiedQPSolver(max_iterations=20000)
                    solved=qp.solve(sparse.csc_matrix(np.triu(hess)),linear,matrix,
                                    np.r_[-b,bounds[:,0]],
                                    np.r_[np.full(len(b),np.inf),bounds[:,1]],candidate)
                    x=solved.x if solved.success else candidate
                    result=OptimizeResult(x=x,fun=objective(x),success=solved.success,
                        nit=sum(attempt['iterations'] for attempt in solved.attempts),message=solved.status)
                else:
                    result=minimize(objective,candidate,jac=grad,method='SLSQP',
                        bounds=bounds,
                        constraints=[dict(type='ineq',fun=lambda u:a@u+b,jac=lambda u:a)],
                        options=dict(maxiter=160,ftol=1e-8))
                solve_elapsed=time.perf_counter()-solve_started
                violation=max(0.,-float(np.min(a@result.x+b)),float(np.max(bounds[:,0]-result.x)),float(np.max(result.x-bounds[:,1])))
                z=(base+np.einsum('ijk,k->ij',jac,result.x))[:,2]
                if np.all(z>0):
                    accelerations=-result.x[::2,None]/np.column_stack([z[:-1],z[1:]])**3
                    violation=max(violation,float(np.max(accelerations))-cfg.max_acceleration)
                    if cfg.max_deceleration is not None:
                        violation=max(violation,float(-np.min(accelerations))-cfg.max_deceleration)
                else: violation=np.inf
                self.last_diagnostics.append(dict(solver=cfg.solver,constraints=len(b)+2*n,order=list(order),solve_time_s=solve_elapsed,iterations=int(getattr(result,"nit",0)),variables=2*n,solver_success=bool(result.success),message=str(result.message),violation=violation))
                if cfg.solver == 'osqp':
                    self.last_diagnostics[-1].update(qp_certified=solved.success,
                        qp_certificate=solved.certificate,qp_attempts=solved.attempts,
                        converged=bool(solved.success and violation<=cfg.feasibility_tolerance))
                if violation<=cfg.feasibility_tolerance and result.success:
                    if best is None or result.fun<best.fun: best=result; best_order=order; best_violation=violation
                    break
        success=best is not None
        controls=best.x if success else np.zeros(2*n)
        states=base+np.einsum('ijk,k->ij',jac,controls); u=controls.reshape(n,2)
        gaps={name:float(np.min(np.abs(states[mask,0]-times[mask]))) for name,mask,times in conflicts}
        return LocalPath(s,states[:,0],states[:,1],1/states[:,2],states[:,2],states[:,3],u[:,0],u[:,1],desired,
            success,'Eq31 feasible' if success else 'No feasible passing order; onboard fallback required',gaps,
            dict(zip([x[0] for x in conflicts],best_order or ())))


def resample_path(path,step=1.):
    """Subdivide original intervals and reintegrate their unchanged controls.

    Preserve every original knot, including non-integer breakpoints, so no fine
    interval crosses a steering/curvature discontinuity. Travel-time costs are
    split by interval length; they are learned costs, not dynamic state times.
    """
    if step<=0:raise ValueError('Positive spatial step required')
    if np.max(np.diff(path.s))<=step+1e-9:return path
    pieces=[np.linspace(a,b,int(np.ceil((b-a)/step))+1)[:-1] for a,b in zip(path.s[:-1],path.s[1:])]
    s=np.r_[np.concatenate(pieces),path.s[-1]]
    idx=np.clip(np.searchsorted(path.s,s[:-1],side='right')-1,0,len(path.steering)-1)
    controls=np.column_stack([np.zeros(len(idx)),path.steering[idx]])
    states=spatial_rollout(s,controls,path.curvature[idx],[0.,path.y[0],0.,path.heading[0]],path.wheelbase)
    return GlobalPath(s,states[:,1],states[:,3],path.steering[idx],path.curvature[idx],
                      path.objective,path.success,path.message,
                      path.travel_time[idx]*np.diff(s)/np.diff(path.s)[idx],path.wheelbase)
