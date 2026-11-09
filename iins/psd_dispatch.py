import time
import warnings
import numpy as np
from scipy.optimize import minimize
from .qcqp_bnb import build_cbf_quadratics,solve_qcqp
from .warm_start import remap_controls


def plan_psd_dispatch(planner,curvature,obstacles,initial,previous,expansion):
    started=time.perf_counter();cfg=planner.config;s=planner.stations;ds=np.diff(s);n=expansion.shape[1]
    base=planner._rollout(np.zeros(len(ds)),curvature)[1]
    gmap=planner._lateral_jacobian(s)@expansion
    hj=np.minimum(np.maximum(s[:-1,None]-s[None,:-1],0.),ds[None,:])/cfg.wheelbase@expansion
    hessian=2*cfg.steering_weight*expansion.T@(ds[:,None]*expansion)+2*cfg.heading_weight*hj.T@(ds[:,None]*hj)
    lower,upper=planner.travel_time.bounds(s)
    lower=np.broadcast_to(lower,s.shape);upper=np.broadcast_to(upper,s.shape)
    bounds=np.tile([cfg.steering_min,cfg.steering_max],(n,1))
    data=[(o,o.lateral_at(s),planner._obstacle_cbf_active(o)) for o in obstacles]
    rows=build_cbf_quadratics(base,gmap,s,data,cfg.cbf_gamma)
    objective_min=float(np.linalg.eigvalsh(hessian).min())
    # Canonical <=0 constraints have matrices -Q, so check their minimum eigenvalues.
    constraint_min=[float(np.linalg.eigvalsh(-r.Q).min()) for r in rows]
    convex=objective_min>=-1e-10 and all(v>=-1e-10 for v in constraint_min)
    route='sqp' if convex else 'bnb'
    planner.last_diagnostics=[];planner.last_retry_diagnostics=[];planner.warm_start_used=False
    def objective(u):return planner._objective(expansion@u,curvature)
    def gradient(u):return expansion.T@planner._objective_jac(expansion@u,curvature)
    def constraint(u):return planner._cbf_constraints(expansion@u,curvature,obstacles)
    def jacobian(u):return planner._cbf_jac(expansion@u,curvature,obstacles)@expansion
    def violation(u):
        c=constraint(u)
        if not np.isfinite(c).all() or not np.isfinite(u).all():return np.inf
        return max(0.,-float(c.min()),float(np.max(bounds[:,0]-u)),float(np.max(u-bounds[:,1])))
    projection=(expansion.T*ds)/(expansion.T@ds)[:,None]
    feed=curvature*cfg.wheelbase
    seeds=[]
    if previous is not None and previous.success:
        warm,planner.warm_start_used=remap_controls(previous.s,previous.steering,s,feed)
        if planner.warm_start_used:seeds.append(('previous',projection@warm))
    if initial is not None:
        initial=np.asarray(initial,dtype=float)
        if initial.shape!=(len(ds),):
            raise ValueError(f"initial_steering must have shape ({len(ds)},)")
        seeds.append(('explicit',projection@np.clip(initial,cfg.steering_min,cfg.steering_max)))
    else:
        seeds.append(('feedforward',projection@feed))
        length=s[-1]-s[0];bend=np.sin(2*np.pi*(s[:-1]-s[0])/length)
        amplitude=min(.35*min(abs(cfg.steering_min),abs(cfg.steering_max)),2*np.pi*cfg.wheelbase*3.5/length**2)
        seeds.extend((name,projection@(feed+sign*amplitude*bend)) for name,sign in [('left',-1),('right',1)])
    best=None;fallback=np.clip(seeds[0][1],bounds[:,0],bounds[:,1]);attempts=[]
    def remember(u):
        nonlocal best,fallback
        u=np.asarray(u).copy()
        if violation(u)<violation(fallback):fallback=u
        f=float(objective(u))
        if violation(u)<=cfg.feasibility_tolerance and np.isfinite(f) and (best is None or f<best[0]):best=(f,u)
    # One common budget for SQP attempts, including both warm-start branches.
    deadline=time.perf_counter()+cfg.dispatch_sqp_time_limit
    class BudgetStop(Exception):pass
    for name,seed in seeds:
        seed=np.clip(seed,bounds[:,0],bounds[:,1]);remember(seed)
        if time.perf_counter()>=deadline:break
        c=constraint(seed)
        if best is None and np.any((c < -cfg.feasibility_tolerance)&(np.linalg.norm(jacobian(seed),axis=1)<1e-10)):
            attempts.append(dict(seed=name,status='zero-gradient infeasible seed skipped'));continue
        local_start=time.perf_counter();last=seed.copy()
        def callback(u):
            nonlocal last
            last=u.copy();remember(u)
            if time.perf_counter()>=deadline:raise BudgetStop()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Values in x were outside bounds during a minimize step')
                result=minimize(objective,seed,jac=gradient,method='SLSQP',bounds=bounds,
                    constraints=[dict(type='ineq',fun=constraint,jac=jacobian)],callback=callback,
                    options=dict(maxiter=cfg.max_iterations,ftol=1e-7,disp=False))
            remember(result.x);status=str(result.message);nit=int(result.nit)
        except BudgetStop:
            remember(last);status='SQP time budget';nit=None
        attempts.append(dict(seed=name,status=status,iterations=nit,elapsed_seconds=time.perf_counter()-local_start))
        if best is not None:break
    sqp_elapsed=time.perf_counter()-started
    events=[];selected='sqp_incumbent' if best is not None else 'none'
    if not convex:
        # B&B always executes for a nonconvex quadratic model, even with an SQP incumbent.
        # Optimize one local quadratic model over the physical control bounds.
        u=best[1].copy() if best is not None else fallback.copy()
        scale=max(1.,abs(float(objective(u))));h=hessian/scale;g=gradient(u)/scale
        linear=g-h@u;constant=float(.5*u@h@u-g@u)
        x,event=solve_qcqp(h,linear,constant,rows,gmap,lower-base,upper-base,bounds,u,
            time_limit=cfg.dispatch_bnb_time_limit,node_limit=cfg.dispatch_bnb_node_limit,
            gap=cfg.dispatch_bnb_relative_gap,tolerance=cfg.feasibility_tolerance,fast_start=best is not None)
        event['certificate_scope']='One quadratic local objective over full control bounds; not original nonlinear objective'
        event['incumbent_from_sqp']=best is not None
        event['accepted']=False
        events.append(event)
        if event['feasible'] and violation(x)<=cfg.feasibility_tolerance and (best is None or objective(x)<best[0]):
            remember(x);selected='bnb_improvement';event['accepted']=True
    u=best[1] if best is not None else fallback
    # Always check the original rollout, including bounds, after selecting the result.
    raw=constraint(u);residual=violation(u);success=best is not None and residual<=cfg.feasibility_tolerance
    message=('SQP feasible' if convex else 'Limited B&B with verified feasible incumbent') if success else 'No feasible solution within dispatch budgets'
    planner.last_diagnostics=[dict(solver='psd_dispatch',route=route,
        objective_min_eigenvalue=objective_min,canonical_constraint_min_eigenvalues=constraint_min,
        psd_test_scope='Quadratic local model, constraints normalized to <=0',
        constraints=len(raw),variables=n,state_intervals=len(ds),control_step_m=cfg.control_step,
        solver_success=bool(success),violation=residual,message=message,solution_source=selected,
        solve_time_s=time.perf_counter()-started,sqp_phase_time_s=sqp_elapsed,
        sqp_attempts=attempts,qp_events=events,qp_solves=len(events),
        original_problem_global_optimality_proven=False)]
    return planner._global_path(expansion@u,curvature,bool(success),message,float(objective(u)))
