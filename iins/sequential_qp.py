"""Elastic trust-region sequential convexification using sparse OSQP subproblems.

Elastic variables exist only inside the search. A returned successful iterate
must satisfy the original nonlinear constraints and control bounds independently.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, nnls
from .certified_qp import CertifiedQPSolver


def _dense_csc(values):
    """Keep structural zeros so OSQP matrix updates preserve the sparsity pattern."""
    rows,cols=values.shape
    return sparse.csc_matrix((values.T.ravel(),np.tile(np.arange(rows),cols),
                              np.arange(cols+1)*rows),shape=values.shape)


def solve_sequential_qp(objective,gradient,constraints,jacobian,initial,bounds,
                        lateral_jacobian,control_hessian,config,constraint_scale=None):
    u=np.asarray(initial,float).copy();bounds=np.asarray(bounds,float)
    u=np.clip(u,bounds[:,0],bounds[:,1]);n=len(u)
    f=float(objective(u));c=np.asarray(constraints(u));m=len(c)
    scale=max(1.,abs(f));penalty=config.scp_penalty
    row_scale=np.ones(m) if constraint_scale is None else np.asarray(constraint_scale,float)
    control_scale=.05
    gmap=np.asarray(lateral_jacobian)
    hessian=control_hessian/scale + .1*gmap.T@gmap/max(1,len(gmap)) + 1e-6*np.eye(n)

    radius=config.scp_trust_radius
    qp=CertifiedQPSolver(config.scp_qp_max_iterations)
    events=[];best=None;converged=False;stalls=0

    def violation(x,values):
        if not np.isfinite(x).all() or not np.isfinite(values).all():return np.inf
        return max(0.,-float(np.min(values)),float(np.max(bounds[:,0]-x)),float(np.max(x-bounds[:,1])))

    def remember(x,value,values,source='scp_iterate'):
        nonlocal best
        residual=violation(x,values)
        if residual<=config.feasibility_tolerance and np.isfinite(value):
            if best is None or value<best.fun:
                best=OptimizeResult(x=x.copy(),fun=value,success=True,violation=residual,solution_source=source)

    def kkt(x):
        values=np.r_[constraints(x)/row_scale,(x-bounds[:,0])/control_scale,(bounds[:,1]-x)/control_scale]
        derivatives=np.vstack([jacobian(x)*control_scale/row_scale[:,None],np.eye(n),-np.eye(n)])
        grad=np.asarray(gradient(x))*control_scale/scale
        active=values<=1e-5
        multipliers=np.zeros(len(values))
        if active.any():
            matrix=derivatives[active].T
            norms=np.maximum(np.linalg.norm(matrix,axis=0),1e-12)
            try:
                fitted,_=nnls(matrix/norms,grad,maxiter=max(1000,10*len(values)))
                multipliers[active]=fitted/norms
            except RuntimeError:
                return dict(valid=False,stationarity=float('inf'),complementarity=float('inf'),primal=violation(x,constraints(x)))
        stationarity=float(np.max(np.abs(grad-derivatives.T@multipliers)))/max(1.,float(np.max(np.abs(grad))))
        complementarity=float(np.sum(np.abs(values*multipliers)))
        primal=violation(x,constraints(x))
        return dict(valid=bool(primal<=config.feasibility_tolerance and stationarity<=1e-4 and complementarity<=1e-5),
                    stationarity=stationarity,complementarity=complementarity,primal=primal)

    remember(u,f,c,source='feasible_seed')
    message='SCP iteration limit'
    for iteration in range(config.scp_max_iterations):
        gradient_value=np.asarray(gradient(u))/scale
        j=np.asarray(jacobian(u))
        if not (np.isfinite(gradient_value).all() and np.isfinite(j).all() and np.isfinite(c).all()):
            message='Nonfinite local model';break
        # d is a scaled control increment. One normalized L-infinity elastic
        # variable avoids adding a separate optimization variable for every row.
        a=sparse.bmat([[_dense_csc(j*control_scale/row_scale[:,None]),np.ones((m,1))],
                       [sparse.eye(n),None],
                       [sparse.csc_matrix(gmap*control_scale),None],
                       [None,sparse.eye(1)]],format='csc')
        lower=np.r_[-c/row_scale,(bounds[:,0]-u)/control_scale,np.full(len(gmap),-radius),0.]
        upper=np.r_[np.full(m,np.inf),(bounds[:,1]-u)/control_scale,np.full(len(gmap),radius),np.inf]
        p=sparse.block_diag((sparse.csc_matrix(np.triu(hessian)*control_scale**2),sparse.eye(1)*1e-8),format='csc')
        q=np.r_[gradient_value*control_scale,penalty]
        solved=qp.solve(p,q,a,lower,upper,np.r_[np.zeros(n),max(0.,float(np.max(-c/row_scale)))])
        event=dict(iteration=iteration,qp_status=solved.status,qp_backend=solved.backend,
                   qp_iterations=sum(x['iterations'] for x in solved.attempts if x['backend']=='osqp'),
                   qp_fallback_iterations=sum(x['iterations'] for x in solved.attempts if x['backend']=='clarabel'),
                   qp_variables=n+1,qp_setup_update_s=solved.setup_time_s,qp_solve_s=solved.solve_time_s,
                   qp_attempts=solved.attempts,qp_certificate=solved.certificate,qp_certified=solved.success,
                   trust_radius_m=radius,accepted=False)
        events.append(event)
        if not solved.success:
            message='QP failed independent optimality checks';break
        qp_residual=max(0.,float(np.max(lower-a@solved.x)),float(np.max(a@solved.x-upper)))
        event['qp_primal_residual']=qp_residual
        direction=solved.x[:n]*control_scale
        step=float(np.max(np.abs(gmap@direction)))
        event['proposed_lateral_step_m']=step
        certificate=kkt(u)
        event['nonlinear_kkt']=certificate
        event['objective']=f
        if certificate['valid']:
            converged=True;message='SCP feasible KKT point';break
        merit=f/scale+penalty*float(np.maximum(-c/row_scale,0.).max())
        accepted=False
        for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
            candidate=np.clip(u+alpha*direction,bounds[:,0],bounds[:,1])
            cf=float(objective(candidate));cc=np.asarray(constraints(candidate))
            if not (np.isfinite(cf) and np.isfinite(cc).all()):continue
            candidate_merit=cf/scale+penalty*float(np.maximum(-cc/row_scale,0.).max())
            predicted=-(alpha*gradient_value@direction+.5*alpha**2*direction@hessian@direction)+penalty*(float(np.maximum(-c/row_scale,0.).max())-float(np.maximum(-(c+alpha*j@direction)/row_scale,0.).max()))
            ratio=(merit-candidate_merit)/max(predicted,1e-14)
            if predicted>0 and ratio>.05 and candidate_merit<merit-1e-12:
                improvement=merit-candidate_merit
                displacement=candidate-u
                grad_delta=np.asarray(gradient(candidate))/scale-gradient_value
                bs=hessian@displacement
                sbs=float(displacement@bs); sy=float(displacement@grad_delta)
                if sbs>1e-14:
                    theta=1. if sy>=.2*sbs else .8*sbs/(sbs-sy)
                    damped=theta*grad_delta+(1-theta)*bs
                    hessian+=np.outer(damped,damped)/(displacement@damped)-np.outer(bs,bs)/sbs
                    hessian=(hessian+hessian.T)/2
                u,f,c=candidate,cf,cc;accepted=True;stalls=0
                remember(u,f,c)
                event.update(accepted=True,alpha=alpha,merit_decrease=improvement,
                             nonlinear_violation=violation(u,c))
                event.update(predicted_decrease=float(predicted),actual_predicted_ratio=float(ratio))
                if ratio>.75 and alpha==1.:radius=min(2.*config.scp_trust_radius,1.5*radius)
                elif ratio<.25:radius*=.5
                break
        if converged:break
        if not accepted:
            stalls+=1;radius*=.5
            if stalls>=12 or radius<1e-7:
                message='SCP merit stalled';break
    residual=violation(u,c)
    if best is None:
        best=OptimizeResult(x=u,fun=f,success=False,violation=residual,solution_source="failed_iterate")
    best.message=message+('; original constraints feasible' if best.success else '; no feasible iterate')
    # Certify the returned point, not a different/latest or trust-limited iterate.
    best.kkt_certificate=kkt(best.x)
    best.nit=len(events);best.converged=bool(best.success and best.kkt_certificate['valid']);best.qp_events=events
    if best.converged: best.message='SCP feasible KKT point; original constraints feasible' 
    return best
