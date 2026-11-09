from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from .certified_qp import CertifiedQPSolver


@dataclass
class QuadraticConstraint:
    # x.T Q x + a.T x + b >= 0
    Q: np.ndarray
    a: np.ndarray
    b: float

    def value(self,x):return float(x@self.Q@x+self.a@x+self.b)


def build_cbf_quadratics(base,gmap,stations,obstacle_data,gamma):
    """Exact expansion of the existing discrete differential CBF in controls."""
    rows=[];ds=np.diff(stations)
    for obstacle,oy,active in obstacle_data:
        offset=base-oy
        for k in np.flatnonzero(active[:-1]&active[1:]):
            c0=gamma-1./ds[k];c1=1./ds[k]
            q=c0*np.outer(gmap[k],gmap[k])+c1*np.outer(gmap[k+1],gmap[k+1])
            a=2*(c0*offset[k]*gmap[k]+c1*offset[k+1]*gmap[k+1])
            b=c0*offset[k]**2+c1*offset[k+1]**2-gamma*obstacle.safe_squared_distance
            rows.append(QuadraticConstraint(q,a,float(b)))
    return rows


def is_convex_qcqp(hessian,rows,tol=1e-10):
    # Objective >=0 Hessian; >=0 constraints must have NEGATIVE semidefinite Q.
    return bool(np.linalg.eigvalsh(hessian).min()>=-tol and
                all(np.linalg.eigvalsh(r.Q).max()<=tol for r in rows))


def solve_qcqp(hessian,linear,constant,rows,matrix,lower,upper,bounds,initial,
               *,time_limit=1.,node_limit=10000,gap=1e-4,tolerance=1e-6,fast_start=False):
    """Minimize .5 x'Hx + linear'x + constant with hard quadratic rows."""
    started=time.perf_counter();hessian=(hessian+hessian.T)*.5
    convex=is_convex_qcqp(hessian,rows)
    def violation(x):
        if x is None or not np.isfinite(x).all():return float('inf')
        ax=matrix@x
        return max(0.,float(np.max(lower-ax,initial=-np.inf)),float(np.max(ax-upper,initial=-np.inf)),
                   float(np.max(bounds[:,0]-x)),float(np.max(x-bounds[:,1])),
                   max((-r.value(x) for r in rows),default=0.))
    if not rows and convex:
        a=sparse.vstack([sparse.csc_matrix(matrix),sparse.eye(len(linear))],format='csc')
        solved=CertifiedQPSolver(20000).solve(sparse.csc_matrix(np.triu(hessian)),linear,a,
            np.r_[lower,bounds[:,0]],np.r_[upper,bounds[:,1]],initial)
        x=solved.x;valid=bool(solved.success and violation(x)<=tolerance)
        return x,dict(backend=solved.backend,classification='convex_qp',status=solved.status,
            feasible=valid,subproblem_optimal=bool(solved.success),nodes_processed=0,branch_events=0,
            lower_bound=None,upper_bound=float(.5*x@hessian@x+linear@x+constant) if valid else None,
            relative_gap=None,violation=violation(x),elapsed_seconds=time.perf_counter()-started)
    try:
        from pyscipopt import Model,quicksum,Eventhdlr,SCIP_EVENTTYPE,SCIP_PARAMSETTING
    except ImportError as exc:
        raise ImportError('qcqp_bnb requires pyscipopt; install the project dependencies') from exc
    class NodeEvents(Eventhdlr):
        def __init__(self):self.branched=0;self.infeasible=0;self.feasible=0
        def eventinit(self):
            for event in (SCIP_EVENTTYPE.NODEBRANCHED,SCIP_EVENTTYPE.NODEINFEASIBLE,SCIP_EVENTTYPE.NODEFEASIBLE):self.model.catchEvent(event,self)
        def eventexec(self,event):
            kind=event.getType()
            if kind==SCIP_EVENTTYPE.NODEBRANCHED:self.branched+=1
            elif kind==SCIP_EVENTTYPE.NODEINFEASIBLE:self.infeasible+=1
            elif kind==SCIP_EVENTTYPE.NODEFEASIBLE:self.feasible+=1
    model=Model('IINS quadratic CBF');model.hideOutput()
    model.setRealParam('limits/time',max(.001,time_limit))
    model.setLongintParam('limits/nodes',node_limit)
    model.setRealParam('limits/gap',gap)
    model.setRealParam('numerics/feastol',min(tolerance*.1,1e-7))
    model.setIntParam('parallel/maxnthreads',1);model.setIntParam('randomization/randomseedshift',0)
    if fast_start:
        # An externally verified SQP incumbent already supplies the primal start.
        # Limit root overhead; SCIP's valid relaxations/enforcement/branching remain.
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
        model.setBoolParam('nlp/disable',True)
        model.setIntParam('presolving/maxrounds',1)
        model.setIntParam('separating/maxroundsroot',2)
        model.setIntParam('propagating/maxroundsroot',2)
    events=NodeEvents();model.includeEventhdlr(events,'node_events','Actual SCIP node events')
    x=[model.addVar(f'u{j}',lb=float(lo),ub=float(hi)) for j,(lo,hi) in enumerate(bounds)]
    def expr(q,a,b):
        return (quicksum(float(q[i,j])*(1 if i==j else 2)*x[i]*x[j]
                         for i in range(len(x)) for j in range(i,len(x)) if q[i,j]!=0)
                +quicksum(float(v)*x[i] for i,v in enumerate(a) if v!=0)+float(b))
    for i,row in enumerate(rows):model.addCons(expr(row.Q,row.a,row.b)>=0,name=f'cbf_{i}')
    for i,(a,lo,hi) in enumerate(zip(matrix,lower,upper)):
        e=quicksum(float(v)*x[j] for j,v in enumerate(a) if v!=0)
        if np.isfinite(lo):model.addCons(e>=float(lo),name=f'linear_lower_{i}')
        if np.isfinite(hi):model.addCons(e<=float(hi),name=f'linear_upper_{i}')
    epigraph=model.addVar('objective',lb=-model.infinity())
    model.addCons(expr(hessian*.5,linear,constant)<=epigraph)
    model.setObjective(epigraph,'minimize')
    if violation(initial)<=tolerance:
        seed=model.createSol()
        for var,value in zip(x,initial):model.setSolVal(seed,var,float(value))
        model.setSolVal(seed,epigraph,float(.5*initial@hessian@initial+linear@initial+constant))
        model.addSol(seed)
    model.optimize()
    status=str(model.getStatus());solution=model.getBestSol()
    candidate=np.asarray([model.getSolVal(solution,v) for v in x]) if solution is not None else None
    valid=violation(candidate)<=tolerance
    def finite_bound(value):return float(value) if np.isfinite(value) and abs(value)<model.infinity() else None
    report=dict(backend='scip',fast_start=fast_start,classification='convex_qcqp' if convex else 'nonconvex_qcqp',
        status=status,feasible=bool(valid),subproblem_optimal=status=='optimal',
        nodes_processed=int(model.getNNodes()),branch_events=events.branched,
        node_infeasible_events=events.infeasible,node_feasible_events=events.feasible,
        lower_bound=finite_bound(model.getDualbound()),upper_bound=finite_bound(model.getPrimalbound()),
        relative_gap=finite_bound(model.getGap()),violation=violation(candidate),
        elapsed_seconds=time.perf_counter()-started,
        certificate_scope='Current quadratic subproblem over supplied control bounds, not the original nonlinear objective')
    model.freeProb()
    return candidate,report
