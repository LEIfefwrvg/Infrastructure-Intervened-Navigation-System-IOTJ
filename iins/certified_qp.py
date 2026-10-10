"""Checked convex-QP solves with an independently verified fallback.

Both backends solve the identical objective and constraints. No penalty,
constraint relaxation, or extra Hessian regularization is added here.
"""
import time
from dataclasses import dataclass
import numpy as np
from scipy import sparse
import osqp
import clarabel


def symmetric_hessian(p):
    upper = sparse.triu(p, format='csc')
    return upper + upper.T - sparse.diags(upper.diagonal(), format='csc')


def qp_certificate(p, q, a, lower, upper, x, y, tolerance=1e-5):
    """Check original-coordinate primal, stationarity and normal-cone residuals.

    OSQP signs: y>0 belongs to an upper bound; y<0 to a lower bound.
    Complementarity is the sum of absolute products, so errors cannot cancel.
    Relative dual/complementarity checks supplement an absolute primal check.
    """
    invalid = dict(valid=False, primal_residual=None, dual_residual=None,
                   dual_relative=None, complementarity_relative=None, dual_sign_residual=None)
    if x is None or y is None:
        return invalid
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != q.shape or y.shape != lower.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return invalid
    px = symmetric_hessian(p) @ x
    ax = a @ x
    aty = a.T @ y
    primal = max(0., float(np.max(lower-ax)), float(np.max(ax-upper)))
    dual = float(np.max(np.abs(px+q+aty)))
    dual_scale = max(1., float(np.max(np.abs(px))), float(np.max(np.abs(q))), float(np.max(np.abs(aty))))
    positive, negative = np.maximum(y, 0.), np.maximum(-y, 0.)
    finite_upper, finite_lower = np.isfinite(upper), np.isfinite(lower)
    sign_error = max(float(np.max(positive[~finite_upper], initial=0.)),
                     float(np.max(negative[~finite_lower], initial=0.)))
    complementarity = float(np.sum(np.abs(positive[finite_upper]*(upper[finite_upper]-ax[finite_upper]))) +
                            np.sum(np.abs(negative[finite_lower]*(ax[finite_lower]-lower[finite_lower]))))
    objective = float(.5*x@px+q@x)
    comp_relative = complementarity/max(1., abs(objective))
    valid = primal <= tolerance and dual/dual_scale <= tolerance and comp_relative <= tolerance and sign_error <= tolerance
    return dict(valid=bool(valid), primal_residual=primal, dual_residual=dual,
                dual_relative=dual/dual_scale, complementarity_relative=comp_relative,
                dual_sign_residual=sign_error)


@dataclass
class CheckedQPResult:
    x: object
    y: object
    success: bool
    backend: str
    status: str
    attempts: list
    certificate: dict
    setup_time_s: float
    solve_time_s: float


class CertifiedQPSolver:
    """OSQP with deterministic rho adaptation, then Clarabel if uncertified."""
    def __init__(self, max_iterations=4000, tolerance=1e-5):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.osqp = None
        self.pattern = None

    def solve(self, p, q, a, lower, upper, initial):
        p, a = sparse.triu(p, format='csc'), sparse.csc_matrix(a)
        q, lower, upper = map(lambda v: np.asarray(v, dtype=float), (q, lower, upper))
        attempts = []
        setup_total = solve_total = 0.
        started = time.perf_counter()
        pattern = (p.shape, p.indices.tobytes(), p.indptr.tobytes(),
                   a.shape, a.indices.tobytes(), a.indptr.tobytes())
        if self.osqp is None or pattern != self.pattern:
            self.osqp = osqp.OSQP()
            self.osqp.setup(P=p, q=q, A=a, l=lower, u=upper, verbose=False,
                eps_abs=1e-6, eps_rel=1e-6, max_iter=self.max_iterations,
                polishing=True, adaptive_rho=True, adaptive_rho_interval=50, rho=.1)
            self.pattern = pattern
        else:
            self.osqp.update(Px=p.data, Ax=a.data, q=q, l=lower, u=upper)
        self.osqp.warm_start(x=initial)
        setup_total += time.perf_counter()-started
        started = time.perf_counter()
        result = self.osqp.solve(raise_error=False)
        elapsed = time.perf_counter()-started
        solve_total += elapsed
        cert = qp_certificate(p, q, a, lower, upper, result.x, result.y, self.tolerance)
        accepted = result.info.status_val in (1, 2) and cert['valid']
        attempts.append(dict(backend='osqp', status=str(result.info.status), iterations=int(result.info.iter),
            solve_time_s=elapsed, accepted=bool(accepted), certificate=cert))
        if accepted:
            return CheckedQPResult(result.x, result.y, True, 'osqp', str(result.info.status),
                                   attempts, cert, setup_total, solve_total)

        # Exact conversion: equality rows use a zero cone; finite upper/lower
        # bounds become A*x <= u and -A*x <= -l, respectively.
        started = time.perf_counter()
        eq = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
        hi = np.isfinite(upper) & ~eq
        lo = np.isfinite(lower) & ~eq
        matrix = sparse.vstack([a[eq], a[hi], -a[lo]], format='csc')
        rhs = np.r_[upper[eq], upper[hi], -lower[lo]]
        cones = []
        if eq.any(): cones.append(clarabel.ZeroConeT(int(eq.sum())))
        if hi.any() or lo.any(): cones.append(clarabel.NonnegativeConeT(int(hi.sum()+lo.sum())))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 200
        settings.max_threads = 1
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
        fallback = clarabel.DefaultSolver(p, q, matrix, rhs, cones, settings)
        setup_total += time.perf_counter()-started
        started = time.perf_counter()
        solved = fallback.solve()
        elapsed = time.perf_counter()-started
        solve_total += elapsed
        x = np.asarray(solved.x)
        z = np.asarray(solved.z)
        dual = np.zeros(len(lower))
        ne, nh = int(eq.sum()), int(hi.sum())
        dual[eq] = z[:ne]
        dual[hi] += z[ne:ne+nh]
        dual[lo] -= z[ne+nh:]
        cert = qp_certificate(p, q, a, lower, upper, x, dual, self.tolerance)
        accepted = str(solved.status) in ('Solved', 'AlmostSolved') and cert['valid']
        attempts.append(dict(backend='clarabel', status=str(solved.status), iterations=int(solved.iterations),
            solve_time_s=elapsed, accepted=bool(accepted), certificate=cert))
        if accepted:
            # Do not retain the failed OSQP dual iterate for the next SCP model.
            self.osqp.warm_start(x=x, y=dual)
        return CheckedQPResult(x if accepted else None, dual if accepted else None, bool(accepted),
                               'clarabel', str(solved.status), attempts, cert, setup_total, solve_total)
