import numpy as np

RISK_THRESHOLD_MPS2 = 1.1


def risk_metrics(acceleration,threshold,dt=None,*,severity_mode="above"):
    if severity_mode not in ("above", "below"):
        raise ValueError("severity_mode must be above or below")
    a=np.asarray(acceleration,float)
    if not np.isfinite(threshold) or threshold<0 or not np.isfinite(a).all(): raise ValueError('Invalid risk input')
    weights=np.ones(a.shape) if dt is None else np.broadcast_to(np.asarray(dt,float),a.shape)
    if not np.isfinite(weights).all() or np.any(weights<=0): raise ValueError('Durations must be positive')
    decel=a<0; exposed=decel&(-a>threshold)
    severity_samples=exposed if severity_mode=="above" else decel&(-a<threshold)
    denominator=float(weights[decel].sum()); numerator=float(weights[exposed].sum())
    return dict(risk_exposure_fraction=numerator/denominator if denominator else None,
                risk_exposure_percent=100*numerator/denominator if denominator else None,
                risk_severity_mps2=float(np.mean(-a[severity_samples]) if severity_mode=="above" else np.average(-a[severity_samples],weights=weights[severity_samples])) if severity_samples.any() else None,
                deceleration_duration=denominator,risk_duration=numerator,threshold_mps2=threshold,
                severity_mode=severity_mode,severity_weighting="samples" if severity_mode=="above" else "duration",
                severity_inequality='0 < -acceleration '+('>' if severity_mode=='above' else '<')+' threshold',
                inequality='acceleration < 0 and -acceleration > threshold',
                denominator='all deceleration time; not all driving time')


def travel_time(times,stations,entry,exit):
    t=np.asarray(times,float); s=np.asarray(stations,float)
    if len(t)!=len(s) or np.any(np.diff(t)<=0) or np.any(np.diff(s)<0): raise ValueError('Invalid trace')
    if entry>=exit or s[0]>entry or s[-1]<exit: return None
    return float(np.interp(exit,s,t)-np.interp(entry,s,t))


def behavior_variation(acceleration):
    a=np.asarray(acceleration,float); mean=abs(float(a.mean()))
    return float(a.std()/mean) if mean>1e-9 else None


def computation_times(samples):
    x=np.asarray(samples,float)
    return dict(calls=len(x),mean_s=float(x.mean()) if len(x) else None,
                p95_s=float(np.percentile(x,95)) if len(x) else None,
                max_s=float(x.max()) if len(x) else None,
                all_below_005s=bool(np.all(x<.05)) if len(x) else None)
