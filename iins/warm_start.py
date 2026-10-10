"""Transport piecewise-constant spatial controls to a receding variable mesh."""
import numpy as np


def remap_controls(old_stations, old_controls, new_stations, tail):
    old=np.asarray(old_stations,float);new=np.asarray(new_stations,float)
    controls=np.asarray(old_controls,float);result=np.asarray(tail,float).copy()
    if controls.shape[0]!=len(old)-1 or result.shape[0]!=len(new)-1:
        raise ValueError('Controls must have one row per spatial interval')
    if np.any(np.diff(old)<=0) or np.any(np.diff(new)<=0):
        raise ValueError('Stations must increase')
    used=False
    for k,(left,right) in enumerate(zip(new[:-1],new[1:])):
        overlap=np.maximum(0.,np.minimum(right,old[1:])-np.maximum(left,old[:-1]))
        length=float(overlap.sum())
        if length>0:
            result[k]=(np.tensordot(overlap,controls,axes=(0,0))+(right-left-length)*result[k])/(right-left)
            used=True
    return result,used
