"""Shared exact interval integration of the linearized spatial vehicle model.

State [time, lateral, slowness, relative_heading], input [moderation, steering].
Inputs and road curvature are held constant on each spatial interval.
"""
import numpy as np


def spatial_matrices(ds,curvature,wheelbase=2.6):
    if ds<=0 or wheelbase<=0:raise ValueError('Positive spatial step and wheelbase required')
    a=np.eye(4);a[0,2]=ds;a[1,3]=ds
    b=np.zeros((4,2));b[0,0]=.5*ds**2;b[2,0]=ds
    b[1,1]=.5*ds**2/wheelbase;b[3,1]=ds/wheelbase
    c=np.array([0.,-.5*curvature*ds**2,0.,-curvature*ds])
    return a,b,c


def spatial_step(state,control,ds,curvature,wheelbase=2.6):
    a,b,c=spatial_matrices(ds,curvature,wheelbase)
    return a@np.asarray(state)+b@np.asarray(control)+c


def spatial_rollout(stations,controls,curvature,initial_state,wheelbase=2.6):
    ds=np.diff(stations);controls=np.asarray(controls)
    if controls.shape!=(len(ds),2):raise ValueError('Expected one two-component control per interval')
    curvature=np.broadcast_to(curvature,(len(ds),))
    states=np.empty((len(ds)+1,4));states[0]=initial_state
    for k,step in enumerate(ds):states[k+1]=spatial_step(states[k],controls[k],step,curvature[k],wheelbase)
    return states
