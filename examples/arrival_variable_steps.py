"""Runnable estimator -> ego ETA -> variable-step planner example.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iins.travel_time import GridSpec,GridTravelTimeEstimator
from iins.global_planner import SpatialGlobalPlanner,GlobalPlannerConfig,ObstacleProfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--solver',choices=['psd_dispatch','qcqp_bnb','scp','slsqp'],default='psd_dispatch')
    args=parser.parse_args();torch.set_num_threads(2)
    base=GridSpec(-8.,120.,-24.,24.,4.,2.)
    grid=GridSpec(-8.,120.,-24.,24.,4.,2.,abs(base.centers[...,1])<7.)
    times=np.arange(-3.,.01,.5)
    histories=np.zeros((2,7,3),dtype=np.float32)
    histories[0,:,0]=4*times;histories[0,:,2]=4.
    histories[1,:,0]=8.+4*times;histories[1,:,1]=3.5;histories[1,:,2]=4.
    occupancy=np.zeros(grid.shape+(7,),dtype=np.float32)
    for vehicle in histories:
        for k,(x,y,_) in enumerate(vehicle):
            index=grid.cell_index(x,y)
            if index is not None:occupancy[index+(k,)]=1.
    estimator=GridTravelTimeEstimator(grid,args.checkpoint)
    forecast=estimator.predict(occupancy,histories,epoch=100.)
    field=forecast.field(future_index=0)
    config=GlobalPlannerConfig(solver=args.solver,horizon=30,ds=1.,control_step=3.,coarse_factor=5,variable_steps=True)
    planner=SpatialGlobalPlanner(field,config)
    stations=np.arange(config.horizon+1)*config.ds
    neighbors=forecast.vehicle_arrivals([dict(id='neighbor',s=8.,y=3.5)],stations,max_duration=30.)
    obstacles=[]
    for vehicle in neighbors:
        valid=np.isfinite(vehicle.times)
        if valid.sum()>=2:
            obstacles.append(ObstacleProfile(vehicle.stations[valid],vehicle.lateral[valid],safe_squared_distance=1.))
    # Planner derives ego ETAs from field.lookup along the initial heading reference.
    path=planner.plan(start_s=0.,initial_y=0.,initial_heading=0.,vehicle_arrivals=neighbors,obstacles=obstacles)
    def finite_list(a):return [float(x) if np.isfinite(x) else None for x in a]
    report=dict(purpose='Illustrative end-to-end integration smoke example, not a paper benchmark',
        checkpoint=str(estimator.checkpoint),arrival_time_source=planner.arrival_time_source,
        epoch=forecast.epoch,full_grid_stations=stations.tolist(),ego_arrival_times=finite_list(planner.last_ego_arrival_times),
        surrounding_arrival_times={v.vehicle_id:finite_list(v.times) for v in neighbors},
        variable_steps=config.variable_steps,k_star=planner.k_star,retained_indices=planner.control_indices.tolist(),
        full_nodes=len(stations),retained_nodes=len(path.s),planner_success=bool(path.success),planner_message=path.message,
        constraint_count=planner.last_diagnostics[-1]['constraints'])
    print(json.dumps(report,indent=2,allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not path.success:raise SystemExit('Planner failed; see report')
    if not np.isfinite(planner.last_ego_arrival_times).all():raise SystemExit('Reference coverage failed; see report')

if __name__=='__main__':main()
