"""Grid-level speed and cross-grid travel-time estimation.

Learned future occupancy-flow prediction, physical units, and vehicle arrival exports.
World-coordinate flow prediction on masked, irregular feasible regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator


Array = np.ndarray


@dataclass(frozen=True)
class GridSpec:
    """Cartesian cell lattice with arbitrary feasible mask; s/y names mean world X/Y."""

    s_min: float
    s_max: float
    y_min: float
    y_max: float
    ds: float
    dy: float
    feasible_mask: Array | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.s_max <= self.s_min or self.y_max <= self.y_min:
            raise ValueError("Grid bounds must have positive extent")
        if self.ds <= 0 or self.dy <= 0:
            raise ValueError("Grid resolution must be positive")
        if self.feasible_mask is not None and np.shape(self.feasible_mask)!=self.shape:
            raise ValueError("feasible_mask must match grid shape")

    @property
    def mask(self):
        return np.ones(self.shape,bool) if self.feasible_mask is None else np.asarray(self.feasible_mask,bool)

    @property
    def centers(self):
        return np.stack(np.meshgrid(self.s_centers,self.y_centers,indexing='ij'),-1)

    @classmethod
    def from_road(cls,road,start_s,dx=4.,dy=2.):
        origin=road.world(start_s,0.)
        stations=np.linspace(max(0,start_s-8),min(road.length,start_s+120),129)
        lo,hi=road.bounds(stations)
        bottom=float(min(road.world(stations,lo)[:,1].min(),road.world(stations,hi)[:,1].min())-2.)
        base=cls(float(origin[0]-8),float(origin[0]+120),bottom,bottom+48.,dx,dy)
        return cls(base.s_min,base.s_max,base.y_min,base.y_max,dx,dy,road.feasible(base.centers))

    @classmethod
    def from_polygon(cls,vertices,resolution=(4.,2.)):
        """Rasterize any simple polygon by ray casting; holes can use feasible_mask directly."""
        v=np.asarray(vertices,float)
        base=cls(v[:,0].min(),v[:,0].max(),v[:,1].min(),v[:,1].max(),*resolution)
        x,y=np.moveaxis(base.centers,-1,0); inside=np.zeros(base.shape,bool)
        for a,b in zip(v,np.roll(v,-1,axis=0)):
            if abs(b[1]-a[1])>1e-12:
                inside^=((a[1]>y)!=(b[1]>y))&(x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        return cls(base.s_min,base.s_max,base.y_min,base.y_max,*resolution,inside)

    @property
    def s_centers(self) -> Array:
        count = int(np.ceil((self.s_max - self.s_min) / self.ds - 1e-5))
        return self.s_min + (np.arange(count) + 0.5) * self.ds

    @property
    def y_centers(self) -> Array:
        count = int(np.ceil((self.y_max - self.y_min) / self.dy - 1e-5))
        return self.y_min + (np.arange(count) + 0.5) * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.s_centers), len(self.y_centers)

    def contains(self,points):
        points=np.asarray(points,float)
        index=np.floor((points-[self.s_min,self.y_min])/[self.ds,self.dy]).astype(int)
        i,j=index[...,0],index[...,1]
        within=(i>=0)&(i<self.shape[0])&(j>=0)&(j<self.shape[1])
        return within & self.mask[np.clip(i,0,self.shape[0]-1),np.clip(j,0,self.shape[1]-1)]

    def cell_index(self, s: float, y: float) -> tuple[int, int] | None:
        i = int(np.floor((s - self.s_min) / self.ds - 1e-5))
        j = int(np.floor((y - self.y_min) / self.dy - 1e-5))
        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1] and self.mask[i,j]:
            return i, j
        return None


@dataclass(frozen=True)
class TravelTimeField:
    """Planner-facing road cost predicted for every grid cell."""

    grid: GridSpec
    velocity: Array
    occupancy: Array
    longitudinal_time: Array
    epoch: float = 0.

    def __post_init__(self) -> None:
        expected = self.grid.shape
        if self.velocity.shape != expected + (2,):
            raise ValueError(f"velocity must have shape {expected + (2,)}")
        if self.occupancy.shape != expected:
            raise ValueError(f"occupancy must have shape {expected}")
        if self.longitudinal_time.shape != expected:
            raise ValueError(f"longitudinal_time must have shape {expected}")
        interpolator = RegularGridInterpolator(
            (self.grid.s_centers, self.grid.y_centers),
            self.longitudinal_time,
            bounds_error=False,
            fill_value=None,
        )
        object.__setattr__(self, "_interpolator", interpolator)

    def reference_valid(self,s,y):
        return self.grid.contains(np.stack(np.broadcast_arrays(s,y),axis=-1))

    def bounds(self,s):
        return self.grid.y_min,self.grid.y_max

    def lookup(self, s: Array | float, y: Array | float) -> Array:
        """Interpolate T(s, y), the time required for one longitudinal cell."""

        s_array, y_array = np.broadcast_arrays(
            np.asarray(s, dtype=float), np.asarray(y, dtype=float)
        )
        # Clamp queries to the represented road area. Unbounded extrapolation
        # can otherwise create negative travel-time costs.
        points = np.column_stack(
            (
                np.clip(s_array.ravel(), self.grid.s_centers[0], self.grid.s_centers[-1]),
                np.clip(y_array.ravel(), self.grid.y_centers[0], self.grid.y_centers[-1]),
            )
        )
        values = np.asarray(self._interpolator(points), dtype=float)
        valid=self.grid.contains(np.stack([s_array,y_array],-1)).ravel()
        if self.grid.feasible_mask is not None:
            values=np.where(valid,values,np.inf)
        return np.maximum(values, 1e-6).reshape(s_array.shape)

    def transition_time(
        self, s: float, y: float, next_s: float, next_y: float, min_speed: float = 0.5
    ) -> float:
        """Estimate time along an arbitrary transition using the local velocity."""

        velocity_interp = RegularGridInterpolator(
            (self.grid.s_centers, self.grid.y_centers),
            self.velocity,
            bounds_error=False,
            fill_value=None,
        )
        displacement = np.array([next_s - s, next_y - y], dtype=float)
        distance = float(np.linalg.norm(displacement))
        if distance == 0:
            return 0.0
        points=np.linspace([s,y],[next_s,next_y],max(2,int(np.ceil(distance/min(self.grid.ds,self.grid.dy)))+1))
        if self.grid.feasible_mask is not None and not self.grid.contains(points).all():
            return float('inf')
        direction = displacement / distance
        query = [
            np.clip(s, self.grid.s_centers[0], self.grid.s_centers[-1]),
            np.clip(y, self.grid.y_centers[0], self.grid.y_centers[-1]),
        ]
        local_velocity = np.asarray(velocity_interp([query])[0], dtype=float)
        projected_speed = max(float(local_velocity @ direction), min_speed)
        return distance / projected_speed


def estimate_ego_arrival_times(field, stations, reference_y, *, epoch=None, integration_step=None):
    stations=np.asarray(stations,dtype=float);reference_y=np.asarray(reference_y,dtype=float)
    if stations.ndim!=1 or len(stations)<2 or reference_y.shape!=stations.shape:
        raise ValueError('Reference needs equal one-dimensional station/lateral arrays')
    if not np.isfinite(stations).all() or not np.isfinite(reference_y).all() or np.any(np.diff(stations)<=0):
        raise ValueError('Reference must be finite with increasing stations')
    epoch=float(getattr(field,'epoch',0.) if epoch is None else epoch)
    step=float(min(field.grid.ds,1.) if integration_step is None else integration_step)
    if not np.isfinite([epoch,step,field.grid.ds]).all() or min(step,field.grid.ds)<=0:
        raise ValueError('Invalid epoch, integration step or field cell length')
    out=np.full(len(stations),np.inf);out[0]=epoch
    for k,(a,b) in enumerate(zip(stations[:-1],stations[1:])):
        edges=np.linspace(a,b,int(np.ceil((b-a)/step))+1)
        middle=(edges[:-1]+edges[1:])*.5
        # Check both endpoints and quadrature nodes, including a blocked start.
        query=np.sort(np.r_[edges,middle]);lateral=np.interp(query,stations,reference_y)
        low,high=field.bounds(query)
        valid=(lateral>=low)&(lateral<=high)
        if hasattr(field,'reference_valid'):valid &= field.reference_valid(query,lateral)
        if not np.all(valid):break
        costs=np.asarray(field.lookup(middle,np.interp(middle,stations,reference_y)),dtype=float)
        if np.isnan(costs).any() or (costs<=0).any():raise ValueError('Traversal times must be positive or +inf')
        if not np.isfinite(costs).all():break
        out[k+1]=out[k]+float(np.sum(costs*np.diff(edges)/field.grid.ds))
    return out


@dataclass(frozen=True)
class VehicleArrival:
    """One vehicle's absolute arrival times and lateral centers at spatial steps.

    inf means the station is behind it or not reached in the prediction budget.
    All consumers must use the same road coordinates and time origin.
    """
    vehicle_id: str
    stations: Array
    times: Array
    lateral: Array
    half_width: float = 1.0

    def __post_init__(self):
        for name in ('stations','times','lateral'):
            object.__setattr__(self,name,np.asarray(getattr(self,name),dtype=float))
        if self.times.shape != self.stations.shape or self.lateral.shape != self.stations.shape:
            raise ValueError('Arrival arrays must have equal shapes')
        if np.any(np.diff(self.stations)<=0) or np.any(np.isnan(self.times)):
            raise ValueError('Stations must increase and times cannot be NaN')


@dataclass(frozen=True)
class FlowForecast:
    grid: GridSpec
    velocity: Array  # [H,W,Tf,2], m/s
    occupancy: Array  # [H,W,Tf], probabilities
    dt: float = .5
    epoch: float = 0.

    def field(self, future_index=0):
        v=self.velocity[:,:,future_index]
        return TravelTimeField(self.grid,v,self.occupancy[:,:,future_index],
                               self.grid.ds/np.maximum(v[...,0],.5),epoch=self.epoch)

    def vehicle_arrivals(self, vehicles, stations, max_duration=120., *, road=None, start_s=0.):
        if road is not None:
            return RoadTravelTimeField(self,road,start_s).vehicle_arrivals(vehicles,stations,max_duration)
        stations=np.asarray(stations,dtype=float); results=[]
        interps=[RegularGridInterpolator((self.grid.s_centers,self.grid.y_centers),self.velocity[:,:,k],bounds_error=False,fill_value=None) for k in range(self.velocity.shape[2])]
        for vehicle in vehicles:
            s=float(vehicle['s']); y=float(vehicle['y']); now=self.epoch
            out=np.full(stations.shape,np.inf); lat=np.full(stations.shape,y)
            if vehicle.get('stopped',False):
                # Static occupancy needs geometric avoidance; never claim infinite time means safe.
                results.append(VehicleArrival(str(vehicle['id']),stations,out,lat,vehicle.get('half_width',1.)))
                continue
            for _ in range(int(max_duration/.1)):
                query=[np.clip(s,self.grid.s_centers[0],self.grid.s_centers[-1]),np.clip(y,self.grid.y_centers[0],self.grid.y_centers[-1])]
                k=min(int((now-self.epoch)/self.dt),len(interps)-1)
                v=interps[k]([query])[0]; next_s=s+max(float(v[0]),0)*.1; next_y=y+float(v[1])*.1
                crossed=(stations>=s)&(stations<=next_s)&~np.isfinite(out)
                fraction=(stations[crossed]-s)/max(next_s-s,1e-12)
                out[crossed]=now+.1*fraction; lat[crossed]=y+(next_y-y)*fraction
                s,y,now=next_s,next_y,now+.1
                if s>=stations[-1]: break
            results.append(VehicleArrival(str(vehicle['id']),stations,out,lat,vehicle.get('half_width',1.)))
        return results


class GridTravelTimeEstimator:
    """Checkpoint-backed predictor; never fills unobserved cells with fixed speed."""
    def __init__(self,grid,checkpoint=None):
        from pathlib import Path
        from .learn.training import load_model
        self.grid=grid
        self.checkpoint=Path(checkpoint) if checkpoint else Path(__file__).parent/'learn/weights/latent_speed_bce.pt'
        self.model=load_model(self.checkpoint)

    def predict(self,occupancy,trajectories,agent_mask=None,epoch=0.):
        import torch
        o=np.asarray(occupancy,dtype=np.float32); t=np.asarray(trajectories,dtype=np.float32)
        if o.shape != self.grid.shape+(7,) or t.ndim!=3 or t.shape[1:]!=(7,3):
            raise ValueError('Expected occupancy [H,W,7], trajectories [N,7,3]=[world_x,world_y,speed]')
        if not np.isfinite(o).all() or not np.isfinite(t).all(): raise ValueError('Nonfinite input')
        a=np.ones(len(t),bool) if agent_mask is None else np.asarray(agent_mask,bool)
        if a.shape!=(len(t),):raise ValueError('agent_mask must have shape [N]')
        if not self.grid.mask.any():raise ValueError('Grid has no feasible cells')
        if len(t)==0:t=np.zeros((1,7,3),np.float32);a=np.zeros(1,bool)
        o=o*self.grid.mask[...,None]
        with torch.no_grad():
            v,p,_=self.model(torch.from_numpy(o)[None],torch.from_numpy(t)[None],torch.from_numpy(a)[None],
                            torch.from_numpy(self.grid.centers.astype(np.float32))[None],torch.from_numpy(self.grid.mask)[None])
        return FlowForecast(self.grid,v[0].numpy(),p[0].sigmoid().numpy()*self.grid.mask[...,None],epoch=epoch)

    def fit(self,trajectories):
        """Compatibility inference from [time,s,y] histories; training uses CLI train."""
        samples=list(trajectories)
        if not samples: raise ValueError('Need trajectory histories for inference')
        end=max(float(x[-1,0]) for x in samples); times=np.linspace(end-3,end,7)
        hist=[]; occupancy=np.zeros(self.grid.shape+(7,),np.float32)
        for sample in samples:
            sample=np.asarray(sample)
            if len(sample)<2 or np.any(np.diff(sample[:,0])<=0): raise ValueError('Need increasing timestamps')
            pos=np.column_stack([np.interp(times,sample[:,0],sample[:,j]) for j in (1,2)])
            speed=np.linalg.norm(np.gradient(pos,.5,axis=0),axis=1)
            hist.append(np.column_stack([pos,speed]))
            for k,(s,y) in enumerate(pos):
                ij=self.grid.cell_index(s,y)
                if ij is not None: occupancy[ij+(k,)]=1
        return self.predict(occupancy,np.asarray(hist)).field()


def main():
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('generate', help='Export an existing bundled split; no random generation'); p.add_argument('--output',required=True); p.add_argument('--split',choices=('train','test'),default='train')
    p.add_argument('--dataset',choices=('etc_derived',),default='etc_derived')
    p=sub.add_parser('train'); p.add_argument('--train-data',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--latent-dim',type=int,default=0)
    p.add_argument('--kl-weight',type=float,default=.001)
    p.add_argument('--kl-warmup-epochs',type=int,default=5)
    p.add_argument('--reconstruction',choices=('legacy','normalized-speed-bce'),default='normalized-speed-bce')
    p.add_argument('--speed-scale-mps',type=float,default=20.)
    p=sub.add_parser('evaluate'); p.add_argument('--data',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output')
    a=parser.parse_args()
    if a.command=='generate':
        from .learn.data import generate
        generate(a.output, split=a.split, dataset=a.dataset)
    elif a.command=='train':
        from .learn.training import train
        train(a.train_data,a.checkpoint,a.epochs,a.seed,a.latent_dim,a.kl_weight,a.kl_warmup_epochs,a.reconstruction,a.speed_scale_mps)
    else:
        from .learn.training import evaluate
        result=evaluate(a.checkpoint,a.data); report=json.dumps(result,indent=2); print(report)
        if a.output:
            Path(a.output).parent.mkdir(parents=True, exist_ok=True)
            Path(a.output).write_text(report+'\n')


class RoadTravelTimeField:
    """Geometry adapter: spatial planner queries a world-coordinate learned flow."""
    def __init__(self,forecast,road,start_s=0.,margin=1.):
        from scipy.spatial import cKDTree
        self.forecast=forecast;self.road=road;self.start_s=start_s;self.margin=margin
        self.grid=forecast.grid
        self.points=self.grid.centers[self.grid.mask]
        self.tree=cKDTree(self.points)
        self.values=forecast.velocity[self.grid.mask]

    @property
    def epoch(self):
        return self.forecast.epoch

    def reference_valid(self,s,y):
        return self.grid.contains(self.road.world(np.asarray(s)+self.start_s,y))

    def world_velocity(self,xy,k=0):
        distance,indices=self.tree.query(xy,k=min(4,len(self.points)))
        weights=1/np.maximum(distance,.1)**2
        return np.sum(self.values[indices,k]*weights[...,None],axis=-2)/weights.sum(axis=-1,keepdims=True)

    def bounds(self,s):
        return self.road.bounds(np.asarray(s)+self.start_s,self.margin)

    def lookup(self,s,y):
        absolute=np.asarray(s)+self.start_s;xy=self.road.world(absolute,y)
        v=self.world_velocity(xy);forward=np.sum(v*self.road.tangent(absolute),axis=-1)
        return self.grid.ds/np.maximum(forward,.5)

    def vehicle_arrivals(self,vehicles,stations,max_duration=25.):
        stations=np.asarray(stations);results=[]
        for actor in vehicles:
            s=float(actor['s']);lat=float(actor['y']);now=self.forecast.epoch
            out=np.full(stations.shape,np.inf);lateral=np.full(stations.shape,lat)
            if not actor.get('stopped',False):
                xy=self.road.world(s+self.start_s,lat)
                for _ in range(int(max_duration/.1)):
                    k=min(int((now-self.forecast.epoch)/self.forecast.dt),self.forecast.velocity.shape[2]-1)
                    v=self.world_velocity(xy,k);next_xy=xy+.1*v
                    ns,nl=self.road.project(next_xy);ns=float(ns)-self.start_s;nl=float(nl)
                    low,high=self.road.bounds(ns+self.start_s,actor.get('half_width',1.));nl=float(np.clip(nl,low,high))
                    crossed=(stations>=s)&(stations<=ns)&~np.isfinite(out)
                    fraction=(stations[crossed]-s)/max(ns-s,1e-9)
                    out[crossed]=now+.1*fraction;lateral[crossed]=lat+(nl-lat)*fraction
                    s,lat,now=ns,nl,now+.1;xy=self.road.world(s+self.start_s,lat)
                    if s>=stations[-1]: break
            results.append(VehicleArrival(str(actor['id']),stations,out,lateral,actor.get('half_width',1.)))
        return results


if __name__=='__main__': main()
