"""Deterministic, seeded V2X impairment and spatial-to-time trajectory conversion."""
from dataclasses import dataclass
import heapq
import numpy as np


@dataclass(frozen=True)
class TimedTrajectory:
    time: np.ndarray
    s: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    generated_step: int


def to_time_domain(path,dt,generated_step=0):
    if not path.success: raise ValueError('Cannot transmit an infeasible local trajectory')
    t=np.asarray(path.arrival_time)
    if dt<=0 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Arrival times must be finite and strictly increasing')
    times=np.unique(np.r_[np.arange(t[0],t[-1],dt),t[-1]])
    return TimedTrajectory(times,np.interp(times,t,path.s),np.interp(times,t,path.y),
                           np.interp(times,t,path.speed),generated_step)


@dataclass(frozen=True)
class CommunicationConfig:
    dt: float = .05
    packet_loss: float = 0.
    burst_loss: float = 0.
    latency_steps: int = 0
    update_interval: float = .1
    seed: int = 0

    def __post_init__(self):
        if not 0<=self.packet_loss<=1 or not 0<=self.burst_loss<=1:
            raise ValueError('Loss rates are fractions in [0,1]')
        if self.dt<=0 or self.update_interval<=0 or self.latency_steps<0 or int(self.latency_steps)!=self.latency_steps:
            raise ValueError('Invalid communication timing')

    @classmethod
    def seconds(cls,latency=0.,**kwargs):
        dt=kwargs.get('dt',.05)
        return cls(latency_steps=int(np.ceil(latency/dt-1e-10)),**kwargs)


class CommunicationChannel:
    def __init__(self,config=None):
        self.config=config or CommunicationConfig(); self.rng=np.random.default_rng(self.config.seed)
        self.queue=[]; self.serial=0; self.last_generation=-1; self.log=[]

    def exchange(self,step,trajectory):
        """Every step attempts a send; latency is an integer number of steps.

        A successful delivery is only used within its time support. Stale packets
        cannot overwrite a newer plan. No delivery means onboard fallback, even
        if an older plan is cached by the vehicle.
        """
        c=self.config; now=step*c.dt
        draw=float(self.rng.random())
        burst=(now%1.)<c.burst_loss-1e-12
        lost=draw<c.packet_loss or burst
        if trajectory is not None and not lost:
            heapq.heappush(self.queue,(step+c.latency_steps,self.serial,trajectory)); self.serial+=1
        received=None
        while self.queue and self.queue[0][0]<=step:
            _,_,candidate=heapq.heappop(self.queue)
            if candidate.generated_step>=self.last_generation and candidate.time[0]<=now<=candidate.time[-1]:
                received=candidate; self.last_generation=candidate.generated_step
        self.log.append(dict(step=step,time=now,random=draw,lost=bool(lost),burst=bool(burst),
                             delivered=received is not None,fallback=received is None,
                             generated_step=None if received is None else received.generated_step))
        return received


def jitter_latencies(maximum,repetitions,seed):
    """User-specified jitter experiment: repeated constant-latency runs, averaged."""
    if maximum<0 or repetitions<1: raise ValueError('Invalid jitter experiment')
    return np.random.default_rng(seed).uniform(0,maximum,repetitions)


class UpdateScheduler:
    def __init__(self,interval):
        if interval<=0: raise ValueError('interval must be positive')
        self.interval=interval; self.next_time=0.

    def due(self,time):
        if time+1e-10<self.next_time: return False
        self.next_time=(np.floor((time+1e-10)/self.interval)+1)*self.interval
        return True
