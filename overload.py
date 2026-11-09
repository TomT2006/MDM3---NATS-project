"""Finite-day overload experiments. Replications are independent whole runs.

A Poisson demand-rate sweep changes demand, not seats, fleet or service rules.
All overload thresholds below are explicit illustrative operating criteria.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from hashlib import sha256
import json
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from model import Simulation, example_config
from results import measurements, audit


@dataclass(frozen=True)
class OverloadRule:
    queue_limit: int = 64                 # Two nominal fleet seat loads (8*4*2).
    persistence_min: int = 30
    growth_window_min: int = 60
    growth_per_hour: float = 16           # Half a nominal fleet seat load/hour.
    sample_min: int = 1

    def validate(self):
        for x in (self.queue_limit,self.persistence_min,self.growth_window_min,self.sample_min):
            if type(x) is not int or x < 1:
                raise ValueError('Positive integer thresholds and grid step required.')
        if self.persistence_min % self.sample_min or self.growth_window_min % self.sample_min:
            raise ValueError('Durations must be multiples of the monitoring step.')
        if not math.isfinite(self.growth_per_hour) or self.growth_per_hour < 0:
            raise ValueError('Growth threshold must be finite and nonnegative.')


def demand_profile(name):
    """Fractions of active demand horizon; all profiles average exactly one."""
    if name == 'constant': return ((0.,1.,1.),)
    if name == 'two_peaks':
        return ((0.,.25,.5),(.25,.5,1.5),(.5,.75,.5),(.75,1.,1.5))
    raise ValueError('Unknown demand profile.')


def passenger_arrivals(config, rate_per_hour, seed, profile='constant'):
    """Independent route Poisson processes via integrated-intensity inversion.

    lambda_ij(t) = rate_per_hour * route_share_ij * profile(t), in persons/hour.
    Same seed preserves the route unit-rate exponential stream across scenarios.
    """
    if not math.isfinite(rate_per_hour) or rate_per_hour < 0:
        raise ValueError('Nonnegative finite demand rate required.')
    weights=config.demand()
    total=sum(weights.values())
    if total <= 0 and rate_per_hour:
        raise ValueError('Positive route weights are required.')
    arrivals=[]
    for i,(route,weight) in enumerate(sorted(weights.items())):
        rate=rate_per_hour*weight/total if total else 0
        if not rate: continue
        parts=demand_profile(profile)
        left=np.array([a*config.demand_stop_min for a,b,w in parts])
        right=np.array([b*config.demand_stop_min for a,b,w in parts])
        rates=np.array([rate*w/60 for a,b,w in parts])
        mass=np.cumsum((right-left)*rates)
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),1729,i]))
        intensity=0.
        while True:
            intensity += float(rng.exponential())
            if intensity >= mass[-1]: break
            j=int(np.searchsorted(mass,intensity,side='right'))
            preceding=mass[j-1] if j else 0
            minute=left[j]+(intensity-preceding)/rates[j]
            arrivals.append((float(minute),*route))
    return sorted(arrivals)


def timetable_ceiling(config):
    """Long-run offered passenger seats/hour, not achievable throughput."""
    return sum(config.seats*60/config.service_intervals_min.get(r,config.timetable_interval_min)
               for r in config.network())


def queue_trace(sim, times_min):
    """Passengers arrived but not departed, including assigned/boarding people.

    Searchsorted measures the right-continuous observed state on a fixed grid.
    Values beyond an early safety stop are NaN, never invented zeros.
    """
    c=sim.config
    result=np.zeros((len(times_min),len(sim.ports)))
    for i,port in enumerate(sim.ports):
        people=[p for p in sim.passengers if p.origin==port]
        arrived=sorted(float(p.arrived*c.time_scale_min) for p in people)
        departed=sorted(float(p.departed*c.time_scale_min) for p in people if p.departed is not None)
        result[:,i]=np.searchsorted(arrived,times_min,side='right')-np.searchsorted(departed,times_min,side='right')
    result[np.asarray(times_min)>float(sim.env.now*c.time_scale_min)+1e-7,:]=np.nan
    return result


def detect_overload(times, queue, rule, demand_stop_min):
    """First grid time: large queue persists AND recent net growth is high.

    Both endpoints belong to the persistence window, i.e. 31 one-minute samples
    span 30 minutes. Detection ends at demand cutoff; draining is not overload.
    """
    rule.validate()
    times=np.asarray(times);queue=np.asarray(queue)
    if len(times)!=len(queue) or len(times)<2 or not np.allclose(np.diff(times),rule.sample_min):
        raise ValueError('A regular matching monitoring grid is required.')
    duration=rule.persistence_min//rule.sample_min
    lag=rule.growth_window_min//rule.sample_min
    streak=0
    for i,(minute,q) in enumerate(zip(times,queue)):
        if minute>demand_stop_min+1e-7 or not np.isfinite(q): break
        streak=streak+1 if q>=rule.queue_limit else 0
        if i>=lag and streak>=duration+1:
            growth=(q-queue[i-lag])*60/rule.growth_window_min
            if growth>=rule.growth_per_hour-1e-9:
                return float(minute)
    return None


def cost_components(sim):
    """Observed finite-horizon waiting, including right-censored passengers."""
    c=sim.config;end=float(sim.env.now*c.time_scale_min)
    waits=[];onboard=[]
    for p in sim.passengers:
        arrived=float(p.arrived*c.time_scale_min)
        departed=float(p.departed*c.time_scale_min) if p.departed is not None else end
        completed=float(p.completed*c.time_scale_min) if p.completed is not None else end
        waits.append(max(0,departed-arrived))
        onboard.append(max(0,completed-departed) if p.departed is not None else 0)
    return dict(total_wait_min=sum(waits),in_transit_min=sum(onboard),
                uncompleted_passengers=sum(p.completed is None for p in sim.passengers),
                flights=sum(f.departed is not None for f in sim.flights),
                aircraft_hours=c.fleet*end/60,
                planned_aircraft_hours=c.fleet*c.horizon_min/60,
                stand_hours=sum(v.stands for v in sim.ports.values())*end/60)


LEGACY_WEIGHTS=dict(passenger_wait_min=1.,uncompleted_passenger=200.,flight=60.,aircraft_hour=50.,stand_hour=0.)


def weighted_score(components, weights=None):
    """Penalty units, not pounds. Stand costs may be supplied explicitly."""
    w={**LEGACY_WEIGHTS,**(weights or {})}
    if set(w)!=set(LEGACY_WEIGHTS) or any(not math.isfinite(x) or x<0 for x in w.values()):
        raise ValueError('Known, finite, nonnegative weights required.')
    return (w['passenger_wait_min']*components['total_wait_min']+
            w['uncompleted_passenger']*components['uncompleted_passengers']+
            w['flight']*components['flights']+w['aircraft_hour']*components['aircraft_hours']+
            w['stand_hour']*components.get('stand_hours',0))


def score_run(sim, rule=None):
    rule=rule or OverloadRule()
    c=sim.config;r=measurements(sim);comp=cost_components(sim)
    times=np.arange(0,c.horizon_min+1e-7,rule.sample_min)
    per_port=queue_trace(sim,times);q=per_port.sum(axis=1)
    event=detect_overload(times,q,rule,c.demand_stop_min)
    end=float(sim.env.now*c.time_scale_min)
    active_end=min(end,c.demand_stop_min)
    last=(times>=max(0,active_end-120))&(times<=active_end)&np.isfinite(q)
    slope=float(np.polyfit(times[last]/60,q[last],1)[0]) if last.sum()>1 else None
    n=r['arrivals']
    row=dict(seed=c.seed,rate=c.total_demand_per_hour,observed_min=end,
        overload=event is not None,overload_time_min=event,
        safety_failure=bool(r['stopped_for_safety']),
        overload_unobserved=event is None and end<c.demand_stop_min-1e-7,
        feasible=not r['stopped_for_safety'],**comp,
        score=weighted_score(comp),arrivals=n,completed=r['completed'],
        served_fraction=r['served_fraction'],timely_fraction=r['timely_fraction'],
        mean_observed_wait_min=comp['total_wait_min']/n if n else None,
        departed_only_mean_wait_min=r['mean_wait_min'],
        departed_only_p95_wait_min=r['p95_wait_min'],
        generalised_time_proxy_min=(2*comp['total_wait_min']+comp['in_transit_min'])/n if n else None,
        monitored_peak_queue=float(np.nanmax(q)),final_backlog=float(q[np.flatnonzero(np.isfinite(q))[-1]]),
        late_growth_per_hour=slope,
        active_departures_per_hour=sum(p.departed is not None and float(p.departed*c.time_scale_min)<active_end
                                      for p in sim.passengers)/(active_end/60),
        minimum_soc=r['minimum_soc'],holding_kwh=r['holding_kwh'],
        admission_deferrals=r['admission_deferrals'])
    return row,times,per_port


def fingerprint(config,rule,profile):
    folder=Path(__file__).resolve().parent
    payload=repr((asdict(config),asdict(rule),profile)).encode()
    for name in ('model.py','safety.py','queues.py','results.py','overload.py'):
        payload+=(folder/name).read_bytes()
    return sha256(payload).hexdigest()


def _trial(job):
    config,rule,profile,seed=job
    c=replace(config,seed=seed)
    arrivals=passenger_arrivals(c,c.total_demand_per_hour,seed,profile)
    sim=Simulation(c,passenger_arrivals=arrivals).run()
    audit(sim)
    row,times,queue=score_run(sim,rule)
    row['profile']=profile
    return row,times,queue


def run_group(config,seeds,folder,rule=None,profile='constant',workers=4):
    """One scenario; save complete replications and explicit provenance."""
    rule=rule or OverloadRule();rule.validate();config.validate()
    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Distinct replication seeds required.')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    signature=fingerprint(config,rule,profile)
    metadata=folder/'metadata.json'
    if metadata.exists():
        previous=json.loads(metadata.read_text())
        if previous['fingerprint']==signature and previous['seeds']==list(seeds):
            return load_group(folder)
        raise ValueError(f'Stale or different scenario at {folder}; use a new output folder.')
    jobs=[(config,rule,profile,seed) for seed in seeds]
    rows=[];traces=[]
    if workers==1:
        outputs=map(_trial,jobs)
        pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=workers)
        outputs=pool.map(_trial,jobs,chunksize=4)
    try:
        for row,times,trace in outputs:
            rows.append(row);traces.append(trace)
            if len(rows)%100==0 or len(rows)==len(seeds):
                print(f'{folder.name}: {len(rows)}/{len(seeds)} audited runs',flush=True)
    finally:
        if pool is not None:pool.shutdown()
    (folder/'runs.json').write_text(json.dumps(rows,indent=1,allow_nan=False)+'\n')
    np.savez_compressed(folder/'traces.npz',times=times,queue=np.asarray(traces,dtype=np.float32),seeds=seeds)
    metadata.write_text(json.dumps(dict(fingerprint=signature,seeds=list(seeds),rule=asdict(rule),
        profile=profile,config_repr=repr(config),rate=config.total_demand_per_hour,
        demand_stop_min=config.demand_stop_min,horizon_min=config.horizon_min,
        timetable_seats_per_hour=timetable_ceiling(config)),indent=2)+'\n')
    return dict(rows=rows,times=times,queue=np.asarray(traces),metadata=json.loads(metadata.read_text()))


def load_group(folder):
    folder=Path(folder)
    with np.load(folder/'traces.npz',allow_pickle=False) as a:
        times,queue,seeds=a['times'],a['queue'],a['seeds']
    rows=json.loads((folder/'runs.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text())
    if list(seeds)!=[r['seed'] for r in rows] or list(seeds)!=meta['seeds']:
        raise ValueError('Misaligned replication records.')
    return dict(rows=rows,times=times,queue=queue,metadata=meta)
