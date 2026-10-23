"""Repeat scenarios with independent seeds; save auditable, reproducible results."""
from dataclasses import asdict, replace
from pathlib import Path
from hashlib import sha256
import json
import math
import numpy as np
from scipy.stats import t as student_t
from model import Simulation, example_config
from results import measurements, audit
from mm1k import theory, simulate


def interval(values):
    """Mean and 95% Student-t CI across independent runs, not individual people."""
    values = np.asarray(values,dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    half = float(student_t.ppf(.975,n-1)*np.std(values,ddof=1)/math.sqrt(n)) if n > 1 else 0
    return dict(mean=mean,low=mean-half,high=mean+half,n=n)


def serialise(value):
    if isinstance(value,dict):
        return {('->'.join(k) if isinstance(k,tuple) else str(k)):serialise(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serialise(v) for v in value]
    return value


def fingerprint(config):
    """Changing configuration OR model code invalidates the saved results."""
    folder=Path(__file__).resolve().parent
    payload=json.dumps(serialise(asdict(config)),sort_keys=True).encode()
    for filename in ('model.py','queues.py','results.py','mm1k.py','experiments.py'):
        payload += (folder/filename).read_bytes()
    return sha256(payload).hexdigest()


def capacity_summary(rows,config):
    """A tested level passes only if EVERY replication meets both targets."""
    answer={}
    for group in ('demand','fleet'):
        levels=sorted({r['level'] for r in rows if r['group']==group})
        passing=[level for level in levels if all(r['metrics']['passes_targets']
                  for r in rows if r['group']==group and r['level']==level)]
        answer[group]=dict(passing_levels=passing,
                          best_tested=(max(passing) if group=='demand' else min(passing)) if passing else None)
    return answer


def run_all(config=None, seeds=(11,22,33,44,55), benchmark_reps=6,
            demand_levels=(12,24,36,48,60,72), fleet_levels=(2,3,4,5,6,8),
            benchmark_duration=12000, benchmark_warmup=1200):
    """Finite-day demand and fleet studies. Seats and timetable stay fixed.

    Different fleet sizes share identical demand streams for each seed. These
    results measure this policy/timetable; they are not a global capacity bound.
    """
    config=config or example_config()
    config.validate()
    if config.demand_per_hour is not None:
        base_total=sum(config.demand_per_hour.values())
        if base_total <= 0: raise ValueError('Demand sweep needs positive route weights.')
    else:
        base_total=config.total_demand_per_hour
    if not seeds or len(set(seeds)) != len(seeds) or benchmark_reps < 2:
        raise ValueError('Use distinct scenario seeds and at least two benchmark replications.')
    rows=[]
    scenarios=[]
    for rate in demand_levels:
        rates=None if config.demand_per_hour is None else {r:v*rate/base_total for r,v in config.demand_per_hour.items()}
        scenarios.append(('demand',rate,replace(config,total_demand_per_hour=rate,demand_per_hour=rates)))
    for fleet in fleet_levels:
        # Automatic redistribution is explicit; custom initial placement is cleared.
        scenarios.append(('fleet',fleet,replace(config,fleet=fleet,initial_counts=None)))
    for K in (k for k in (1,2,4,8) if k >= max(config.capacities(p)['pads'] for p in config.port_names)):
        overrides={p:{**v,'K':K} for p,v in config.port_overrides.items()}
        scenarios.append(('K',K,replace(config,queue_capacity=K,port_overrides=overrides)))
    for delay in (0,5):
        scenarios.append(('delay',delay,replace(config,ground_delay_mean_min=delay)))
    for group,level,setting in scenarios:
        for seed in seeds:
            sim=Simulation(replace(setting,seed=seed)).run()
            audit(sim)
            rows.append(dict(group=group,level=level,seed=seed,metrics=measurements(sim)))
        print(f'{group}={level}: {len(seeds)} network runs checked',flush=True)
    benchmark=[]
    for rho in (.5,1,1.5):
        for K in (1,4,8):
            for seed in range(1,benchmark_reps+1):
                benchmark.append(dict(rho=rho,K=K,seed=seed,
                    metrics=simulate(rho,1,K,benchmark_duration,benchmark_warmup,seed)))
        print(f'M/M/1/K rho={rho}: replications complete',flush=True)
    return dict(fingerprint=fingerprint(config),config=serialise(asdict(config)),seeds=list(seeds),
                benchmark_reps=benchmark_reps,benchmark_duration_min=benchmark_duration,
                benchmark_warmup_min=benchmark_warmup,network_runs=rows,benchmark_runs=benchmark,
                capacity=capacity_summary(rows,config),
                scope='Finite-day capacity for this timetable, policy and stated service targets; not an optimum.')


def save_results(data,path):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load_results(path,config):
    data=json.loads(Path(path).read_text())
    if data['fingerprint'] != fingerprint(config):
        raise ValueError('Saved experiments use different code/configuration. Set RUN_EXPERIMENTS=True to recompute.')
    return data


if __name__=='__main__':
    destination=Path(__file__).with_name('experiment_results.json')
    save_results(run_all(),destination)
    print('Saved',destination)
