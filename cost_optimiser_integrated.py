"""Configurable weighted-penalty sweeps; uncertainty is across independent runs.

This is a finite grid comparison, not an optimiser of the flight schedule.
Money is not implied by the weights. No simulation starts on import.
"""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import csv
import math
import numpy as np
from model import ScenarioConfig, Simulation, example_config
from results import audit
from overload import score_run as assess_run, weighted_score, LEGACY_WEIGHTS
from uncertainty import bootstrap_mean

BASELINE_PARAMETERS=asdict(example_config())
COST_WEIGHTS=LEGACY_WEIGHTS.copy()
DEFAULT_MULTIPLIERS=np.linspace(.5,2,10)
INTEGER_PARAMETERS={'fleet','seats','stands_per_port','queue_capacity','chargers_per_port','pads_per_port'}


def build_config(parameters,seed):
    values=deepcopy(parameters)
    overrides=values.pop('port_overrides',{})
    for key in list(values):
        if key.startswith('chargers_') and key!='chargers_per_port':
            port=key[len('chargers_'):]
            if port not in values['port_names']:raise ValueError('Unknown port in charging override.')
            overrides.setdefault(port,{})['chargers']=values.pop(key)
    values.update(port_overrides=overrides,seed=int(seed))
    c=ScenarioConfig(**values);c.validate();return c


def calculate_cost(components,weights=None):
    return weighted_score(components,weights)


def score_run(simulation,weights=None):
    row,_,_=assess_run(simulation)
    row['cost']=weighted_score(row,weights)
    # A stopped failure is explicitly ineligible; partial costs remain inspectable.
    row['ranking_eligible']=row['feasible']
    return row


def get_sweep_values(parameter,values,multipliers,baseline):
    selected=list(values) if values is not None else [baseline[parameter]*x for x in multipliers]
    integer=parameter in INTEGER_PARAMETERS or parameter.startswith('chargers_')
    if integer:
        if any(not math.isfinite(v) or int(v)!=v for v in selected):
            raise ValueError('Integer parameter sweeps require explicit integer values.')
        selected=[int(v) for v in selected]
    return list(dict.fromkeys(selected))


def get_seeds(seeds,value):
    answer=tuple(seeds.get(value,())) if isinstance(seeds,dict) else tuple(seeds)
    if not answer or len(set(answer))!=len(answer):raise ValueError('Distinct nonempty seed list required.')
    return answer


def run_parameter_sweep(parameter,values=None,*,multipliers=DEFAULT_MULTIPLIERS,seeds=(1,2,3),
        baseline=None,cost_weights=None,csv_path='cost_sweep_results.csv',make_plot=True,show_progress=True):
    baseline=deepcopy(BASELINE_PARAMETERS if baseline is None else baseline)
    plan=[]
    for value in get_sweep_values(parameter,values,multipliers,baseline):
        for seed in get_seeds(seeds,value):
            parameters=deepcopy(baseline);parameters[parameter]=value
            # Validate the entire sweep before starting; invalid settings are not silently skipped.
            plan.append((value,build_config(parameters,seed)))
    rows=[]
    for value,config in plan:
        sim=Simulation(config).run();audit(sim)
        rows.append(dict(parameter=parameter,parameter_value=value,**score_run(sim,cost_weights)))
        if show_progress:print(f'{parameter}={value}, seed={config.seed}: audited',flush=True)
    write_csv(rows,csv_path)
    if make_plot:plot_sweep_results(rows,parameter)
    return rows


def write_csv(rows,csv_path):
    if not rows:return
    path=Path(csv_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_sweep_results(rows,parameter,ax=None):
    import matplotlib.pyplot as plt
    groups={}
    for r in rows:
        if str(r.get('ranking_eligible',True)).lower() not in ('true','1'):continue
        groups.setdefault(float(r['parameter_value']),[]).append(float(r['cost']))
    if not groups:raise ValueError('No eligible runs.')
    levels=sorted(groups);stats=[bootstrap_mean(groups[v]) for v in levels]
    if ax is None:_,ax=plt.subplots()
    m=np.array([s['mean'] for s in stats]);lo=np.array([s['low'] for s in stats]);hi=np.array([s['high'] for s in stats])
    ax.errorbar(levels,m,yerr=[np.maximum(0,m-lo),np.maximum(0,hi-m)],fmt='o-',capsize=3)
    ax.set(xlabel=parameter,ylabel='Weighted penalty (not currency)',title='Mean and 95% whole-run bootstrap interval')
    return ax


def plot_from_csv(csv_path):
    with Path(csv_path).open() as f:rows=list(csv.DictReader(f))
    return plot_sweep_results(rows,rows[0]['parameter'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-csv',type=Path)
    args=parser.parse_args()
    if args.plot_csv:
        plot_from_csv(args.plot_csv)
        import matplotlib.pyplot as plt
        plt.show()
    else:parser.print_help()
