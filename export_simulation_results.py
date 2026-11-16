"""Optional adapter, run separately in the ORIGINAL simulator's environment.

This file contains no aircraft/scheduling/queue model. The TAG notebook never
imports it. It imports the existing simulator only when explicitly run.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
import argparse
import hashlib
import json
import math
import sys


def trial(job):
    simulator,label,config,seed=job
    sys.path.insert(0,simulator)
    model=import_module('model');outputs=import_module('results');demand=import_module('overload')
    config={**config,'seed':seed,'port_names':tuple(config['port_names'])}
    for key in ('routes_km','demand_per_hour','service_intervals_min'):
        if config.get(key):config[key]={tuple(k.split('->')):v for k,v in config[key].items()}
    c=model.ScenarioConfig(**config)
    arrivals=demand.passenger_arrivals(c,sum(c.demand().values()),seed)
    sim=model.Simulation(c,passenger_arrivals=arrivals).run();outputs.audit(sim)
    minute=lambda t:None if t is None else float(t*c.time_scale_min)
    grid={p:0. for p in sim.ports}
    for _,port,a,b,before,after in sim.log.charges:
        grid[port]+=float(max(0,min(sim.env.now,b)-a)/(b-a))*(after-before)*c.battery_kwh/c.charging_efficiency
    initial=c.fleet*c.initial_soc*c.battery_kwh
    terminal=sum(a.soc*c.battery_kwh for a in sim.aircraft)
    restoration=(initial-terminal)/c.charging_efficiency
    holding=outputs.measurements(sim)['holding_kwh']
    mission=sum((before-after)*c.battery_kwh for _,_,before,after,_ in sim.log.energy_used)
    assert math.isclose((sum(grid.values())+restoration)*c.charging_efficiency,mission+holding,abs_tol=1e-7)
    record=dict(end_min=minute(sim.env.now),safety_failure=sim.failure_event.triggered,
        incomplete=sum(p.completed is None for p in sim.passengers),
        passengers=[dict(id=p.name,origin=p.origin,dest=p.dest,arrival_min=minute(p.arrived),
            boarding_min=minute(p.boarded),departure_min=minute(p.departed),completion_min=minute(p.completed)) for p in sim.passengers],
        grid_kwh_by_port=grid,initial_battery_kwh=initial,terminal_battery_kwh=terminal,
        terminal_replenishment_kwh=restoration,energy_with_replenishment_kwh=sum(grid.values())+restoration,
        mission_kwh=mission,holding_kwh=holding,
        flights=sum(f.departed is not None for f in sim.flights),
        empty_flights=sum(f.departed is not None and not f.passengers for f in sim.flights),
        aircraft_terminal=[dict(name=a.name,port=a.port,soc=float(a.soc),busy=a.busy) for a in sim.aircraft])
    return dict(option=label,seed=seed,original_12h=record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator',type=Path,required=True)
    parser.add_argument('--configs',type=Path,default=Path(__file__).parent/'data/simulation_configs.json')
    parser.add_argument('--output',type=Path,required=True,help='New JSON file; existing files are not overwritten.')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new export filename to preserve existing results.')
    simulator=args.simulator.resolve()
    cfg=json.loads(args.configs.read_text())['configs']
    if len(cfg)!=2:raise ValueError('This appraisal comparison requires two configurations.')
    jobs=[]
    for label,entry in zip(('one_charger','two_chargers'),cfg.values()):
        for seed in entry['seeds']:jobs.append((str(simulator),label,entry['config'],seed))
    if args.workers<1:raise ValueError('Positive worker count required.')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(trial,jobs))
    metadata=dict(configs=cfg,source_code_hashes={f:hashlib.sha256((simulator/f).read_bytes()).hexdigest()
        for f in ('model.py','queues.py','safety.py','results.py','overload.py')},
        protocol='Fixed-horizon export; all offered passengers retained; energy includes terminal battery restoration accounting.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(metadata=metadata,runs=rows),allow_nan=False))
    print(f'Exported {len(rows)} runs to {args.output}')


if __name__=='__main__':main()
