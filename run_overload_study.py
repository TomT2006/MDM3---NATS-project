"""Pilot -> independent confirmation -> continuing-demand/profile sensitivity.

No random time-segment stitching. Each worker runs an intact physical trajectory.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,os
from model import example_config
from overload import run_group,OverloadRule
from uncertainty import summarise


def run_study(destination,workers=4,pilot_n=30,confirm_n=1000,sensitivity_n=100,
              config=None,rates=(12,24,36,48,60,72,96,120),reference_rate=36,rule=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    base=config or example_config()
    seat_load=base.fleet*base.seats
    rule=rule or OverloadRule(queue_limit=2*seat_load,
        growth_per_hour=seat_load//2 if seat_load%2==0 else seat_load/2)
    rates=tuple(sorted(set(rates)))
    if reference_rate not in rates or not any(reference_rate<r<max(rates) for r in rates):
        raise ValueError("Include the reference rate, at least one higher intermediate rate and a stress rate.")
    pilot=[]
    for rate in rates:
        group=run_group(replace(base,total_demand_per_hour=rate),range(1,pilot_n+1),
                        destination/f'pilot_{rate}',rule,workers=workers)
        p=sum(r['overload'] for r in group['rows'])/len(group['rows'])
        pilot.append(dict(rate=rate,probability=p))
        print('PILOT',rate,'overload probability',p,flush=True)
    # Adaptive selection uses pilot only. Independent confirmation seeds avoid
    # recycling the same random outcomes to select and then validate a threshold.
    transition=min((x for x in pilot if reference_rate<x['rate']<max(rates)),key=lambda x:abs(x['probability']-.5))['rate']
    confirmed=(reference_rate,transition,max(rates))
    plan=dict(pilot_rates=list(rates),pilot_n=pilot_n,confirmed_rates=list(confirmed),
              confirmation_n=confirm_n,confirmation_seeds=[10001,10000+confirm_n],
              sensitivity_n=sensitivity_n,transition_selected_from_independent_pilot=transition,
              rule=rule.__dict__,workers=workers,reference_rate=reference_rate,
              demand_stop_min=base.demand_stop_min,horizon_min=base.horizon_min)
    (destination/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('CONFIRMATION PLAN',plan,flush=True)
    for rate in confirmed:
        run_group(replace(base,total_demand_per_hour=rate),range(10001,10001+confirm_n),
                  destination/f'confirm_{rate}',rule,workers=workers)
    for rate in confirmed:
        run_group(replace(base,total_demand_per_hour=rate,demand_stop_min=1440,horizon_min=1440),
                  range(30001,30001+sensitivity_n),destination/f'continuous_{rate}',rule,workers=workers)
    run_group(replace(base,total_demand_per_hour=transition),range(10001,10001+sensitivity_n),
              destination/f'peaks_{transition}',rule,profile='two_peaks',workers=workers)
    (destination/'COMPLETE.txt').write_text('All planned independent network replications completed and audited.\n')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('overload_data'))
    parser.add_argument('--workers',type=int,default=min(6,os.cpu_count() or 2))
    parser.add_argument('--pilot-n',type=int,default=30)
    parser.add_argument('--confirm-n',type=int,default=1000)
    parser.add_argument('--sensitivity-n',type=int,default=100)
    args=parser.parse_args()
    run_study(args.output,args.workers,args.pilot_n,args.confirm_n,args.sensitivity_n)
