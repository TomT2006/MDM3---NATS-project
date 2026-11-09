"""Analysis of saved whole-run outputs; does not execute the physical simulator."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from scipy.stats import binomtest
from overload import load_group,weighted_score,LEGACY_WEIGHTS,OverloadRule,fingerprint
from uncertainty import (summarise,trajectory_bands,threshold_sensitivity,
                        paired_comparison,holm_adjust,bootstrap_mean,exact_probability)
from model import example_config


def checked_group(folder,config,profile='constant'):
    group=load_group(folder)
    if group['metadata']['fingerprint']!=fingerprint(config,OverloadRule(**group['metadata']['rule']),profile):
        raise ValueError(f'Source/configuration mismatch for {folder}.')
    return group


def analyse(folder,config=None):
    folder=Path(folder);base=config or example_config()
    plan=json.loads((folder/'study_plan.json').read_text())
    summaries={};groups={};bands={};thresholds={}
    for phase in ('pilot','confirm','continuous'):
        rates=plan['pilot_rates'] if phase=='pilot' else plan['confirmed_rates']
        for rate in rates:
            name=f'{phase}_{rate}'
            c=replace(base,total_demand_per_hour=rate)
            if phase=='continuous':c=replace(c,demand_stop_min=1440,horizon_min=1440)
            g=checked_group(folder/name,c);groups[name]=g
            summaries[name]=summarise(g)
            if phase in ('confirm','continuous'):bands[name]=trajectory_bands(g)
            if phase=='confirm':thresholds[name]=threshold_sensitivity(g)
    transition=plan['transition_selected_from_independent_pilot']
    name=f'peaks_{transition}'
    g=checked_group(folder/name,replace(base,total_demand_per_hour=transition),'two_peaks')
    groups[name]=g;summaries[name]=summarise(g);bands[name]=trajectory_bands(g)
    # The two adjacent confirmation comparisons were specified before the runs.
    pairs=[]
    rates=plan['confirmed_rates']
    for a,b in zip(rates,rates[1:]):
        comparison=paired_comparison(groups[f'confirm_{a}']['rows'],groups[f'confirm_{b}']['rows'],'mean_observed_wait_min')
        comparison.update(A=a,B=b);pairs.append(comparison)
    if all('pvalue' in r for r in pairs):
        for r,p in zip(pairs,holm_adjust([r['pvalue'] for r in pairs])):r['holm_pvalue']=p
    # Same passengers per route in the profile comparison; their times change.
    common={r['seed'] for r in g['rows']}
    constant=[r for r in groups[f'confirm_{transition}']['rows'] if r['seed'] in common]
    profile_pair=paired_comparison(constant,g['rows'],'mean_observed_wait_min')
    profile_pair['same_arrival_counts']=all(a['arrivals']==b['arrivals'] for a,b in zip(sorted(constant,key=lambda r:r['seed']),sorted(g['rows'],key=lambda r:r['seed'])))
    # Matched queue growth at 12 and 24 hours of the SAME continuing-demand run.
    horizon=[]
    for rate in rates:
        g=groups[f'continuous_{rate}'];q=g['queue'].sum(axis=2);t=g['times']
        if not np.all(np.isfinite(q)):
            horizon.append(dict(rate=rate,status='unavailable after safety stops'));continue
        for hour in (12,24):
            ix=np.where(np.isclose(t,hour*60))[0][0]
            mask=(t>=hour*60-120)&(t<=hour*60)
            slopes=[float(np.polyfit(t[mask]/60,row[mask],1)[0]) for row in q]
            horizon.append(dict(rate=rate,horizon_h=hour,backlog=bootstrap_mean(q[:,ix]),late_growth=bootstrap_mean(slopes)))
    # Recomputed stand/fleet sweeps, not the provenance-free supplied CSVs.
    cost_plan=json.loads((folder/'cost_plan.json').read_text())
    cost={};weight_sensitivity={};cost_groups={}
    import itertools
    for parameter,levels in cost_plan.items():
        cost[parameter]=[];cost_groups[parameter]={}
        for entry in levels:
            level=entry['level'];c=replace(base,**{parameter:level})
            g=checked_group(folder/entry['folder'],c);cost_groups[parameter][level]=g
            failed=sum(r['safety_failure'] for r in g['rows'])
            cost[parameter].append(dict(level=level,
                score=None if failed else bootstrap_mean([r['score'] for r in g['rows']]),
                served=None if failed else bootstrap_mean([r['served_fraction'] for r in g['rows']]),
                timely=None if failed else bootstrap_mean([r['timely_fraction'] for r in g['rows']]),
                failures=failed))
        winners=[]
        for factors in itertools.product((.5,1.,2.),repeat=4):
            for stand_weight in (0.,5.,20.):
                weights={k:LEGACY_WEIGHTS[k]*v for k,v in zip(('passenger_wait_min','uncompleted_passenger','flight','aircraft_hour'),factors)}
                weights['stand_hour']=stand_weight
                means={level:np.mean([weighted_score(r,weights) for r in g['rows']])
                       for level,g in cost_groups[parameter].items() if all(r['feasible'] for r in g['rows'])}
                winners.append(dict(factors=factors,stand_weight=stand_weight,
                    best_tested_level=min(means,key=means.get) if means else None))
        from collections import Counter
        weight_sensitivity[parameter]=dict(settings=len(winners),
            best_tested_counts=dict(Counter(r['best_tested_level'] for r in winners)),
            note='Frequency across analyst-selected weights, not probability of a true optimum.',
            details=winners)
    cost_pairs=[]
    for parameter,a,b in [('stands_per_port',2,3),('fleet',8,10)]:
        r=paired_comparison(cost_groups[parameter][a]['rows'],cost_groups[parameter][b]['rows'],'score')
        r.update(parameter=parameter,A=a,B=b);cost_pairs.append(r)
    if all('pvalue' in r for r in cost_pairs):
        for r,p in zip(cost_pairs,holm_adjust([r['pvalue'] for r in cost_pairs])):r['holm_pvalue']=p
    # No precise crossing is interpolated from a few noisy scenario levels.
    below=[r for r in rates if summaries[f'confirm_{r}']['overload']['high']<.5]
    above=[r for r in rates if summaries[f'confirm_{r}']['overload']['low']>.5]
    bracket=dict(below_half_risk_tested=below,above_half_risk_tested=above,
                 note='Pointwise exact intervals. No monotone risk curve or unique critical rate is assumed.')
    plan['demand_stop_min']=base.demand_stop_min
    plan['horizon_min']=base.horizon_min
    plan['timetable_seats_per_hour']=groups[f"confirm_{rates[0]}"]['metadata']['timetable_seats_per_hour']
    result=dict(plan=plan,summaries=summaries,bands=bands,thresholds=thresholds,
                demand_comparisons=pairs,profile_comparison=profile_pair,continuing_horizons=horizon,
                cost_sweeps=cost,cost_comparisons=cost_pairs,weight_sensitivity=weight_sensitivity,
                transition_bracket=bracket)
    (folder/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Whole-run bootstrap analysis saved.',flush=True)
    return result


if __name__=='__main__':analyse(Path(__file__).with_name('overload_data'))
