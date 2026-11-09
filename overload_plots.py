"""Scientific figures for saved replication/bootstrapping results."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from overload import load_group


def finish(fig,name):
    fig.tight_layout();Path('figures').mkdir(exist_ok=True)
    fig.savefig(Path('figures')/name,dpi=160,bbox_inches='tight');plt.show()


def demand_design():
    t=np.linspace(0,10,401)
    profile=np.array([.5 if x<2.5 or 5<=x<7.5 else 1.5 for x in t])
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for rate in (36,48,120):axes[0].plot(t,np.full(len(t),rate),label=f'{rate} passengers/h')
    axes[0].set(xlabel='Active demand time (h)',ylabel='Network passenger arrival rate',title='Separate constant-demand scenarios')
    axes[0].legend(fontsize=8)
    axes[1].plot(t,profile,label='Two-peak sensitivity')
    axes[1].axhline(1,color='gray',ls='--',label='Same daily mean')
    axes[1].set(xlabel='Active demand time (h)',ylabel='Multiplier of scenario mean',ylim=(0,1.8))
    axes[1].legend(fontsize=8);finish(fig,'11_demand_design.png')


def risk_and_service(data):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    pilot=sorted([(v['rate'],v) for k,v in data['summaries'].items() if k.startswith('pilot_')])
    confirm=sorted([(v['rate'],v) for k,v in data['summaries'].items() if k.startswith('confirm_')])
    for group,fmt,label in [(pilot,'o--',f"Pilot: {data['plan']['pilot_n']} runs/level"),(confirm,'s',f"Confirmation: {data['plan']['confirmation_n']:,} runs/level")]:
        x=[r for r,v in group];m=np.array([v['overload']['probability'] for r,v in group])
        low=np.array([v['overload']['low'] for r,v in group]);high=np.array([v['overload']['high'] for r,v in group])
        axes[0].errorbar(x,m,yerr=[m-low,high-m],fmt=fmt,capsize=3,label=label)
    axes[0].axvline(data['plan']['timetable_seats_per_hour'],color='gray',ls=':',label='Timetable seat-rate ceiling')
    axes[0].set(xlabel='Offered passengers/hour',ylabel='Observed overload probability',ylim=(-.03,1.03))
    axes[0].legend(fontsize=7)
    for key,label,color in [('served_fraction','Completed','#2677ad'),('timely_fraction','Departed within 60 min','#c65b47')]:
        valid_pilot=[(r,v) for r,v in pilot if key in v['metrics']]
        valid_confirm=[(r,v) for r,v in confirm if key in v['metrics']]
        x=[r for r,v in valid_pilot];m=[v['metrics'][key]['mean'] for r,v in valid_pilot]
        axes[1].plot(x,m,'o--',color=color,alpha=.5)
        x=[r for r,v in valid_confirm];stats=[v['metrics'][key] for r,v in valid_confirm]
        m=np.array([s['mean'] for s in stats]);lo=np.array([s['low'] for s in stats]);hi=np.array([s['high'] for s in stats])
        axes[1].errorbar(x,m,yerr=[m-lo,hi-m],fmt='s',color=color,capsize=3,label=label)
    axes[1].set(xlabel='Offered passengers/hour',ylabel='Fraction of all arrivals',ylim=(0,1.03))
    axes[1].legend(fontsize=8)
    rates=[r for r,v in confirm]
    for key,label in [('active_departures_per_hour','Observed departures/h')]:
        valid=[(r,v) for r,v in confirm if key in v['metrics']]
        s=[v['metrics'][key] for r,v in valid];m=np.array([x['mean'] for x in s])
        axes[2].errorbar([r for r,v in valid],m,yerr=[m-np.array([x['low'] for x in s]),np.array([x['high'] for x in s])-m],fmt='o-',capsize=3,label=label)
    axes[2].plot(rates,rates,'--',color='gray',label='Demand equals departures')
    axes[2].set(xlabel='Offered passengers/hour',ylabel='Passengers/hour during active demand')
    axes[2].legend(fontsize=8);finish(fig,'12_overload_risk.png')


def ensemble_trajectories(data):
    rates=data['plan']['confirmed_rates'];fig,axes=plt.subplots(1,len(rates),figsize=(14,4),sharey=True)
    for ax,rate in zip(axes,rates):
        b=data['bands'][f'confirm_{rate}']
        if not b['available']:
            ax.text(.5,.5,b['reason'],ha='center',wrap=True,transform=ax.transAxes);continue
        t=np.array(b['times'])/60
        ax.fill_between(t,b['prediction_low'],b['prediction_high'],color='#c7ddeb',label='Central 95% of individual runs')
        ax.fill_between(t,b['simultaneous_low'],b['simultaneous_high'],color='#2677ad',alpha=.5,label='95% simultaneous mean band')
        ax.plot(t,b['mean'],color='#12527b',label='Ensemble mean')
        ax.axhline(data['plan']['rule']['queue_limit'],color='#c65b47',ls=':',label='Queue threshold (other conditions apply)')
        ax.axvline(data['plan']['demand_stop_min']/60,color='gray',ls='--');ax.set(xlabel='Hours',title=f'{rate} passengers/h')
    axes[0].set_ylabel('Passengers arrived but not departed')
    axes[0].legend(fontsize=7);finish(fig,'13_ensemble_queues.png')


def time_and_sensitivity(data,folder):
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for rate in data['plan']['confirmed_rates']:
        g=load_group(Path(folder)/f'confirm_{rate}');t=g['times']
        event=np.array([r['overload_time_min'] if r['overload_time_min'] is not None else np.inf for r in g['rows']])
        risk=[np.mean(event<=x) for x in t]
        axes[0].plot(t/60,risk,label=f'{rate}/h')
    axes[0].axvline(data['plan']['demand_stop_min']/60,color='gray',ls='--',label='Event window ends')
    axes[0].set(xlabel='Hours',ylabel='Fraction with observed overload by time',ylim=(0,1.03))
    axes[0].legend(fontsize=8)
    rules=data['thresholds'][f"confirm_{data['plan']['confirmed_rates'][0]}"]
    labels=['Primary',f"Q={rules[1]['rule']['queue_limit']}",f"Q={rules[2]['rule']['queue_limit']}",
            f"Persist {rules[3]['rule']['persistence_min']}",f"Persist {rules[4]['rule']['persistence_min']}",
            'Growth 0',f"Growth {rules[6]['rule']['growth_per_hour']:g}"]
    for rate in data['plan']['confirmed_rates']:
        rows=data['thresholds'][f'confirm_{rate}']
        axes[1].plot(range(7),[r['probability'] for r in rows],'o-',label=f'{rate}/h')
    axes[1].set(xticks=range(7),xticklabels=labels,ylabel='Observed overload probability',ylim=(0,1.03))
    axes[1].tick_params(axis='x',labelrotation=40);axes[1].legend(fontsize=8)
    for rate in data['plan']['confirmed_rates']:
        b=data['bands'][f'continuous_{rate}']
        if not b['available']:
            axes[2].text(.05,.95,f'{rate}/h: mean unavailable after safety stops',transform=axes[2].transAxes);continue
        t=np.array(b['times'])/60
        axes[2].plot(t,b['mean'],label=f'{rate}/h')
        axes[2].fill_between(t,b['simultaneous_low'],b['simultaneous_high'],alpha=.15)
    axes[2].set(xlabel='Hours of continuing demand',ylabel='Mean passenger backlog',title=f"Continuing demand: {data['plan']['sensitivity_n']} runs/level")
    axes[2].legend(fontsize=8);finish(fig,'14_time_and_thresholds.png')


def cost_comparisons(data):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,label in zip(axes[:2],('stands_per_port','fleet'),('Stands per port','Aircraft fleet')):
        rows=[r for r in data['cost_sweeps'][key] if r['score'] is not None]
        if not rows:
            ax.text(.5,.5,'No complete-day ranking available',ha='center',transform=ax.transAxes);continue
        m=np.array([r['score']['mean'] for r in rows])
        lo=np.array([r['score']['low'] for r in rows]);hi=np.array([r['score']['high'] for r in rows])
        ax.errorbar([r['level'] for r in rows],m,yerr=[m-lo,hi-m],fmt='o-',capsize=3)
        ax.set(xlabel=label,ylabel='Mean weighted penalty (not currency)',title='95% whole-run bootstrap CI; 30 runs/level')
    for key,label in [('stands_per_port','Stands'),('fleet','Fleet')]:
        counts=data['weight_sensitivity'][key]['best_tested_counts']
        items=[(r['level'],counts.get(str(r['level']),counts.get(r['level'],0))) for r in data['cost_sweeps'][key]]
        x=[k for k,v in items];y=[v for k,v in items]
        axes[2].plot(x,y,'o-',label=label)
    axes[2].set(xlabel='Lowest-mean-score tested configuration',ylabel='Count across 243 assumed weight settings',title='Weight sensitivity, not an optimum probability')
    axes[2].legend(fontsize=8);finish(fig,'15_cost_uncertainty.png')


def profile_comparison(data,folder):
    rate=data['plan']['transition_selected_from_independent_pilot']
    peak=load_group(Path(folder)/f'peaks_{rate}');constant=load_group(Path(folder)/f'confirm_{rate}')
    seeds=set(r['seed'] for r in peak['rows']);ix=[i for i,r in enumerate(constant['rows']) if r['seed'] in seeds]
    if any(not r['feasible'] for r in peak['rows']) or any(not constant['rows'][i]['feasible'] for i in ix):
        print('Full-day paired profile comparison unavailable after safety stops.');return
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for g,indices,label in [(constant,ix,'Constant rate'),(peak,list(range(len(peak['rows']))),'Two peaks; same total demand')]:
        q=g['queue'][indices].sum(axis=2)
        axes[0].plot(g['times']/60,q.mean(axis=0),label=label)
    axes[0].set(xlabel='Hours',ylabel='Mean passenger backlog',title=f'Matched {len(ix)} runs at mean demand {rate}/h')
    axes[0].legend(fontsize=8)
    a={r['seed']:r for r in constant['rows']};delta=[r['mean_observed_wait_min']-a[r['seed']]['mean_observed_wait_min'] for r in peak['rows']]
    axes[1].hist(delta,bins=15,color='#2677ad');axes[1].axvline(0,color='gray',ls='--')
    axes[1].set(xlabel='Two peaks minus constant: observed wait/passenger (min)',ylabel='Matched replications')
    finish(fig,'16_demand_profile.png')


def bootstrap_distribution(data,folder):
    """Individual-day variation versus bootstrap uncertainty in the mean."""
    rate=data['plan']['transition_selected_from_independent_pilot']
    group=load_group(Path(folder)/f'confirm_{rate}')
    if any(not r['feasible'] or r['mean_observed_wait_min'] is None for r in group['rows']):
        print('Bootstrap mean-wait distribution unavailable: incomplete days or no arrivals.');return
    values=np.array([r['mean_observed_wait_min'] for r in group['rows']])
    rng=np.random.default_rng(20261109);means=[]
    for start in range(0,5000,128):
        indices=rng.integers(0,len(values),size=(min(128,5000-start),len(values)))
        means.extend(values[indices].mean(axis=1))
    lo,hi=np.quantile(means,[.025,.975])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(values,bins=25,color='#8ab6d0')
    axes[0].set(xlabel='One-day mean observed wait/arrival (min)',ylabel='Independent physical runs',title=f'{len(values):,} runs at {rate} passengers/hour')
    axes[1].hist(means,bins=35,color='#2677ad')
    for value in (lo,hi):axes[1].axvline(value,color='#c65b47',ls='--')
    axes[1].set(xlabel='Resampled mean of day-level waiting (min)',ylabel='Whole-run bootstrap draws',title='5,000 resamples; dashed 95% percentile limits')
    finish(fig,'17_bootstrap_distribution.png')
