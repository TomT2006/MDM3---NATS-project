"""Tables and report figures from the saved post-processing data only."""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from appraisal import (HERE, AppraisalCase, load_inputs, load_pairs, bootstrap_indices,
    mean_ci, daily_components, benefit_budget, upfront_budget, annual_cost_factor,
    discount_factor, residual_time_adjustment)

BLUE='#0072B2'; ORANGE='#D55E00'; GREY='#687582'


def analyse(case=None):
    case=case or AppraisalCase();pairs,metadata=load_pairs();inputs=load_inputs()
    indices=bootstrap_indices(len(pairs))
    ci=lambda values:mean_ci(values,indices)
    physical=[]
    for label,key in [('Waiting before boarding (passenger-hours/day)','wait_hours'),
                      ('Time after boarding (passenger-hours/day)','other_hours'),
                      ('Grid energy including battery restoration (kWh/day)','energy_with_replenishment_kwh'),
                      ('Flights/day','flights'),('Unfinished passengers/day','unfinished_12h'),
                      ('Runs with any unfinished journey (fraction)','unfinished_day')]:
        for suffix,option in [('a','One charger per port'),('b','Two chargers per port')]:
            physical.append(dict(metric=label,option=option,**ci([p[key+'_'+suffix] for p in pairs])))
    daily=[];budgets=[]
    for purpose in ('other_non_work','commuting'):
        c=replace(case,purpose=purpose)
        for name,values in daily_components(pairs,inputs,c,case.appraisal_year).items():
            daily.append(dict(purpose=purpose,component=name,**ci(values)))
        budgets.append(dict(purpose=purpose,**ci(benefit_budget(pairs,inputs,c))))
    variation=.6 if case.purpose=='other_non_work' else .25
    variations=[('Reference case',case),('Waiting multiplier 1',replace(case,waiting_multiplier=1)),
        (f'Time value -{variation:.0%}',replace(case,time_value_factor=1-variation)),
        (f'Time value +{variation:.0%}',replace(case,time_value_factor=1+variation)),
        ('Low energy cost',replace(case,energy_scenario='low')),
        ('High energy cost',replace(case,energy_scenario='high')),
        ('Published TAG discount rates',replace(case,discount_rates=(.035,.03,.025)))]
    sensitivities=[dict(scenario=name,case=asdict(c),**ci(benefit_budget(pairs,inputs,c))) for name,c in variations]
    annualisation=[]
    for days in (100,200,300,365):
        for years in (5,10,15):
            c=replace(case,operating_days=days,years=years)
            annualisation.append(dict(operating_days=days,years=years,**ci(benefit_budget(pairs,inputs,c))))
    residual=[]
    for minutes in (0,60,120,240):
        v=residual_time_adjustment(pairs,inputs,case,case.appraisal_year,minutes)
        residual.append(dict(assumed_remaining_generalised_minutes=minutes,**ci(v)))
    return dict(case=asdict(case),simulation_metadata=metadata,physical=physical,daily=daily,
        partial_budgets=budgets,sensitivities=sensitivities,annualisation=annualisation,
        residual_time_sensitivity=residual,
        extra_unfinished=ci([p['extra_unfinished'] for p in pairs]),
        extra_unfinished_day=ci([p['extra_unfinished_day'] for p in pairs]),
        wait_hours_saved=ci([p['wait_hours_saved'] for p in pairs]),
        other_hours_saved=ci([p['other_hours_saved'] for p in pairs]),
        extra_energy=ci([p['extra_energy_kwh'] for p in pairs]),
        scope='Observed 12-hour time component less energy. Partial, not total welfare or profit. No failed or unfinished passenger is removed.'),pairs,inputs


def export_tables(result,pairs,folder=HERE/'results'):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    (folder/'appraisal_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    tables={k:result[k] for k in ('physical','daily','partial_budgets','annualisation','residual_time_sensitivity')}
    tables['paired_physical_results']=pairs
    tables['sensitivities']=[{k:v for k,v in x.items() if k!='case'} for x in result['sensitivities']]
    for name,rows in tables.items():
        with (folder/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def figures(result,pairs,inputs,folder=HERE/'figures'):
    """Five figures; concise labels, vector PDF/SVG and 300 dpi PNG."""
    folder=Path(folder);folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
    case=AppraisalCase(**result['case']);ix=bootstrap_indices(len(pairs));captions={}
    ci=lambda x:mean_ci(x,ix)
    def save(fig,name,caption):
        for ax in fig.axes:
            ax.grid(axis='x' if name in ('02_daily_value','05_sensitivity') else 'y',color='#E5E7EB',lw=.6)
            ax.set_axisbelow(True)
        for ext in ('pdf','svg','png'):fig.savefig(folder/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.1)
        captions[name]=caption;plt.close(fig)
    def err(ax,values,y,color):
        st=ci(values)
        ax.errorbar(st['mean'],y,xerr=[[st['mean']-st['low']],[st['high']-st['mean']]],fmt='o',capsize=4,color=color)

    fig,axs=plt.subplots(1,2,figsize=(8.5,3.7),layout='constrained')
    for ax,key,title in zip(axs,['wait_hours','other_hours'],['Before boarding','After boarding']):
        for k,(suffix,color) in enumerate([('a',BLUE),('b',ORANGE)]):
            values=[60*p[key+'_'+suffix]/p['arrivals_'+suffix] for p in pairs]
            st=ci(values)
            ax.bar(k,st['mean'],color=color,width=.5)
            ax.errorbar(k,st['mean'],yerr=[[st['mean']-st['low']],[st['high']-st['mean']]],fmt='none',capsize=4,color='black')
        ax.set_xticks([0,1],['1 charger','2 chargers']);ax.set_title(title)
        ax.set_xlabel('Chargers per vertiport');ax.set_ylabel('Observed time (min/passenger)');ax.set_ylim(bottom=0)
    save(fig,'01_passenger_time','Mean observed time per offered passenger over 12 hours, across 80 paired replications. All passengers are retained, including unfinished journeys. Before-boarding time is separated from later ground/onboard time. Bars show run-mean averages; error bars are 95% whole-run percentile bootstrap confidence intervals (5,000 resamples). This is a finite-horizon time burden, not eventual whole-journey time.')

    fig,ax=plt.subplots(figsize=(7.3,3.7),layout='constrained')
    for y,purpose,color in [(0,'other_non_work',BLUE),(1,'commuting',ORANGE)]:
        values=daily_components(pairs,inputs,replace(case,purpose=purpose),case.appraisal_year)['partial_net_benefit']
        jitter=np.random.default_rng(7).uniform(-.12,.12,len(values))
        ax.scatter(values,y+jitter,color=color,alpha=.23,s=14,zorder=2)
        err(ax,values,y,color)
    ax.set_yticks([0,1],['Other non-work','Commuting']);ax.set_ylim(-.45,1.45)
    ax.set_xlabel('Partial benefit (£/day, 2023 prices)')
    ax.axvline(0,color=GREY,lw=.8);ax.set_title('Partial daily benefit from additional chargers')
    save(fig,'02_daily_value',f'Separate journey-purpose scenarios using {case.appraisal_year} TAG time values in real 2023 market-price units, a waiting multiplier of {case.waiting_multiplier:g}, and {case.energy_scenario} DESNZ energy resource costs. Light dots are individual paired-run results; dark markers and bars show the mean and its 95% paired bootstrap CI. Energy includes an accounting allowance to restore terminal battery energy. Unobserved journey outcomes, infrastructure, maintenance, fleet repositioning, degradation and externalities remain outside this partial benefit. It is not operator profit.')

    fig,ax=plt.subplots(figsize=(7.3,4.1),layout='constrained')
    max_annual=max(r['mean'] for r in result['partial_budgets'])/annual_cost_factor(case)
    costs=np.linspace(0,max_annual*1.05,101)
    for purpose,color,label in [('other_non_work',BLUE,'Other non-work'),('commuting',ORANGE,'Commuting')]:
        c=replace(case,purpose=purpose);b=benefit_budget(pairs,inputs,c);st=ci(b)
        discount=discount_factor(c.appraisal_year,c);factor=annual_cost_factor(c)
        ax.plot(costs/1000,(st['mean']-costs*factor)/discount/1e6,color=color,label=label)
        ax.fill_between(costs/1000,(st['low']-costs*factor)/discount/1e6,(st['high']-costs*factor)/discount/1e6,color=color,alpha=.15)
    ax.axhline(0,color=GREY,lw=.8);ax.set_ylim(bottom=0);ax.set_xlim(left=0)
    ax.set_xlabel('Extra annual costs (£000, 2023 prices)')
    ax.set_ylabel('Upfront budget (£m, 2023 prices)')
    ax.set_title('Budget frontier for additional charging capacity');ax.legend(frameon=False)
    save(fig,'03_investment_frontier',f'Conditional upfront budget for the compared package, with expenditure in {case.appraisal_year}, {case.operating_days} operating days per year and a {case.years}-year scenario from {case.opening_year}. Curves show the partial time-and-energy benefit remaining after an explicit additional annual cost/liability; shading is the 95% conditional bootstrap CI of the mean. These annual costs must cover omitted service outcomes and other recurring differences. Renewal/residual impacts require a separate allowance. Neither the horizontal intercept nor the vertical intercept is a charger price quote or a full investment recommendation.')

    fig,ax=plt.subplots(figsize=(7.3,4.1),layout='constrained')
    days=np.arange(25,366,10)
    for years,color in [(5,BLUE),(10,ORANGE),(15,'#009E73')]:
        # Proportionality to operating days is an explicit repeated-day scenario.
        c=replace(case,years=years,operating_days=1);st=ci(benefit_budget(pairs,inputs,c))
        ax.plot(days,days*st['mean']/1e6,label=f'{years} years',color=color)
        ax.fill_between(days,days*st['low']/1e6,days*st['high']/1e6,color=color,alpha=.12)
    ax.set_xlabel('Operating days per year (scenario)');ax.set_ylabel('Partial benefit budget (£m, PV 2023)')
    ax.set_title('Effect of annual use and appraisal duration');ax.legend(frameon=False);ax.set_xlim(0,365);ax.set_ylim(bottom=0)
    save(fig,'04_annualisation',f'{case.purpose.replace(chr(95),chr(32))} valuation scenario with constant repeated daily demand and operation. Durations of 5, 10 and 15 years and annual days are illustrative scenarios, not measured aircraft/charger life or a demand forecast. Time and energy values vary by year using the official saved series; present values use 2023 prices and a 2023 PV base. Shading shows conditional 95% confidence intervals for simulation means. Unresolved residual journeys, fleet reset, future deterioration, replacement and other costs remain to be met from this partial budget.')

    fig,ax=plt.subplots(figsize=(7.3,4.6),layout='constrained')
    rows=result['sensitivities'];ys=np.arange(len(rows))
    for y,r in zip(ys,rows):
        color=ORANGE if y==0 else BLUE
        ax.errorbar(r['mean']/1e6,y,xerr=[[(r['mean']-r['low'])/1e6],[(r['high']-r['mean'])/1e6]],fmt='o',capsize=3,color=color)
    ax.set_yticks(ys,[r['scenario'] for r in rows]);ax.invert_yaxis()
    ax.axvline(rows[0]['mean']/1e6,color=GREY,lw=.8,ls='--')
    ax.set_xlabel(f'Partial benefit budget (£m, PV {case.pv_base_year})');ax.set_title('Assumption sensitivity: '+case.purpose.replace('_',' '))
    save(fig,'05_sensitivity',f'One-at-a-time valuation sensitivities around {case.operating_days} operating days/year, {case.years} years from {case.opening_year}, {case.purpose.replace(chr(95),chr(32))} travel, and the configured reference discount schedule. The value-of-time range is from TAG A1.3 (60% for other non-work, 25% for commuting). Waiting multiplier 1 tests transferring the public-transport multiplier of 2 to eVTOL. Energy scenarios come from DESNZ. The published TAG comparison uses the older May 2026 schedule. Bars are 95% conditional bootstrap CIs, not probabilities assigned to the alternative assumptions or a simultaneous confidence region.')
    (folder/'captions.json').write_text(json.dumps(captions,indent=2))
    return captions


def main():
    result,pairs,inputs=analyse();export_tables(result,pairs);figures(result,pairs,inputs)
    print(json.dumps({'daily':result['daily'],'partial_budgets':result['partial_budgets']},indent=2))

if __name__=='__main__':main()
