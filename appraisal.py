"""TAG-informed post-processing of saved results. No simulator imports or runs.

All monetary values are real 2023 GBP in market-price units. Present values
are discounted to 2023. Missing infrastructure costs are not treated as zero:
the default result is a partial budget available to meet those costs.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppraisalCase:
    purpose: str = 'other_non_work'      # Alternative: commuting; no assumed mix.
    waiting_multiplier: float = 2.0     # TAG public-transport transfer proxy.
    time_value_factor: float = 1.0      # For published TAG sensitivity ranges.
    energy_scenario: str = 'central'    # DESNZ low / central / high.
    operating_days: int = 300           # Illustrative operating scenario, not data.
    opening_year: int = 2027            # Illustrative timing, not a launch forecast.
    years: int = 10                     # Scenario duration, not charger life.
    appraisal_year: int = 2026
    pv_base_year: int = 2023
    discount_rates: tuple = (.03, .025, .0225)  # HMT October 2026 schedule.

    def validate(self):
        if self.purpose not in ('other_non_work','commuting'):
            raise ValueError('No supported business or mixed-purpose default.')
        if self.energy_scenario not in ('low','central','high'):
            raise ValueError('Choose a DESNZ energy scenario.')
        for n in (self.operating_days,self.opening_year,self.years,self.appraisal_year,self.pv_base_year):
            if type(n) is not int: raise ValueError('Years and days must be integers.')
        if not 0 < self.operating_days <= 365 or self.years < 1:
            raise ValueError('Use 1-365 operating days and a positive duration.')
        if not self.pv_base_year <= self.appraisal_year <= self.opening_year:
            raise ValueError('Require PV base <= appraisal year <= opening year.')
        if len(self.discount_rates)!=3 or any(not math.isfinite(r) or r<=-1 for r in self.discount_rates):
            raise ValueError('Three finite discount rates greater than -1 required.')
        if any(not math.isfinite(x) or x<0 for x in (self.waiting_multiplier,self.time_value_factor)):
            raise ValueError('Finite nonnegative valuation factors required.')


def load_inputs(path=HERE/'data/verified_inputs.json'):
    return json.loads(Path(path).read_text())


def load_pairs(path=HERE/'data/simulation_results.json'):
    """Observed 12-hour outcomes, including every unfinished passenger.

    Waiting and onboard/ground time stop at the shared observation horizon.
    The unobserved residual burden is a separate sensitivity, never assumed to
    be an observed zero. Recovery diagnostics do not enter the appraisal.
    """
    data=json.loads(Path(path).read_text());groups={k:{} for k in ('one_charger','two_chargers')}
    for run in data['runs']:
        group=groups[run['option']]
        if run['seed'] in group: raise ValueError('Duplicate replication seed.')
        group[run['seed']]=run
    if set(groups['one_charger'])!=set(groups['two_chargers']):
        raise ValueError('Both options require the same complete seed set.')
    if len(groups['one_charger'])<2: raise ValueError('Multiple paired runs required.')
    pairs=[]
    for seed in sorted(groups['one_charger']):
        a,b=(groups[k][seed] for k in groups)
        if any(r['original_12h']['safety_failure'] for r in (a,b)):
            raise ValueError('Safety-stopped run: do not value or silently drop it.')
        if not math.isclose(a['original_12h']['end_min'],b['original_12h']['end_min']):
            raise ValueError('A common complete observation horizon is required.')
        row={'seed':seed}
        for suffix,run in [('a',a),('b',b)]:
            final=run['original_12h'];people=final['passengers'];count=len(people)
            if not count: raise ValueError('Cannot value an empty passenger cohort.')
            end=final['end_min'];wait=other=0.
            for p in people:
                ts=[p[k] for k in ('arrival_min','boarding_min','departure_min','completion_min') if p[k] is not None]
                if any(not math.isfinite(t) or t>end for t in ts) or ts!=sorted(ts):
                    raise ValueError('Invalid passenger time record.')
                boarded=end if p['boarding_min'] is None else p['boarding_min']
                completed=end if p['completion_min'] is None else p['completion_min']
                wait+=boarded-p['arrival_min']
                other+=completed-boarded
            row['arrivals_'+suffix]=count
            row['wait_hours_'+suffix]=wait/60
            row['other_hours_'+suffix]=other/60
            for k in ('energy_with_replenishment_kwh','flights','empty_flights','end_min','terminal_replenishment_kwh'):
                row[k+'_'+suffix]=final[k]
            row['grid_kwh_'+suffix]=sum(final['grid_kwh_by_port'].values())
            row['unfinished_12h_'+suffix]=run['original_12h']['incomplete']
            row['unfinished_day_'+suffix]=int(run['original_12h']['incomplete']>0)
        ca=[(p['id'],p['origin'],p['dest'],p['arrival_min']) for p in a['original_12h']['passengers']]
        cb=[(p['id'],p['origin'],p['dest'],p['arrival_min']) for p in b['original_12h']['passengers']]
        if ca!=cb: raise ValueError('Options do not contain identical offered journeys.')
        row['wait_hours_saved']=row['wait_hours_a']-row['wait_hours_b']
        row['other_hours_saved']=row['other_hours_a']-row['other_hours_b']
        row['extra_energy_kwh']=row['energy_with_replenishment_kwh_b']-row['energy_with_replenishment_kwh_a']
        row['extra_unfinished']=row['unfinished_12h_b']-row['unfinished_12h_a']
        row['extra_unfinished_day']=row['unfinished_day_b']-row['unfinished_day_a']
        pairs.append(row)
    return pairs,data['metadata']


def bootstrap_indices(n, resamples=5000, seed=20261116):
    if n<2 or resamples<100: raise ValueError('At least two runs and 100 resamples required.')
    return np.random.default_rng(seed).integers(0,n,size=(resamples,n))


def mean_ci(values, indices=None):
    """Percentile 95% CI of the mean, resampling whole paired runs."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.all(np.isfinite(values)):
        raise ValueError('At least two finite run-level measurements required.')
    indices=bootstrap_indices(len(values)) if indices is None else np.asarray(indices)
    if indices.ndim!=2 or indices.shape[1]!=len(values) or indices.min()<0 or indices.max()>=len(values):
        raise ValueError('Bootstrap indices must match the replication count.')
    low,high=np.quantile(values[indices].mean(axis=1),[.025,.975])
    return dict(mean=float(values.mean()),low=float(low),high=float(high),n=len(values),resamples=len(indices))


def time_value(inputs, case, year):
    """GBP/person-hour, real 2023 market prices; growth is not inflation."""
    index={r['year']:r['index'] for r in inputs['time_value_growth']['rows']}
    if year not in index: raise ValueError('Time-value year outside the saved evidence series.')
    base=inputs['time_values'][case.purpose]
    return base['value']*index[year]/index[base['value_year']]*case.time_value_factor


def energy_value(inputs, case, year):
    """LRVC rebased from 2025 prices, then converted to market-price units."""
    data={r['year']:r for r in inputs['resource_energy_projection']}
    if year not in data: raise ValueError('Energy year outside the saved evidence series.')
    pence=data[year][case.energy_scenario+'_pence_per_kwh']
    factor=inputs['unit_of_account']['market_price_conversion_factor']['value']
    deflator=inputs['price_conversion']['gdp_deflator_2025']['value']
    return pence/deflator*factor  # /100 pence and *100/deflator for rebasing cancel.


def daily_components(pairs, inputs, case, year=2026):
    """Valued observed time component minus energy; NOT whole-journey welfare."""
    case.validate()
    hours=np.array([case.waiting_multiplier*p['wait_hours_saved']+p['other_hours_saved'] for p in pairs])
    extra=np.array([p['extra_energy_kwh'] for p in pairs])
    benefits=hours*time_value(inputs,case,year)
    costs=extra*energy_value(inputs,case,year)
    return {'time_benefit':benefits,'extra_energy_cost':costs,'partial_net_benefit':benefits-costs}


def residual_time_adjustment(pairs, inputs, case, year, remaining_minutes):
    """Explicit illustrative common remaining GJT per unfinished journey.

    This is already generalised time; do not apply the waiting multiplier again.
    It does not include lost-trip utility or a ground-mode fare. There is no
    default because the simulator does not identify these remaining outcomes.
    """
    if not math.isfinite(remaining_minutes) or remaining_minutes<0:
        raise ValueError('Explicit nonnegative remaining generalised minutes required.')
    return -np.array([p['extra_unfinished'] for p in pairs])*remaining_minutes/60*time_value(inputs,case,year)


def discount_factor(year, case):
    """PV at 2023. First rate bridges 2023 to the 2026 appraisal date.

    Declining-rate bands then start at the appraisal year, not opening year.
    """
    if year<case.appraisal_year: raise ValueError('Costs must occur at or after appraisal.')
    n=year-case.appraisal_year
    if n>125: raise ValueError('HMT schedule is limited to 125 years.')
    durations=(min(n,30),min(max(n-30,0),45),max(n-75,0))
    denominator=(1+case.discount_rates[0])**(case.appraisal_year-case.pv_base_year)
    for r,t in zip(case.discount_rates,durations): denominator*=(1+r)**t
    return 1/denominator


def annual_cost_factor(case):
    case.validate()
    return sum(discount_factor(y,case) for y in range(case.opening_year,case.opening_year+case.years))


def benefit_budget(pairs, inputs, case):
    """Partial PV budget for all costs omitted from time-plus-energy benefits.

    Repeats the finite-day demand/operating experiment. No demand growth,
    deterioration or fleet-position reset is silently modelled.
    """
    case.validate();total=np.zeros(len(pairs))
    for year in range(case.opening_year,case.opening_year+case.years):
        total+=case.operating_days*daily_components(pairs,inputs,case,year)['partial_net_benefit']*discount_factor(year,case)
    return total


def upfront_budget(pairs, inputs, case, annual_additional_cost):
    """Available upfront capital in appraisal-year expenditure, real 2023 GBP.

    annual_additional_cost is explicitly supplied, in market-price real GBP/year,
    and covers ALL further annual differences. Renewal/residual impacts remain
    outside this frontier unless included as annual equivalents.
    """
    if not math.isfinite(annual_additional_cost) or annual_additional_cost<0:
        raise ValueError('An explicit nonnegative annual cost scenario is required.')
    return (benefit_budget(pairs,inputs,case)-annual_additional_cost*annual_cost_factor(case))/discount_factor(case.appraisal_year,case)


def net_present_value(pairs, inputs, case, *, upfront_cost, annual_additional_cost,
                      renewals, residual_value):
    """Scenario NPV only when all remaining cost categories are explicitly supplied.

    renewals maps calendar year to real 2023 market-price GBP. Explicit empty
    mapping / zero residual is an assumption by the caller, never a missing value.
    """
    numbers=[upfront_cost,annual_additional_cost,residual_value]+list(renewals.values())
    if any(x is None or not math.isfinite(x) or x<0 for x in numbers):
        raise ValueError('Explicit finite nonnegative costs/residual values required.')
    end=case.opening_year+case.years-1
    if any(type(y) is not int or not case.appraisal_year<=y<=end for y in renewals):
        raise ValueError('Renewal dates must be within the appraisal.')
    return (benefit_budget(pairs,inputs,case)-upfront_cost*discount_factor(case.appraisal_year,case)
            -annual_additional_cost*annual_cost_factor(case)
            -sum(v*discount_factor(y,case) for y,v in renewals.items())
            +residual_value*discount_factor(end,case))
