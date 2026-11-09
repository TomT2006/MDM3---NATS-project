"""Whole-replication inference; time points and passengers are not iid samples."""
import math
import numpy as np
from scipy.stats import binomtest, ttest_rel
from overload import detect_overload, OverloadRule


def bootstrap_mean(values, resamples=5000, seed=20261108, confidence=.95):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.all(np.isfinite(values)):
        raise ValueError('At least two finite independent-run summaries required.')
    if resamples<100 or not 0<confidence<1:raise ValueError('Invalid bootstrap settings.')
    rng=np.random.default_rng(seed);distribution=[]
    for start in range(0,resamples,128):
        ix=rng.integers(0,len(values),size=(min(128,resamples-start),len(values)))
        distribution.extend(values[ix].mean(axis=1))
    alpha=(1-confidence)/2
    lo,hi=np.quantile(distribution,[alpha,1-alpha])
    return dict(mean=float(values.mean()),low=float(lo),high=float(hi),n=len(values),
                resamples=resamples,method='whole-run percentile bootstrap',
                prediction_low=float(np.quantile(values,.025)),prediction_high=float(np.quantile(values,.975)))


def exact_probability(events,confidence=.95):
    events=np.asarray(events,dtype=bool)
    if not len(events):raise ValueError('At least one replication required.')
    k=int(events.sum());n=len(events)
    ci=binomtest(k,n).proportion_ci(confidence_level=confidence,method='exact')
    return dict(probability=k/n,low=float(ci.low),high=float(ci.high),events=k,n=n,
                method='Clopper-Pearson exact binomial')


def holm_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p);out=np.empty(len(p));running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(p)-rank)*p[i]);out[i]=min(1.,running)
    return out.tolist()


def paired_comparison(rows_a,rows_b,metric,resamples=5000,seed=20261108):
    a={int(r['seed']):r for r in rows_a};b={int(r['seed']):r for r in rows_b}
    if len(a)!=len(rows_a) or len(b)!=len(rows_b) or set(a)!=set(b):
        raise ValueError('Complete unique matched seed pairs required.')
    keys=sorted(a)
    if any(not a[s].get('feasible',True) or not b[s].get('feasible',True) for s in keys):
        return dict(status='unavailable: at least one paired run stopped for safety')
    x=np.array([float(a[s][metric]) for s in keys]);y=np.array([float(b[s][metric]) for s in keys])
    diff=y-x
    result=bootstrap_mean(diff,resamples,seed)
    # A large-sample paired t test of mean difference; normality is not asserted.
    p=1. if np.all(diff==0) else (0. if np.std(diff)==0 else float(ttest_rel(y,x).pvalue))
    result.update(metric=metric,definition='B minus A, matched replication seed',
                  pvalue=p,test='paired t test of mean difference; large-sample approximation',
                  practical_threshold_min=10 if metric=='mean_observed_wait_min' else None,
                  paired_correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None)
    return result


def trajectory_bands(group,resamples=2000,seed=20261108,stride=5):
    """Same whole-run resample indices at every time; simultaneous grid band.

    A max-standardised-deviation bootstrap gives a nominal 95% simultaneous
    confidence band over this displayed finite grid, not all continuous times.
    """
    q=group['queue'].sum(axis=2)[:,::stride].astype(float)
    if not np.all(np.isfinite(q)):
        return dict(available=False,reason='Unconditional mean trajectory unavailable after safety stops.')
    n=len(q)
    if n<2:raise ValueError('Multiple replications required.')
    mean=q.mean(axis=0);se=q.std(axis=0,ddof=1)/math.sqrt(n)
    valid=se>1e-12
    rng=np.random.default_rng(seed);boot=[]
    for start in range(0,resamples,100):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(100,resamples-start))/n
        boot.append(weights@q)
    boot=np.vstack(boot)
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    maxima=np.max(np.abs((boot[:,valid]-mean[valid])/se[valid]),axis=1) if np.any(valid) else np.zeros(resamples)
    critical=float(np.quantile(maxima,.95))
    return dict(available=True,times=group['times'][::stride].tolist(),mean=mean.tolist(),
                pointwise_low=lo.tolist(),pointwise_high=hi.tolist(),
                simultaneous_low=np.maximum(0,mean-critical*se).tolist(),
                simultaneous_high=(mean+critical*se).tolist(),
                prediction_low=np.quantile(q,.025,axis=0).tolist(),
                prediction_high=np.quantile(q,.975,axis=0).tolist(),
                resamples=resamples,n=n,grid_minutes=int(stride*np.diff(group['times'])[0]))


def summarise(group,resamples=5000):
    rows=group['rows'];failed=any(r['safety_failure'] for r in rows)
    answer=dict(rate=group['metadata']['rate'],n=len(rows),
                overload=exact_probability([r['overload'] for r in rows]),
                safety=exact_probability([r['safety_failure'] for r in rows]),
                combined_failure=exact_probability([r['overload'] or r['safety_failure'] for r in rows]),
                overload_unknown_before_horizon=sum(r['overload_unobserved'] for r in rows),
                economic_ranking_eligible=not failed,metrics={})
    # Do not mix truncated safety runs with complete-day scores or silently drop them.
    if not failed:
        for i,key in enumerate(('score','mean_observed_wait_min','served_fraction','timely_fraction',
               'monitored_peak_queue','late_growth_per_hour','active_departures_per_hour','generalised_time_proxy_min')):
            values=[r[key] for r in rows]
            if all(v is not None and math.isfinite(v) for v in values):
                answer['metrics'][key]=bootstrap_mean(values,resamples,20261108+i)
    event_times=[r['overload_time_min'] for r in rows if r['overload']]
    answer['time_to_overload']=dict(observed_events=len(event_times),
        median_among_events=float(np.median(event_times)) if event_times else None,
        no_observed_overload=len(rows)-len(event_times),
        note='Median conditional on observed events; non-events are not assigned a failure time.')
    return answer


def threshold_sensitivity(group):
    from dataclasses import asdict,replace
    base=OverloadRule(**group['metadata']['rule'])
    settings=[base,replace(base,queue_limit=max(1,base.queue_limit//2)),
              replace(base,queue_limit=base.queue_limit*3//2),
              replace(base,persistence_min=max(base.sample_min,base.persistence_min//2//base.sample_min*base.sample_min)),
              replace(base,persistence_min=base.persistence_min*2),
              replace(base,growth_per_hour=0),replace(base,growth_per_hour=base.growth_per_hour*2)]
    rows=[]
    for rule in settings:
        events=[detect_overload(group['times'],q.sum(axis=1),rule,group['metadata']['demand_stop_min']) is not None
                for q in group['queue']]
        rows.append(dict(rule=asdict(rule),**exact_probability(events)))
    return rows
