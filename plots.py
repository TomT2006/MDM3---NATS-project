"""Presentation plots only. No scheduling or simulation logic lives here."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from results import measurements, waiting_trace
from experiments import interval
from mm1k import theory, transient

COLORS = {'cruise':'#2677ad','takeoff':'#54a4d3','landing':'#84c2e4',
          'charging':'#efad35','charger_wait':'#c65b47','idle':'#d6dce1',
          'boarding':'#4f9c82','unloading':'#81bba4','dispatch_wait':'#a77db5',
          'taxi_in':'#a1aeb8','taxi_out':'#758797','ground_delay':'#e4b3b0',
          'departure_pad_wait':'#aa4d80','arrival_pad_wait':'#614884'}


def finish(fig, filename):
    fig.tight_layout()
    Path('figures').mkdir(exist_ok=True)
    fig.savefig(Path('figures')/filename,dpi=160,bbox_inches='tight')
    plt.show()


def time_and_queues(sims):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=[str(s.config.headway_min) for s in sims]
    bottom=np.zeros(len(sims))
    for state,color in COLORS.items():
        values=np.array([measurements(s)['state_fraction'].get(state,0)*100 for s in sims])
        axes[0].bar(labels,values,bottom=bottom,color=color,label=state.replace('_',' '))
        bottom+=values
    axes[0].set(xlabel='Minimum same-route headway (min)',ylabel='Share of all aircraft time (%)',ylim=(0,100))
    axes[0].legend(fontsize=8,ncol=2,loc='upper left',bbox_to_anchor=(0,-.2))
    s=sims[1] if len(sims)>1 else sims[0]
    x=np.linspace(0,float(s.env.now)*s.config.time_scale_min-1e-6,300)
    trace=waiting_trace(s,x)
    for i,label in enumerate(('Passengers awaiting boarding','Aircraft waiting for charger','Aircraft at pad queues (N)','Future pad permits (R)')):
        axes[1].plot(x/60,trace[:,i],label=label)
    axes[1].axvline(s.config.demand_stop_min/60,color='gray',ls=':',label='Demand stops')
    axes[1].set(xlabel='Elapsed hours',ylabel='Count',title=f'Network totals: headway {s.config.headway_min} min')
    axes[1].legend(fontsize=8)
    finish(fig,'01_time_and_queues.png')


def rotation(sim, hours=4):
    c=sim.config
    fig,axes=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]})
    names=[a.name for a in sim.aircraft]
    end=min(float(sim.env.now),float(c.time(hours*60)))
    for name,state,port,a,b in sim.log.intervals(end):
        y=names.index(name)
        axes[0].broken_barh([(float(a)*c.time_scale_min/60,float(b-a)*c.time_scale_min/60)],
                           (y-.35,.7),facecolor=COLORS[state])
        if state=='cruise':
            f=next((f for f in sim.flights if f.aircraft==name and f.departed is not None and abs(float(f.departed-a))<1e-8),None)
            if f:
                axes[0].text(float(a+b)/2*c.time_scale_min/60,y,f'{f.route.origin}→{f.route.dest}',
                             ha='center',va='center',color='white',fontsize=6)
    axes[0].set(yticks=range(len(names)),yticklabels=names,xlim=(0,hours),ylabel='Aircraft',title='Aircraft rotations and ground activities')
    for name in names[:3]:
        events=[]
        for n,t,before,after,fid in sim.log.energy_used:
            if n==name: events.append((float(t),before,after))
        for n,p,a,b,before,after in sim.log.charges:
            if n==name:
                actual_end=min(float(b),float(sim.env.now))
                actual_soc=before+(after-before)*(actual_end-float(a))/float(b-a)
                events.extend([(float(a),before,before),(actual_end,actual_soc,actual_soc)])
        x=[0];y=[c.initial_soc*100]
        for t,before,after in sorted(events):
            x.extend([t*c.time_scale_min/60]*2);y.extend([before*100,after*100])
        axes[1].plot(x,y,label=name)
    axes[1].axhline(c.reserve_soc*100,color='#c65b47',ls='--',label='Reserve')
    axes[1].set(xlabel='Elapsed hours',ylabel='Battery (%)',xlim=(0,hours),ylim=(0,105))
    axes[1].legend(ncol=4,fontsize=8)
    finish(fig,'02_rotations_and_battery.png')


def resources(sim):
    r=measurements(sim); ports=list(sim.ports)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8)); x=np.arange(len(ports))
    axes[0].bar(x-.18,[100*r['pad_utilisation'][p] for p in ports],.36,label='Pads')
    axes[0].bar(x+.18,[100*r['charger_utilisation'][p] for p in ports],.36,label='Chargers')
    labels=[f"{p}\n{sim.config.capacities(p)['chargers']} charger(s)" for p in ports]
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Busy fraction of installed capacity (%)',ylim=(0,100))
    axes[0].legend()
    axes[1].bar(x-.18,[r['aircraft_queues'][p]['L'] for p in ports],.36,color='#2677ad',label='Mean actual N')
    axes[1].bar(x+.18,[r['aircraft_queues'][p]['mean_reservations'] for p in ports],.36,color='#a77db5',label='Mean future permits R')
    axes[1].plot(x,[sim.ports[p].queue.capacity for p in ports],'ko--',label='K per port')
    axes[1].set(xticks=x,xticklabels=ports,ylabel='Aircraft / reserved places',title='Actual queue and future permits are separate')
    axes[1].legend(fontsize=8)
    finish(fig,'03_resources.png')


def benchmark(data):
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key,label in zip(axes,('blocking','Lq','Wq'),('Blocked fraction','Mean queue length Lq','Mean queue wait Wq (min)')):
        for K in (1,4,8):
            x=np.array([.5,1,1.5]); stats=[interval([r['metrics'][key] for r in data['benchmark_runs'] if r['K']==K and r['rho']==rho]) for rho in x]
            means=np.array([r['mean'] for r in stats]); errors=np.array([r['high']-r['mean'] for r in stats])
            line=ax.errorbar(x,means,yerr=errors,fmt='o',capsize=3,label=f'Simulation K={K}')
            ax.plot(x,[theory(rho,1,K)[key] for rho in x],color=line[0].get_color(),ls='--')
        ax.set(xlabel='Offered load ρ = λ/μ',ylabel=label)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same aircraft queue under Poisson input: dots = replication means ± 95% CI; dashed = exact theory',fontsize=11)
    finish(fig,'04_mm1k_validation.png')


def transient_plot():
    x=np.linspace(0,25,101); p=transient(1,1,4,x); exact=theory(1,1,4)['pn']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for n in range(5): axes[0].plot(x,p[:,n],label=f'P{n}(t)')
    axes[0].axhline(.2,ls=':',color='black')
    axes[0].set(xlabel='Elapsed minutes',ylabel='State probability',title='Starts empty; fixed λ = μ = 1/min, K = 4')
    axes[0].legend(ncol=2,fontsize=8)
    axes[1].bar(range(5),exact,color='#2677ad')
    axes[1].set(xlabel='Aircraft in pad system n (includes service)',ylabel='Stationary probability',ylim=(0,1),title='Long-run distribution: each state 1/5')
    finish(fig,'05_transient.png')


def capacity(data):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    wait_target=data['config']['wait_target_min']
    for column,(group,label) in enumerate((('demand','Passenger demand (people/hour)'),('fleet','Number of aircraft'))):
        levels=sorted({r['level'] for r in data['network_runs'] if r['group']==group})
        for key,legend in (('served_fraction','Completed / all arrivals'),('timely_fraction',f'Departed within {wait_target:g} min / all arrivals')):
            stats=[interval([r['metrics'][key] for r in data['network_runs'] if r['group']==group and r['level']==v]) for v in levels]
            axes[0,column].errorbar(levels,[s['mean'] for s in stats],yerr=[s['high']-s['mean'] for s in stats],marker='o',capsize=3,label=legend)
        for target in sorted({data['config']['served_target'],data['config']['timely_target']}):
            axes[0,column].axhline(target,color='gray',ls=':',label=f'Illustrative {target:.0%} target')
        axes[0,column].set(xlabel=label,ylabel='Fraction of passenger cohort',ylim=(-.05,1.08))
        for key,legend in (('service_load_factor','Load factor: passenger flights'),('load_factor','Load factor: including repositioning')):
            stats=[interval([r['metrics'][key] for r in data['network_runs'] if r['group']==group and r['level']==v]) for v in levels]
            axes[1,column].errorbar(levels,[s['mean'] for s in stats],yerr=[s['high']-s['mean'] for s in stats],marker='o',capsize=3,label=legend)
        axes[1,column].set(xlabel=label,ylabel='Occupied / offered seats',ylim=(-.05,1.08))
    axes[0,0].legend(fontsize=7); axes[1,0].legend(fontsize=7)
    fig.suptitle('Finite-day capacity studies; seats and timetable held fixed',fontsize=12)
    finish(fig,'06_capacity.png')


def sensitivity(data):
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for ax,group,label in zip(axes,('K','delay'),('Aircraft pad capacity K','Mean extra ground delay (min)')):
        levels=sorted({r['level'] for r in data['network_runs'] if r['group']==group})
        for key,legend in (('served_fraction','Completed / all arrivals'),('timely_fraction','Departed within target / all arrivals')):
            stats=[interval([r['metrics'][key] for r in data['network_runs'] if r['group']==group and r['level']==v]) for v in levels]
            ax.errorbar(levels,[s['mean'] for s in stats],yerr=[s['high']-s['mean'] for s in stats],marker='o',capsize=3,label=legend)
        ax.set(xlabel=label,ylabel='Fraction',ylim=(-.05,1.08))
    axes[0].legend(fontsize=8)
    finish(fig,'07_K_and_delay.png')


def reliability(sim):
    c=sim.config
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    waits=sorted(float(p.departed-p.arrived)*c.time_scale_min for p in sim.passengers if p.departed is not None)
    axes[0].step(waits,np.arange(1,len(waits)+1)/len(sim.passengers),where='post',color='#2677ad')
    axes[0].axvline(c.wait_target_min,color='gray',ls=':')
    axes[0].set(xlabel='Arrival-to-departure wait (min)',ylabel='Departed by this wait / ALL arrivals',ylim=(0,1.03),title='Missing mass = passengers not yet departed')
    for route in list(sim.routes)[:3]:
        times=sorted(float(f.departed)*c.time_scale_min for f in sim.flights if f.kind=='service' and
                     f.departed is not None and (f.route.origin,f.route.dest)==route)
        if len(times)>1:
            axes[1].plot(times[1:],np.diff(times),'o-',label='→'.join(route),markersize=3)
    axes[1].axhline(c.timetable_interval_min,color='gray',ls=':',label='Default scheduled interval')
    axes[1].set(xlabel='Departure time (min)',ylabel='Actual gap between passenger flights (min)',title='Cancelled services and delay change actual gaps')
    axes[1].legend(fontsize=8)
    finish(fig,'08_passenger_reliability.png')


def aircraft_queues(sim):
    """Actual aircraft occupancy and realised FIFO waits; no Poisson fit assumed."""
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    port=next(iter(sim.ports));q=sim.ports[port].queue;c=sim.config
    end=float(sim.env.now*c.time_scale_min)/60
    for history,label in ((q.history,'Actual aircraft N'),(q.reservation_history,'Future permits R')):
        axes[0].step([float(t*c.time_scale_min)/60 for t,n in history]+[end],
                     [n for t,n in history]+[history[-1][1]],where='post',label=label)
    axes[0].axhline(q.capacity,color='gray',ls=':',label='K; policy enforces N + R ≤ K')
    axes[0].set(xlabel='Elapsed hours',ylabel='Count',title=f'Port {port}: arrival into queue differs from booking')
    axes[0].legend(fontsize=8)
    for operation in ('takeoff','landing'):
        tickets=[t for v in sim.ports.values() for t in v.queue.tickets
                 if t.operation==operation and t.started is not None]
        axes[1].scatter([float(t.arrived*c.time_scale_min)/60 for t in tickets],
                        [float((t.started-t.arrived)*c.time_scale_min) for t in tickets],
                        s=12,alpha=.65,label=operation)
    axes[1].set(xlabel='Time aircraft joined pad queue (h)',ylabel='Actual aircraft queue wait (min)',
                title='All ports; excludes preparation and charger waiting')
    axes[1].legend(fontsize=8)
    finish(fig,'09_aircraft_pad_queues.png')
