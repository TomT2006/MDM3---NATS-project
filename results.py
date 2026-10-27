"""Convert event records to physical measurements; never changes the simulation."""
from collections import Counter, defaultdict
import math
import numpy as np


def measurements(sim, start_min=0, end_min=None):
    """Passenger cohort arrives within window; follow it to the observed run end.

    Resource utilisation and flights use the window itself. Default: whole run.
    Wait percentiles exclude passengers not yet departed: report their count too.
    """
    c, end = sim.config, sim.env.now
    left = c.time(start_min)
    right = end if end_min is None else c.time(end_min)
    if not 0 <= left < right <= end + 1e-9:
        raise ValueError("Measurement window must be inside the completed run.")
    minutes = float((right - left) * c.time_scale_min)
    people = [p for p in sim.passengers if left <= p.arrived < right]
    departed = [p for p in people if p.departed is not None]
    completed = [p for p in people if p.completed is not None]
    waits = [float((p.departed - p.arrived) * c.time_scale_min) for p in departed]
    timely = sum(w <= c.wait_target_min + 1e-8 for w in waits)
    flights = [f for f in sim.flights if f.departed is not None and left <= f.departed < right]
    service = [f for f in flights if f.kind == 'service']
    scheduled = [f for f in sim.flights if f.kind == 'service' and left <= f.scheduled < right]
    delays = [float((f.departed - f.scheduled) * c.time_scale_min) for f in service]
    state_time = Counter()
    port_time = {p: Counter() for p in sim.ports}
    for aircraft, state, port, a, b in sim.log.intervals(right, left):
        state_time[state] += float((b - a) * c.time_scale_min)
        if port is not None:
            port_time[port][state] += float((b - a) * c.time_scale_min)
    grid_kwh = 0
    for aircraft, port, a, b, before, after in sim.log.charges:
        overlap = max(0, min(right, b) - max(left, a))
        grid_kwh += overlap / (b - a) * (after - before) * c.battery_kwh / c.charging_efficiency
    pad_load, charger_load, request_load = {}, {}, {}
    for p, v in sim.ports.items():
        pad_load[p] = (port_time[p]['takeoff'] + port_time[p]['landing']) / minutes
        charger_load[p] = port_time[p]['charging'] / (minutes * v.chargers.capacity) if v.chargers else 0
        request_load[p] = v.queue.statistics(left, right)
    n = len(people)
    result = dict(arrivals=n, completed=len(completed), not_completed=n-len(completed),
                  departed=len(departed), not_departed=n-len(departed),
                  unassigned=sum(p.flight is None for p in people),
                  assigned_not_departed=sum(p.flight is not None and p.departed is None for p in people),
                  departed_not_completed=sum(p.departed is not None and p.completed is None for p in people),
                  served_fraction=len(completed)/n if n else None,
                  timely_fraction=timely/n if n else None,
                  mean_wait_min=float(np.mean(waits)) if waits else None,
                  p95_wait_min=float(np.percentile(waits,95)) if waits else None,
                  flights=len(flights), service_flights=len(service),
                  reposition_flights=sum(f.kind == 'reposition' for f in flights),
                  flight_rate_per_hour=len(flights)/(minutes/60),
                  passengers_per_hour=len([p for p in sim.passengers if p.completed is not None and left <= p.completed < right])/(minutes/60),
                  load_factor=sum(len(f.passengers) for f in flights)/(c.seats*len(flights)) if flights else 0,
                  service_load_factor=sum(len(f.passengers) for f in service)/(c.seats*len(service)) if service else 0,
                  mean_delay_min=float(np.mean(delays)) if delays else None,
                  p95_delay_min=float(np.percentile(delays,95)) if delays else None,
                  reassignments=sum(f.preferred_aircraft is not None and f.aircraft != f.preferred_aircraft for f in service),
                  scheduled_status=dict(Counter(f.status for f in scheduled)),
                  cancellation_reasons=dict(Counter(f.cancel_reason for f in scheduled if f.status == 'cancelled')),
                  state_fraction={k: v/(minutes*c.fleet) for k,v in state_time.items()},
                  pad_utilisation=pad_load, charger_utilisation=charger_load,
                  aircraft_queues=request_load, admission_deferrals=sum(left <= t < right for t,_,_ in sim.log.deferrals),
                  grid_kwh=float(grid_kwh), observed_minutes=minutes,
                  flying_fraction=sum(state_time[x] for x in ('takeoff','cruise','landing'))/(minutes*c.fleet),
                  airborne_wait_fraction=state_time['arrival_pad_wait']/(minutes*c.fleet),
                  charging_fraction=state_time['charging']/(minutes*c.fleet))
    decisions = [d for d in sim.log.safety_checks if start_min <= d['checked_min'] < float(right*c.time_scale_min)]
    rejected = [d for d in decisions if not d['allowed']]
    committed = [d for d in decisions if d['committed']]
    holds = list(sim.log.holding_used)
    for a in sim.aircraft:
        if a.holding is not None:
            start, before, fid = a.holding
            holds.append((a.name, start, end, before, a.soc, fid))
    holding_kwh = sum(float(max(0, min(right,b)-max(left,a))*c.time_scale_min)*c.holding_kw/60
                      for name,a,b,before,after,fid in holds)
    result.update(
        dispatches_approved=len(committed),
        rejected_candidate_pairs=len({(d['flight'],d['aircraft']) for d in rejected}),
        rejected_service_flights=len({d['flight'] for d in rejected if d['flight'].startswith('F')}),
        safety_rejection_reasons=dict(Counter(d['reason'] for d in rejected)),
        nominal_rotation_conflicts=sum(d['next_nominal_feasible'] is False for d in committed),
        holding_kwh=holding_kwh,
        holding_allowance_exceedances=sum(
            float((b-a)*c.time_scale_min) > sim.safety.hold_minutes[next(f.route.dest for f in sim.flights if f.name==fid)]+1e-8
            for name,a,b,before,after,fid in holds),
        safety_failures=len(sim.log.failures),
        stopped_for_safety=sim.failure_event.triggered,
        minimum_soc=min([c.initial_soc]+[a.soc for a in sim.aircraft]+
                        [after for name,t,before,after,fid in sim.log.energy_used]+
                        [after for name,a,b,before,after,fid in holds]),
        all_dispatches_approved=all(f.safety is not None and f.safety.allowed for f in flights))
    result['passes_targets'] = not sim.failure_event.triggered and n > 0 and result['served_fraction'] >= c.served_target and result['timely_fraction'] >= c.timely_target
    if c.costs is not None:
        costs = c.costs
        result['scenario_contribution'] = (len([p for p in sim.passengers if p.completed is not None and left <= p.completed < right])*costs['fare']
            - len(flights)*costs['per_flight'] - grid_kwh*costs['per_grid_kwh']
            - c.fleet*(minutes/60)*costs['per_aircraft_hour'])
    return result


def audit(sim):
    """Physical/conservation checks, independent of the scheduler's decisions."""
    c, eps = sim.config, 1e-7
    for t, allocated in sim.log.allocations:
        assert sum(allocated.values()) == c.fleet, 'Lost aircraft stand claim'
        assert all(0 <= n <= sim.ports[p].stands for p,n in allocated.items()), 'Stand claims exceed capacity'
    chargers, routes = defaultdict(list), defaultdict(list)
    names = {a.name for a in sim.aircraft}
    flights = {f.name:f for f in sim.flights}
    aircraft_intervals = defaultdict(list)
    for port, v in sim.ports.items():
        assert v.queue.audit()
        for t in v.queue.tickets:
            assert not t.blocked, 'Operating aircraft was discarded'
            assert t.aircraft in names and t.flight in flights, 'Invented aircraft or flight'
            f = flights[t.flight]
            assert f.aircraft == t.aircraft, 'Queue aircraft differs from assignment'
            aircraft_intervals[t.aircraft].append((t.arrived, t.completed or sim.env.now))
            if t.operation == 'takeoff':
                assert port == f.route.origin
                assert t.arrived + eps >= f.scheduled, 'Premature departure admission'
                assert t.completed == f.departed
            else:
                assert port == f.route.dest and f.departed is not None
                assert abs(t.arrived-f.departed-f.route.flight_time) < eps, 'Landing before flight'
                assert t.completed == f.landed
        for (fid, operation), aircraft in v.queue.reservations.items():
            assert fid in flights and aircraft == flights[fid].aircraft, 'Orphan permit'
    for intervals in aircraft_intervals.values():
        intervals.sort()
        assert all(b <= x+eps for (a,b),(x,y) in zip(intervals,intervals[1:])), 'Aircraft in two queues'
    for f in sim.flights:
        if f.departed is not None:
            routes[f.route.origin, f.route.dest].append(f.departed)
    for departures in routes.values():
        departures.sort()
        assert all(b-a+eps >= c.time(c.headway_min) for a,b in zip(departures,departures[1:])), 'Headway violation'
    for name, port, a, b, before, after in sim.log.charges:
        assert sim.ports[port].chargers is not None, 'Charging at unequipped port'
        assert 0 <= before <= after <= 1+eps, 'Invalid charge'
        chargers[port].extend([(a,1),(b,-1)])
    for port, events in chargers.items():
        count = 0
        for t, delta in sorted(events):
            count += delta
            assert 0 <= count <= sim.ports[port].chargers.capacity, 'Charger overbooking'
    flights = {f.name:f for f in sim.flights}
    for name, t, before, after, fid in sim.log.energy_used:
        f = flights[fid]
        assert before+eps >= sim.required(f.route), 'Unsafe energy dispatch'
        assert abs(before-after-f.route.energy) < eps, 'Flight energy debit'
        assert after+eps >= c.reserve_soc, 'Reserve violation'
    for name, start, end, before, after, fid in sim.log.holding_used:
        expected = float((end-start)*c.time_scale_min)*c.holding_kw/(60*c.battery_kwh)
        assert abs(before-after-expected) < eps, 'Holding energy debit'
        assert after+eps >= c.reserve_soc, 'Unreported reserve crossing'
    for f in sim.flights:
        if f.aircraft is not None:
            assert f.safety is not None and f.safety.allowed, 'Dispatch without approval'
            assert f.safety.soc+eps >= f.safety.required_soc
            assert f.safety.charger_wait_min <= c.max_charger_wait_min+eps
    for a in sim.aircraft:
        added = sum((after-before)*max(0,min(1,(sim.env.now-start)/(end-start)))
                    for name,port,start,end,before,after in sim.log.charges if name == a.name)
        used = sum(before-after for name,t,before,after,fid in sim.log.energy_used if name == a.name)
        used += sum(before-after for name,start,end,before,after,fid in sim.log.holding_used if name == a.name)
        if a.holding is not None:
            used += a.holding[1]-a.soc
        assert abs(a.soc-(c.initial_soc+added-used)) < eps, 'Energy balance'
        assert c.reserve_soc-eps <= a.soc <= 1+eps, 'Invalid current battery'
    assigned = [p.name for f in sim.flights for p in f.passengers]
    assert len(assigned) == len(set(assigned)), 'Passenger assigned twice'
    for f in sim.flights:
        assert len(f.passengers) <= c.seats, 'Too many passengers'
        for p in f.passengers:
            assert (p.origin,p.dest) == (f.route.origin,f.route.dest) and p.flight == f.name, 'Wrong destination'
            times = [t for t in (p.arrived,p.boarded,p.departed,p.completed) if t is not None]
            assert times == sorted(times), 'Passenger event order'
    for v in sim.ports.values():
        assert all(0 <= n <= v.queue.capacity for t,n in v.queue.history), 'K exceeded'
    total = sum(b-a for name,state,port,a,b in sim.log.intervals(sim.env.now))
    assert abs(total-c.fleet*sim.env.now) < eps*max(1,c.fleet), 'Aircraft time not conserved'
    return True


def waiting_trace(sim, times_min):
    """Passenger waiting, charger waiting, actual pad N and future permits R."""
    values = []
    for minute in times_min:
        t = sim.config.time(minute)
        passenger = sum(p.arrived <= t and (p.boarded is None or p.boarded > t) for p in sim.passengers)
        charging = sum(state == 'charger_wait' and a <= t < b for _,state,port,a,b in sim.log.intervals(sim.env.now))
        operations = sum(next(n for at,n in reversed(v.queue.history) if at <= t) for v in sim.ports.values())
        reserved = sum(next(n for at,n in reversed(v.queue.reservation_history) if at <= t) for v in sim.ports.values())
        values.append((passenger,charging,operations,reserved))
    return np.array(values)
