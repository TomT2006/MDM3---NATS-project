"""Energy dispatch gate and charging forecast; not an aircraft certification model.

No random future service samples are read by this planner. A quantile is an
allowance, not an upper bound. The simulator stops if holding uses the reserve.
"""
from dataclasses import dataclass, asdict
from math import inf
from scipy.stats import erlang


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str
    aircraft: str
    flight: str
    checked_min: float
    soc: float
    required_soc: float
    margin_soc: float
    charger: str
    recovery_path: tuple
    holding_allowance_min: float
    charger_arrival_min: float
    charger_wait_min: float
    charged_ready_min: float
    next_nominal_flight: str = None
    next_nominal_feasible: bool = None


class SafetyPolicy:
    def __init__(self, sim):
        self.sim = sim
        c = sim.config
        # At most K-1 operations precede a newly admitted aircraft in FIFO.
        self.hold_minutes = {
            p: float(erlang.ppf(c.holding_quantile, v.queue.capacity-1,
                               scale=c.capacities(p)['pad_service_mean_min']))
            if v.queue.capacity > 1 else 0.0 for p, v in sim.ports.items()}
        self.last_reason = {}
        self.tentative = []

    def leg_budget(self, route):
        c = self.sim.config
        return ((1+c.mission_energy_margin)*route.energy +
                c.holding_kw*self.hold_minutes[route.dest]/(60*c.battery_kwh))

    def recovery_path(self, port):
        path = [port]
        while self.sim.ports[port].chargers is None:
            port = self.sim.escape_next[port]
            path.append(port)
        return tuple(path)

    def charge_minutes(self, port, soc):
        c = self.sim.config
        return 60*max(0, c.charge_target_soc-soc)*c.battery_kwh / (
            self.sim.ports[port].charger_kw*c.charging_efficiency)

    def charge_forecast(self, port, arrival_min, soc, exclude):
        """FCFS workload forecast: current jobs, then committed inbound plans.

        Inbound plans are conservatively placed ahead; actual requests are still
        FCFS on physical arrival. This is not a reserved slot or a guaranteed ETA.
        """
        s, c = self.sim, self.sim.config
        now = float(s.env.now*c.time_scale_min)
        v = s.ports[port]
        slots = [now]*v.chargers.capacity
        accounted = {exclude}
        active = [a for a in s.aircraft if a.port == port and a.charging is not None]
        for i, a in enumerate(active):
            slots[i] = float(a.charging[1]*c.time_scale_min)
            accounted.add(a.name)
        jobs = []
        for a in v.charge_order:
            if a.name not in accounted:
                jobs.append((now, self.charge_minutes(port, a.soc)))
                accounted.add(a.name)
        for f in s.flights + self.tentative:
            d = f.safety
            if (f.status in ('assigned', 'flying') and d is not None and
                    d.charger == port and f.aircraft not in accounted):
                # An earlier commitment is kept in the workload estimate even
                # if its planned arrival is later than this candidate's arrival.
                jobs.append((max(now, d.charger_arrival_min),
                             max(0, d.charged_ready_min-d.charger_arrival_min-d.charger_wait_min)))
                accounted.add(f.aircraft)
        for eta, duration in jobs:
            i = min(range(len(slots)), key=slots.__getitem__)
            slots[i] = max(slots[i], eta)+duration
        start = max(arrival_min, min(slots))
        return start-arrival_min, start+self.charge_minutes(port, soc)

    def assess(self, aircraft, flight, passenger_count=0):
        s, c = self.sim, self.sim.config
        r = flight.route
        now = float(s.env.now*c.time_scale_min)
        path = self.recovery_path(r.dest)
        required = s.required(r)
        margin = aircraft.soc-required
        first_ready = (max(now+passenger_count*c.board_min_per_passenger+c.taxi_out_min,
                           float(flight.scheduled*c.time_scale_min),
                           float(r.last_departure*c.time_scale_min)+c.headway_min) +
                       c.capacities(r.origin)['pad_service_mean_min'] +
                       float(r.flight_time*c.time_scale_min)+self.hold_minutes[r.dest] +
                       c.capacities(r.dest)['pad_service_mean_min']+c.taxi_in_min+
                       passenger_count*c.unload_min_per_passenger+c.ground_delay_mean_min)
        eta = first_ready
        for a, b in zip(path, path[1:]):
            leg = s.routes[a, b]
            eta += (c.taxi_out_min+c.capacities(a)['pad_service_mean_min']+
                    float(leg.flight_time*c.time_scale_min)+self.hold_minutes[b]+
                    c.capacities(b)['pad_service_mean_min']+c.taxi_in_min)
        charger_soc = aircraft.soc-(required-c.reserve_soc)
        wait, ready = self.charge_forecast(path[-1], eta, charger_soc, aircraft.name)
        reason = ('insufficient energy' if margin < -1e-9 else
                  'charging queue forecast exceeds limit' if wait > c.max_charger_wait_min+1e-9 else 'accepted')
        decision = SafetyDecision(reason == 'accepted', reason, aircraft.name, flight.name,
                                  now, aircraft.soc, required, margin, path[-1], path,
                                  self.hold_minutes[r.dest], eta, wait, ready)
        # A nominal rotation is soft. A conflict is reported; the normal
        # scheduler may swap aircraft, delay or cancel that later service.
        future = [f for f in s.flights if f is not flight and f.status == 'pending' and
                  f.preferred_aircraft == aircraft.name and f.scheduled > flight.scheduled]
        if future:
            next_f = min(future, key=lambda f: f.scheduled)
            remaining = aircraft.soc-self.leg_budget(r)
            can_continue = remaining+1e-9 >= s.required(next_f.route)
            charge_first = (s.ports[r.dest].chargers is not None and
                            (remaining < c.charge_trigger_soc or not can_continue))
            location = path[-1] if charge_first or not can_continue else r.dest
            available = ready if charge_first or not can_continue else first_ready
            decision.next_nominal_flight = next_f.name
            decision.next_nominal_feasible = (
                location == next_f.route.origin and
                available <= float(next_f.scheduled*c.time_scale_min)+c.max_assignment_delay_min)
        return decision

    def record(self, decision, committed=False):
        key = (decision.flight, decision.aircraft)
        # Rejected retries are recorded only when the reason changes.
        if committed or (not decision.allowed and self.last_reason.get(key) != decision.reason):
            self.sim.log.safety_checks.append(dict(asdict(decision), committed=committed))
        self.last_reason[key] = decision.reason

