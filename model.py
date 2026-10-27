"""Centralised finite-fleet simulator with real aircraft pad queues.

Dimensionless event times and battery fractions; physical inputs in Config.
K counts actual waiting/in-service aircraft. Separate permits protect future
landings. Exponential pad service is exact; Poisson aircraft input is an
approximation. See MODEL_AND_MATH.txt for the precise boundary.
"""
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite, inf
import random
import simpy
from queues import AircraftPadQueue
from safety import SafetyPolicy


@dataclass
class ScenarioConfig:
    port_names: tuple = ("A", "B", "C", "D")
    fleet: int = 8
    seats: int = 4                         # One common aircraft type
    stands_per_port: int = 4
    chargers_per_port: int = 1             # Zero is allowed
    pads_per_port: int = 1
    queue_capacity: int = 4               # K: actual aircraft including pad service
    port_overrides: dict = field(default_factory=dict)
    initial_counts: dict = None
    initial_soc: float = 0.9
    routes_km: dict = None                 # None: every directed pair, 30 km
    default_distance_km: float = 30
    speed_kmh: float = 200                 # Scenario assumption, not measured
    headway_min: float = 3
    taxi_out_min: float = 1
    pad_service_mean_min: float = 1        # Exponential, for either pad operation
    taxi_in_min: float = 1
    board_min_per_passenger: float = 1
    unload_min_per_passenger: float = 1
    battery_kwh: float = 100               # Illustrative energy model
    vertical_energy_kwh: float = 8
    cruise_kwh_per_km: float = 0.4
    charger_kw: float = 120
    charging_efficiency: float = 1
    reserve_soc: float = 0.15
    charge_trigger_soc: float = 0.6
    charge_target_soc: float = 0.9
    holding_kw: float = 30                # Synthetic airborne holding power
    holding_quantile: float = 0.99        # Per-landing queue allowance, not a guarantee
    mission_energy_margin: float = 0.10   # Extra planned energy, not an automatic debit
    max_charger_wait_min: float = 60      # Operational forecast limit
    total_demand_per_hour: float = 36
    demand_per_hour: dict = None           # Optional directed-route rates
    demand_mode: str = "poisson"           # Or regular, for hand checks
    timetable_interval_min: float = 30
    service_intervals_min: dict = field(default_factory=dict)
    first_departure_min: float = 10
    max_assignment_delay_min: float = 30   # Deadline to COMMIT, not a bound on random delay
    decision_interval_min: float = 0.5
    ground_delay_mean_min: float = 0       # Exponential extra delay AFTER unloading
    demand_stop_min: float = 600           # Then 120 min to observe outstanding trips
    horizon_min: float = 720
    time_scale_min: float = 10
    length_scale_km: float = 30
    seed: int = 42
    wait_target_min: float = 60            # Illustrative service criteria
    served_target: float = 0.95
    timely_target: float = 0.95
    costs: dict = None                     # Optional, explicit scenario costs

    def time(self, minutes):
        # Exact ratios keep simultaneous events simultaneous when units change.
        return Fraction(str(minutes)) / Fraction(str(self.time_scale_min))

    def capacities(self, port):
        defaults = dict(stands=self.stands_per_port, chargers=self.chargers_per_port,
                        pads=self.pads_per_port, K=self.queue_capacity, charger_kw=self.charger_kw,
                        pad_service_mean_min=self.pad_service_mean_min)
        return {**defaults, **self.port_overrides.get(port, {})}

    def network(self):
        return dict(self.routes_km) if self.routes_km is not None else {
            (a, b): self.default_distance_km for a in self.port_names for b in self.port_names if a != b}

    def demand(self):
        routes = self.network()
        return dict(self.demand_per_hour) if self.demand_per_hour is not None else {
            r: self.total_demand_per_hour / len(routes) for r in routes}

    def energy(self, distance_km):
        return (self.vertical_energy_kwh + self.cruise_kwh_per_km * distance_km) / self.battery_kwh

    def validate(self):
        if len(self.port_names) < 2 or len(set(self.port_names)) != len(self.port_names):
            raise ValueError("Use at least two distinct vertiports.")
        for n in (self.fleet, self.seats):
            if type(n) is not int or n < 1:
                raise ValueError("Fleet and seats must be positive integers.")
        positive = (self.speed_kmh, self.default_distance_km, self.battery_kwh, self.charger_kw,
                    self.timetable_interval_min, self.decision_interval_min, self.horizon_min,
                    self.time_scale_min, self.length_scale_km, self.pad_service_mean_min)
        nonnegative = (self.headway_min, self.taxi_out_min, self.taxi_in_min, self.first_departure_min,
                       self.board_min_per_passenger, self.unload_min_per_passenger,
                       self.vertical_energy_kwh, self.cruise_kwh_per_km, self.total_demand_per_hour,
                       self.max_assignment_delay_min, self.ground_delay_mean_min, self.wait_target_min,
                       self.holding_kw, self.mission_energy_margin, self.max_charger_wait_min)
        if any(not isfinite(x) or x <= 0 for x in positive) or any(not isfinite(x) or x < 0 for x in nonnegative):
            raise ValueError("Durations, energy and rates must have valid finite signs.")
        if not 0 < self.demand_stop_min <= self.horizon_min or not isfinite(self.demand_stop_min):
            raise ValueError("Demand must stop within the finite observation horizon.")
        if not (0 <= self.reserve_soc < self.charge_target_soc <= 1 and
                self.reserve_soc <= self.initial_soc <= 1 and
                0 <= self.charge_trigger_soc <= self.charge_target_soc):
            raise ValueError("Invalid charge fractions or reserve.")
        if not 0 < self.charging_efficiency <= 1 or not all(0 <= x <= 1 for x in (self.served_target, self.timely_target)):
            raise ValueError("Efficiency and service targets are fractions.")
        if not 0 < self.holding_quantile < 1 or self.mission_energy_margin > 1:
            raise ValueError("Holding quantile must be between 0 and 1; energy margin between 0 and 1.")
        if self.demand_mode not in {"poisson", "regular"} or type(self.seed) is not int:
            raise ValueError("Use poisson/regular demand and an integer seed.")
        if set(self.port_overrides) - set(self.port_names):
            raise ValueError("Unknown port in overrides.")
        for p in self.port_names:
            c = self.capacities(p)
            if set(c) != {"stands", "chargers", "pads", "K", "charger_kw", "pad_service_mean_min"}:
                raise ValueError("Port overrides: stands, chargers, pads, K, charger_kw, pad_service_mean_min.")
            for k in ("stands", "pads", "K", "chargers"):
                if type(c[k]) is not int or c[k] < (0 if k == "chargers" else 1):
                    raise ValueError("Invalid port capacity.")
            if c['pads'] != 1:
                raise ValueError('This model uses one pad per port; multiple pads require M/M/c/K.')
            if not isfinite(c['pad_service_mean_min']) or c['pad_service_mean_min'] <= 0:
                raise ValueError('Pad service mean must be positive and finite.')
            if c['K'] < c['pads'] or not isfinite(c['charger_kw']) or c['charger_kw'] <= 0:
                raise ValueError("K must include all pads; charging power must be positive.")
        if self.fleet > sum(self.capacities(p)["stands"] for p in self.port_names):
            raise ValueError("All aircraft initially occupy stands; fleet exceeds them.")
        routes = self.network()
        if not routes:
            raise ValueError("At least one route is needed.")
        for (a, b), d in routes.items():
            if a not in self.port_names or b not in self.port_names or a == b or not isfinite(d) or d <= 0:
                raise ValueError("Invalid route or distance.")
        if any(not any(a == p for a, b in routes) for p in self.port_names):
            raise ValueError("Every port needs an outgoing route.")
        for r, value in self.demand().items():
            if r not in routes or not isfinite(value) or value < 0:
                raise ValueError("Demand needs a valid direct route and nonnegative rate.")
        if any(r not in routes or not isfinite(v) or v <= 0 for r, v in self.service_intervals_min.items()):
            raise ValueError("Invalid timetable override.")
        if self.costs is not None:
            keys = {"fare", "per_flight", "per_grid_kwh", "per_aircraft_hour"}
            if set(self.costs) != keys or any(not isfinite(v) or v < 0 for v in self.costs.values()):
                raise ValueError("Provide all four nonnegative scenario costs; see README.")


@dataclass
class Route:
    origin: str
    dest: str
    flight_time: float
    energy: float
    last_departure: float = -inf
    departure_pending: bool = False


@dataclass
class Passenger:
    name: int
    origin: str
    dest: str
    arrived: float
    flight: str = None
    boarded: float = None
    departed: float = None
    completed: float = None


@dataclass
class Flight:
    name: str
    route: Route
    scheduled: float                       # Planned cruise entry, dimensionless
    preferred_aircraft: str = None         # Soft nominal assignment, not a commitment
    kind: str = "service"
    status: str = "pending"
    aircraft: str = None
    passengers: list = field(default_factory=list)
    departed: float = None
    landed: float = None
    completed: float = None
    cancel_reason: str = None
    safety: object = None                 # Dispatch decision, including charging forecast


class Vertiport:
    def __init__(self, env, capacities, seed, time_scale):
        self.stands, self.allocated = capacities['stands'], 0
        self.chargers = simpy.Resource(env, capacities['chargers']) if capacities['chargers'] else None
        self.charge_order = []            # FCFS requests, registered before processes run
        self.charger_kw = capacities['charger_kw']
        self.queue = AircraftPadQueue(env, capacities['K'], 1/capacities['pad_service_mean_min'], seed, time_scale)


class Recorder:
    def __init__(self):
        self.states, self.allocations = {}, []
        self.charges, self.energy_used, self.deferrals = [], [], []
        self.holding_used, self.safety_checks, self.failures = [], [], []

    def state(self, aircraft, time, state, port):
        self.states.setdefault(aircraft, []).append((time, state, port))

    def intervals(self, end, start=0):
        for aircraft, history in self.states.items():
            for i, (time, state, port) in enumerate(history):
                following = history[i + 1][0] if i + 1 < len(history) else end
                left, right = max(time, start), min(following, end)
                if right > left:
                    yield aircraft, state, port, left, right


class Aircraft:
    def __init__(self, sim, name, port):
        self.sim, self.name, self.port = sim, name, port
        self.busy, self.base_soc, self.charging = False, sim.config.initial_soc, None
        self.clearance, self.holding = None, None
        self.set_state("idle")

    @property
    def soc(self):
        if self.holding is not None:
            start, before, fid = self.holding
            c = self.sim.config
            return before-float((self.sim.env.now-start)*c.time_scale_min)*c.holding_kw/(60*c.battery_kwh)
        if self.charging is None:
            return self.base_soc
        start, end, before, after = self.charging
        return before + (after - before) * min(1, (self.sim.env.now - start) / (end - start))

    def set_state(self, state):
        self.sim.log.state(self.name, self.sim.env.now, state, self.port)

    def activity(self, state, duration):
        self.set_state(state)
        yield self.sim.env.timeout(duration)

    def start_charge(self):
        self.busy = True
        self.sim.ports[self.port].charge_order.append(self)
        self.sim.env.process(self.charge())

    def finish_holding(self):
        if self.holding is not None:
            start, before, fid = self.holding
            after = self.soc
            self.sim.log.holding_used.append((self.name, start, self.sim.env.now, before, after, fid))
            self.base_soc, self.holding = after, None
        self.set_state('landing')

    def charge(self):
        s, c = self.sim, self.sim.config
        port = s.ports[self.port]
        self.set_state("charger_wait")
        with port.chargers.request() as request:
            yield request
            target = max(self.base_soc, c.charge_target_soc)
            minutes = 60 * (target - self.base_soc) * c.battery_kwh / (port.charger_kw * c.charging_efficiency)
            if minutes > 1e-9:
                start, end = s.env.now, s.env.now + c.time(minutes)
                self.charging = (start, end, self.base_soc, target)
                s.log.charges.append((self.name, self.port, *self.charging))
                yield from self.activity("charging", end - start)
                self.base_soc, self.charging = target, None
        port.charge_order.remove(self)
        self.busy = False
        self.set_state("idle")

    def fly(self, flight):
        s, c = self.sim, self.sim.config
        for p in flight.passengers:
            p.boarded = s.env.now
        yield from self.activity("boarding", c.time(len(flight.passengers) * c.board_min_per_passenger))
        self.set_state("dispatch_wait")
        yield self.clearance
        yield from self.activity("taxi_out", c.time(c.taxi_out_min))
        # Conservative route release: actual cruise entries are at least h apart.
        release = max(flight.scheduled, flight.route.last_departure + c.time(c.headway_min))
        yield from self.activity("dispatch_wait", max(0, release-s.env.now))
        self.set_state("departure_pad_wait")
        ticket = s.ports[self.port].queue.arrive((flight.name, 'takeoff'), self.name,
                                                lambda: self.set_state('takeoff'))
        yield ticket.done
        flight.departed, flight.status = s.env.now, "flying"
        flight.route.last_departure, flight.route.departure_pending = s.env.now, False
        for p in flight.passengers:
            p.departed = s.env.now
        before = self.base_soc
        assert flight.safety.allowed and before+1e-9 >= s.required(flight.route)
        self.base_soc -= flight.route.energy
        s.log.energy_used.append((self.name, s.env.now, before, self.base_soc, flight.name))
        self.port = None
        yield from self.activity("cruise", flight.route.flight_time)
        self.port = flight.route.dest
        self.set_state("arrival_pad_wait")
        self.holding = (s.env.now, self.base_soc, flight.name)
        ticket = s.ports[self.port].queue.arrive((flight.name, 'landing'), self.name,
                                                self.finish_holding)
        if c.holding_kw > 0:
            minutes = max(0, (self.base_soc-c.reserve_soc)*c.battery_kwh*60/c.holding_kw)
            reserve_time = s.env.timeout(c.time(minutes))
            yield ticket.start_event | reserve_time
            if ticket.started is None:
                s.log.failures.append(dict(time_min=float(s.env.now*c.time_scale_min),
                    aircraft=self.name, flight=flight.name, reason='holding reached protected reserve'))
                if not s.failure_event.triggered:
                    s.failure_event.succeed(s.log.failures[-1])
                return  # Run stops here, rather than continuing with depleted energy.
        yield ticket.done
        flight.landed = s.env.now
        yield from self.activity("taxi_in", c.time(c.taxi_in_min))
        yield from self.activity("unloading", c.time(len(flight.passengers) * c.unload_min_per_passenger))
        flight.completed, flight.status = s.env.now, "completed"
        for p in flight.passengers:
            p.completed = s.env.now
        # Realised only now: the controller cannot see this future disturbance.
        if c.ground_delay_mean_min:
            delay = c.time(s.delay_rng[self.name].expovariate(1 / c.ground_delay_mean_min))
            yield from self.activity("ground_delay", delay)
        self.busy = False
        self.set_state("idle")


class Scheduler:
    """Centralised, oldest-service-first heuristic; not a global optimiser."""
    def __init__(self, sim):
        self.sim = sim

    def due(self):
        s, c = self.sim, self.sim.config
        lead = c.time(c.seats * c.board_min_per_passenger + c.taxi_out_min)
        due = []
        for f in s.flights:
            if f.status != "pending":
                continue
            if s.env.now > f.scheduled + c.time(c.max_assignment_delay_min) + 1e-9:
                f.status, f.cancel_reason = "cancelled", "assignment deadline"
            elif f.scheduled <= s.env.now + lead + 1e-9:
                due.append(f)
        return sorted(due, key=lambda f: (f.scheduled, f.name))

    def requests(self):
        s, c = self.sim, self.sim.config
        due, requests = self.due(), {}
        for a in s.aircraft:
            if a.busy or s.ports[a.port].chargers is None:
                continue
            needs = [s.required(f.route) for f in due if f.route.origin == a.port]
            if a.soc < c.charge_target_soc - 1e-9 and (a.soc < c.charge_trigger_soc or
                                                      (needs and a.soc < min(needs))):
                a.start_charge()
        for f in due:
            origin = f.route.origin
            if origin in requests or f.route.departure_pending:
                continue
            passengers = [p for p in s.waiting[origin, f.route.dest] if p.flight is None][:c.seats]
            if not passengers and f.kind == "service":
                if s.env.now >= f.scheduled - 1e-9:
                    f.status, f.cancel_reason = "cancelled", "no passengers"
                continue
            ready = []
            for a in s.aircraft:
                if not a.busy and a.port == origin:
                    decision = s.safety.assess(a, f, len(passengers))
                    s.safety.record(decision)
                    if decision.allowed:
                        ready.append((a, decision))
            if ready:
                a, decision = min(ready, key=lambda pair: (
                    pair[0].name != f.preferred_aircraft, -pair[0].soc, pair[0].name))
                f.safety = decision
                requests[origin] = (a, f, passengers)
        # Energy recovery from non-charging ports, then empty moves towards unmet demand.
        for a in s.aircraft:
            if a.busy or a.port in requests:
                continue
            active = [f for f in due if f.route.origin == a.port]
            target = None
            if s.ports[a.port].chargers is None and (a.soc < c.charge_trigger_soc or
                    (active and not any(a.soc + 1e-9 >= s.required(f.route) for f in active))):
                target = s.escape_next[a.port]
            elif not active:
                shortages = [f.route.origin for f in due if f.kind == "service" and
                             any(p.flight is None for p in s.waiting[f.route.origin, f.route.dest]) and
                             not any(b.port == f.route.origin and not b.busy for b in s.aircraft) and
                             not any(g.route.dest == f.route.origin and g.status in {"assigned", "flying"}
                                     for g in s.flights)]
                reachable = [p for p in shortages if p != a.port and s.path[a.port, p] is not None]
                if reachable:
                    dest = min(reachable, key=lambda p: s.path_distance[a.port, p])
                    target = s.path[a.port, dest]
            if target is not None:
                route = s.routes[a.port, target]
                if not route.departure_pending and a.soc + 1e-9 >= s.required(route):
                    f = Flight(f"R{len(s.flights)}_{a.port}", route, s.env.now, kind="reposition")
                    decision = s.safety.assess(a, f)
                    s.safety.record(decision)
                    if decision.allowed:
                        f.safety = decision
                        requests[a.port] = (a, f, [])
        return requests

    def choose_batch(self, requests):
        s = self.sim
        for origin in requests:
            chain, seen, port = [], {}, origin
            while port in requests:
                if port in seen:
                    return chain[seen[port]:]
                seen[port] = len(chain)
                item = requests[port]
                chain.append(item)
                port = item[1].route.dest
                if s.ports[port].allocated < s.ports[port].stands:
                    return chain
        return []

    def run(self):
        s, c = self.sim, self.sim.config
        while True:
            batch = self.choose_batch(self.requests())
            if batch:
                needed = Counter(p for a, f, people in batch for p in (a.port, f.route.dest))
                full = [p for p, count in needed.items() if not s.ports[p].queue.fits(count)]
                if full:
                    # A retry is not a lost aircraft. Keep everyone on the ground.
                    s.log.deferrals.append((s.env.now, tuple(f.name for a,f,p in batch), tuple(full)))
                    batch = []
            # Refresh forecasts sequentially so one batch includes its own
            # earlier commitments in the charging workload. No time elapses.
            if batch:
                checked = []
                for a, f, people in batch:
                    f.safety = s.safety.assess(a, f, len(people))
                    s.safety.record(f.safety)
                    if not f.safety.allowed:
                        break
                    checked.append(f)
                    f.status, f.aircraft = 'assigned', a.name
                    if f.kind == 'reposition':
                        s.safety.tentative.append(f)
                valid = len(checked) == len(batch)
                for f in checked:
                    f.status, f.aircraft = 'pending', None
                s.safety.tentative.clear()
                if not valid:
                    batch = []
            if not batch:
                yield s.env.timeout(c.time(c.decision_interval_min))
                continue
            transfer = max(c.time(len(people)*c.board_min_per_passenger) for a,f,people in batch)
            for a, f, passengers in batch:
                if f.kind == 'reposition':
                    s.flights.append(f)
                s.safety.record(f.safety, committed=True)
                a.busy, a.clearance = True, s.env.event()
                f.aircraft, f.passengers, f.status = a.name, passengers, 'assigned'
                f.route.departure_pending = True
                for p in passengers:
                    p.flight = f.name
                for port, operation in ((a.port, 'takeoff'), (f.route.dest, 'landing')):
                    assert s.ports[port].queue.reserve((f.name, operation), a.name)
                s.env.process(a.fly(f))
            # Atomic pushback allows a cycle of full stands to exchange aircraft.
            # Destination claims stay reserved until these aircraft leave again.
            yield s.env.timeout(transfer)
            for a, f, people in batch:
                s.ports[a.port].allocated -= 1
                s.ports[f.route.dest].allocated += 1
            s.log.allocations.append((s.env.now, {p:v.allocated for p,v in s.ports.items()}))
            for a, f, people in batch:
                a.clearance.succeed()


class Simulation:
    def __init__(self, config=None, passenger_arrivals=None, flight_schedule=None):
        self.config = config or example_config()
        self.config.validate()
        c = self.config
        self.env, self.log = simpy.Environment(), Recorder()
        self.failure_event = self.env.event()
        self.ports = {p: Vertiport(self.env, c.capacities(p), c.seed+2000003+i, c.time_scale_min)
                      for i,p in enumerate(c.port_names)}
        self.routes = {(a, b): Route(a, b, c.time(60 * d / c.speed_kmh), c.energy(d))
                       for (a, b), d in c.network().items()}
        self.safety = SafetyPolicy(self)
        self._paths()
        counts = dict(c.initial_counts) if c.initial_counts is not None else {p: 0 for p in c.port_names}
        if c.initial_counts is None:
            for _ in range(c.fleet):
                p = min((p for p in counts if counts[p] < self.ports[p].stands), key=counts.get)
                counts[p] += 1
        if (set(counts) != set(c.port_names) or sum(counts.values()) != c.fleet or
                any(type(n) is not int or not 0 <= n <= self.ports[p].stands for p, n in counts.items())):
            raise ValueError("Initial aircraft counts must cover ports, fit stands and sum to fleet.")
        self.aircraft = []
        for port, count in counts.items():
            if count and c.initial_soc + 1e-9 < self.escape[port] + c.reserve_soc:
                raise ValueError("An initial aircraft cannot reach charging with its reserve.")
            self.ports[port].allocated = count
            for _ in range(count):
                self.aircraft.append(Aircraft(self, f"A{len(self.aircraft) + 1}", port))
        self.log.allocations.append((0, counts.copy()))
        self.delay_rng = {a.name: random.Random(c.seed + 100000 + i) for i, a in enumerate(self.aircraft)}
        self.passengers, self.waiting = [], {r: [] for r in self.routes}
        self.flights = self._schedule(flight_schedule)
        self.scheduler = Scheduler(self)
        if passenger_arrivals is None:
            for i, (route, rate) in enumerate(sorted(c.demand().items())):
                if rate:
                    self.env.process(self._demand(route, rate, random.Random(c.seed + 1009 * (i + 1))))
        else:
            arrivals = sorted(passenger_arrivals)
            if any(not isfinite(t) or not 0 <= t < c.demand_stop_min or (a, b) not in self.routes for t, a, b in arrivals):
                raise ValueError("Explicit arrivals need valid times and direct routes.")
            self.env.process(self._given_demand(arrivals))
        self.env.process(self.scheduler.run())

    def _paths(self):
        c = self.config
        self.escape = {p: 0.0 if v.chargers else inf for p, v in self.ports.items()}
        self.escape_next = {p: None for p in self.ports}
        # Bellman relaxation: least-energy path to any charging port, directed routes.
        for _ in self.ports:
            for (a, b), r in self.routes.items():
                cost = self.safety.leg_budget(r)
                if cost + self.escape[b] < self.escape[a]:
                    self.escape[a], self.escape_next[a] = cost + self.escape[b], b
        if any(not isfinite(value) for value in self.escape.values()):
            raise ValueError("Every port must have a directed path to a charger.")
        if any(self.required(r) > 1 + 1e-9 for r in self.routes.values()):
            raise ValueError("A route plus its charging escape and reserve exceeds a full battery.")
        self.path_distance = {(a, b): 0 if a == b else inf for a in self.ports for b in self.ports}
        self.path = {(a, b): None for a in self.ports for b in self.ports}
        for (a, b), r in self.routes.items():
            self.path_distance[a, b], self.path[a, b] = r.flight_time, b
        for k in self.ports:
            for a in self.ports:
                for b in self.ports:
                    if self.path_distance[a, k] + self.path_distance[k, b] < self.path_distance[a, b]:
                        self.path_distance[a, b] = self.path_distance[a, k] + self.path_distance[k, b]
                        self.path[a, b] = self.path[a, k]

    def required(self, route):
        return self.safety.leg_budget(route) + self.escape[route.dest] + self.config.reserve_soc

    def _schedule(self, given):
        c, services = self.config, []
        if given is None:
            for i, route in enumerate(self.routes):
                interval = c.service_intervals_min.get(route, c.timetable_interval_min)
                minute = c.first_departure_min + i * interval / len(self.routes)
                while minute < c.horizon_min:
                    services.append((minute, *route, None))
                    minute += interval
        else:
            services = [tuple(row) if len(row) == 4 else (*row, None) for row in given]
        names = {a.name for a in self.aircraft}
        if any(not isfinite(t) or t < 0 or (a, b) not in self.routes or
               (preferred is not None and preferred not in names) for t, a, b, preferred in services):
            raise ValueError("Invalid explicit flight schedule.")
        flights = [Flight(f"F{i + 1}", self.routes[a, b], c.time(t), preferred)
                   for i, (t, a, b, preferred) in enumerate(sorted(services, key=lambda r: r[0]))]
        # Nominal rotation uses mean/full-load timing, never future realised delays.
        nominal = {a.name: [a.port, 0] for a in self.aircraft}
        for f in flights:
            candidates = [name for name, (port, ready) in nominal.items() if port == f.route.origin]
            if f.preferred_aircraft is None and candidates:
                f.preferred_aircraft = min(candidates, key=lambda name: (nominal[name][1], name))
            if f.preferred_aircraft:
                previous = nominal[f.preferred_aircraft][1]
                ready = max(f.scheduled, previous + c.time(c.seats * c.board_min_per_passenger + c.taxi_out_min))
                ready += f.route.flight_time + c.time(c.capacities(f.route.origin)['pad_service_mean_min'] +
                    c.capacities(f.route.dest)['pad_service_mean_min'] + c.taxi_in_min + c.seats*c.unload_min_per_passenger)
                nominal[f.preferred_aircraft] = [f.route.dest, ready]
        return flights

    def _add_passenger(self, route):
        p = Passenger(len(self.passengers) + 1, *route, self.env.now)
        self.passengers.append(p)
        self.waiting[route].append(p)

    def _demand(self, route, rate, rng):
        c = self.config
        mean_gap = 60 / rate
        while True:
            gap = c.time(rng.expovariate(1 / mean_gap) if c.demand_mode == "poisson" else mean_gap)
            if self.env.now + gap >= c.time(c.demand_stop_min):
                return
            yield self.env.timeout(gap)
            self._add_passenger(route)

    def _given_demand(self, arrivals):
        for minute, origin, dest in arrivals:
            yield self.env.timeout(max(0, self.config.time(minute) - self.env.now))
            self._add_passenger((origin, dest))

    def run(self, until_min=None):
        end = self.config.time(self.config.horizon_min if until_min is None else until_min)
        if not isfinite(end) or end <= self.env.now:
            raise ValueError("Run to a later finite absolute time.")
        if self.failure_event.triggered:
            raise ValueError("This run stopped at its energy failure; start a new scenario.")
        self.env.run(until=self.env.timeout(end-self.env.now) | self.failure_event)
        return self


def example_config(**changes):
    """Four-port example; all other networks use the same classes."""
    overrides = {"A": {"chargers": 2}, "B": {"chargers": 0}, "D": {"chargers": 0}}
    ports = changes.get("port_names", ScenarioConfig.port_names)
    defaults = dict(port_overrides={p: v for p, v in overrides.items() if p in ports})
    defaults.update(changes)
    return ScenarioConfig(**defaults)
