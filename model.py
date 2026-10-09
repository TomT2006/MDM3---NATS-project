"""Fixed-fleet eVTOL teaching model. SimPy time is dimensionless; see START_HERE.ipynb."""

from dataclasses import dataclass, field
from itertools import cycle
from math import isfinite
import simpy


@dataclass
class ScenarioConfig:
    port_names: tuple = ("A", "B")
    fleet: int = 8
    stands_per_port: int = 4
    chargers_per_port: int = 4
    pads_per_port: int = 1
    port_overrides: dict = field(default_factory=dict)  # e.g. {"B": {"chargers": 2}}
    initial_counts: dict = None                       # None: spread fleet across ports
    routes_km: dict = None                            # None: directed ring, 30 km per leg
    speed_kmh: float = 200
    headway_min: float = 3
    charge_min: float = 10
    passengers: int = 4
    unload_min_per_passenger: float = 1
    board_min_per_passenger: float = 1
    taxi_out_min: float = 1
    takeoff_min: float = 1
    landing_min: float = 1
    taxi_in_min: float = 1
    horizon_min: float = 720
    time_scale_min: float = 10
    length_scale_km: float = 30

    def time(self, minutes):
        return minutes / self.time_scale_min

    def capacities(self, port):
        defaults = dict(stands=self.stands_per_port, chargers=self.chargers_per_port,
                        pads=self.pads_per_port)
        return {**defaults, **self.port_overrides.get(port, {})}

    def network(self):
        if self.routes_km is not None:
            return dict(self.routes_km)
        ports = self.port_names
        return {(p, ports[(i + 1) % len(ports)]): 30 for i, p in enumerate(ports)}

    def validate(self):
        ports = self.port_names
        if len(ports) < 2 or len(set(ports)) != len(ports):
            raise ValueError("Use at least two distinct port names.")
        positive = (self.speed_kmh, self.horizon_min, self.time_scale_min,
                    self.length_scale_km, self.takeoff_min, self.landing_min)
        nonnegative = (self.headway_min, self.charge_min, self.taxi_out_min,
                       self.taxi_in_min, self.unload_min_per_passenger,
                       self.board_min_per_passenger)
        if any(not isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Speed, scales, horizon and pad times must be positive and finite.")
        if any(not isfinite(x) or x < 0 for x in nonnegative):
            raise ValueError("Other durations must be nonnegative and finite.")
        if type(self.fleet) is not int or self.fleet < 1:
            raise ValueError("Fleet must be a positive integer.")
        if type(self.passengers) is not int or self.passengers < 0:
            raise ValueError("Passengers must be a nonnegative integer.")
        if set(self.port_overrides) - set(ports):
            raise ValueError("A capacity override refers to an unknown port.")
        for port in ports:
            caps = self.capacities(port)
            if set(caps) != {"stands", "chargers", "pads"}:
                raise ValueError("Capacity keys are stands, chargers and pads.")
            if any(type(n) is not int or n < 1 for n in caps.values()):
                raise ValueError("Capacities must be positive integers.")
        if self.fleet > sum(self.capacities(p)["stands"] for p in ports):
            raise ValueError("This initial condition starts every aircraft on a stand.")
        routes = self.network()
        for (origin, dest), distance in routes.items():
            if origin not in ports or dest not in ports or origin == dest:
                raise ValueError("Each route must link two different known ports.")
            if not isfinite(distance) or distance <= 0:
                raise ValueError("Route distances must be positive and finite.")
        if any(not any(a == p for a, b in routes) for p in ports):
            raise ValueError("Each port needs at least one outgoing route.")


@dataclass
class Route:
    origin: str
    dest: str
    flight_time: float
    last_departure: float = float("-inf")


@dataclass
class FlightPlan:
    aircraft: str
    route: Route
    transfer_start: float
    takeoff_start: float
    departure: float             # End of takeoff / entry to cruise
    landing_start: float
    arrival: float               # End of landing
    departure_pad: int
    arrival_pad: int


class Vertiport:
    def __init__(self, env, capacities):
        self.stands = capacities["stands"]
        self.allocated = 0        # Occupied stands PLUS stands promised to inbound flights
        self.chargers = simpy.Resource(env, capacity=capacities["chargers"])
        self.pads = [[] for _ in range(capacities["pads"])]

    def next_pad_slot(self, start, duration, now):
        """Earliest gap on any pad; intervals are [start, end)."""
        candidates = []
        for index, calendar in enumerate(self.pads):
            calendar[:] = [(a, b) for a, b in calendar if b > now + 1e-10]
            candidate = start
            for a, b in calendar:
                if candidate + duration <= a + 1e-10:
                    break
                if candidate < b - 1e-10:
                    candidate = b
            candidates.append((candidate, index))
        return min(candidates)

    def reserve_pad(self, index, start, end):
        self.pads[index].append((start, end))
        self.pads[index].sort()


class Recorder:
    def __init__(self):
        self.states = {}           # aircraft -> [(time, state, port), ...]
        self.flights = []          # Confirmed plans; some finish after the run horizon
        self.allocations = []      # (time, {port: occupied + reserved stands})

    def state(self, aircraft, time, state, port):
        self.states.setdefault(aircraft, []).append((time, state, port))

    def intervals(self, end, start=0):
        """Clip every activity to the observation window, including unfinished ones."""
        for aircraft, history in self.states.items():
            for i, (time, state, port) in enumerate(history):
                following = history[i + 1][0] if i + 1 < len(history) else end
                left, right = max(time, start), min(following, end)
                if right > left:
                    yield aircraft, state, port, left, right


class Aircraft:
    def __init__(self, sim, name, port):
        self.sim, self.name, self.port = sim, name, port
        self.route, self.clearance = None, None

    def activity(self, state, duration):
        self.sim.log.state(self.name, self.sim.env.now, state, self.port)
        yield self.sim.env.timeout(duration)

    def run(self):
        s, c = self.sim, self.sim.config
        while True:
            self.route = next(s.destinations[self.port])
            s.log.state(self.name, s.env.now, "dispatch_wait", self.port)
            plan = yield s.scheduler.request(self)
            yield from self.activity("taxi_out", c.time(c.taxi_out_min))
            yield from self.activity("takeoff", c.time(c.takeoff_min))
            self.port = None
            yield from self.activity("cruise", plan.route.flight_time)
            self.port = plan.route.dest
            yield from self.activity("landing", c.time(c.landing_min))
            yield from self.activity("taxi_in", c.time(c.taxi_in_min))
            yield from self.activity("unloading", c.time(c.passengers * c.unload_min_per_passenger))
            s.log.state(self.name, s.env.now, "charger_wait", self.port)
            with s.ports[self.port].chargers.request() as request:
                yield request
                yield from self.activity("charging", c.time(c.charge_min))
            yield from self.activity("boarding", c.time(c.passengers * c.board_min_per_passenger))


class Scheduler:
    """Conservative coordinated departures: a path to a free stand, or a cycle."""
    def __init__(self, sim):
        self.sim, self.pending = sim, []
        self.changed = sim.env.event()

    def request(self, aircraft):
        aircraft.clearance = self.sim.env.event()
        self.pending.append(aircraft)
        if not self.changed.triggered:
            self.changed.succeed()
        return aircraft.clearance

    def choose_batch(self):
        for first in self.pending:
            chain, seen, aircraft = [], {}, first
            while aircraft is not None:
                if aircraft.port in seen:
                    return chain[seen[aircraft.port]:]   # Closed cycle of stand exchanges
                seen[aircraft.port] = len(chain)
                chain.append(aircraft)
                dest = self.sim.ports[aircraft.route.dest]
                if dest.allocated < dest.stands:
                    return chain                      # Path ending at a free stand
                aircraft = next((a for a in self.pending if a.port == aircraft.route.dest), None)
        return []

    def plan_batch(self, batch):
        s, c = self.sim, self.sim.config
        out, takeoff, land = c.time(c.taxi_out_min), c.time(c.takeoff_min), c.time(c.landing_min)
        start = max([s.env.now] + [a.route.last_departure + c.time(c.headway_min) - out - takeoff
                                   for a in batch])
        while True:
            plans, shift = [], 0
            for a in batch:
                depart = start + out + takeoff
                arrive = depart + a.route.flight_time
                dep_slot, dep_pad = s.ports[a.port].next_pad_slot(start + out, takeoff, s.env.now)
                arr_slot, arr_pad = s.ports[a.route.dest].next_pad_slot(arrive, land, s.env.now)
                shift = max(shift, dep_slot - (start + out), arr_slot - arrive)
                plans.append(FlightPlan(a.name, a.route, start, start + out, depart,
                                        arrive, arrive + land, dep_pad, arr_pad))
            if shift <= 1e-10:
                return plans
            start += shift          # Move the whole coordinated batch; then check again

    def run(self):
        s = self.sim
        while True:
            batch = self.choose_batch()
            if not batch:
                self.changed = s.env.event()
                yield self.changed
                continue
            plans = self.plan_batch(batch)
            yield s.env.timeout(max(0, plans[0].transfer_start - s.env.now))
            # Commit stand exchanges together, before releasing any aircraft.
            for a in batch:
                s.ports[a.port].allocated -= 1
                s.ports[a.route.dest].allocated += 1
            s.log.allocations.append((s.env.now, {p: v.allocated for p, v in s.ports.items()}))
            for a, plan in zip(batch, plans):
                s.ports[a.port].reserve_pad(plan.departure_pad, plan.takeoff_start, plan.departure)
                s.ports[a.route.dest].reserve_pad(plan.arrival_pad, plan.landing_start, plan.arrival)
                a.route.last_departure = plan.departure
                s.log.flights.append(plan)
                self.pending.remove(a)
                a.clearance.succeed(plan)


class Simulation:
    def __init__(self, config=None):
        self.config = config or ScenarioConfig()
        self.config.validate()
        c = self.config
        self.env, self.log = simpy.Environment(), Recorder()
        self.ports = {p: Vertiport(self.env, c.capacities(p)) for p in c.port_names}
        speed = c.speed_kmh / (60 * c.length_scale_km / c.time_scale_min)
        self.routes = [Route(a, b, (distance / c.length_scale_km) / speed)
                       for (a, b), distance in c.network().items()]
        self.destinations = {p: cycle([r for r in self.routes if r.origin == p]) for p in self.ports}
        counts = c.initial_counts
        if counts is None:
            counts = {p: 0 for p in self.ports}
            for _ in range(c.fleet):
                p = min((p for p in self.ports if counts[p] < self.ports[p].stands), key=counts.get)
                counts[p] += 1
        if (set(counts) != set(self.ports) or sum(counts.values()) != c.fleet
                or any(type(n) is not int or not 0 <= n <= self.ports[p].stands for p, n in counts.items())):
            raise ValueError("Initial counts must cover every port, sum to fleet, and fit on the stands.")
        self.aircraft = []
        for p, n in counts.items():
            self.ports[p].allocated = n
            self.aircraft.extend(Aircraft(self, f"{p}{i + 1}", p) for i in range(n))
        self.log.allocations.append((0, dict(counts)))
        self.scheduler = Scheduler(self)
        # All aircraft start ready, charged and boarded.
        for aircraft in self.aircraft:
            self.env.process(aircraft.run())
        self.env.process(self.scheduler.run())

    def run(self, until_min=None):
        """Run to an absolute time in minutes. Call again with a later time to extend."""
        until = self.config.time(self.config.horizon_min if until_min is None else until_min)
        if not isfinite(until) or until <= self.env.now:
            raise ValueError("The new finite horizon must be later than the current time.")
        self.env.run(until=until)
        return self
