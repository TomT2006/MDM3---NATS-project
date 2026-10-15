"""A real FIFO aircraft pad queue; future permits are counted separately.

One shared takeoff/landing server, exponential service, K aircraft including
service. Operational arrivals come from the finite fleet. With Poisson offers,
no reservations and no retries, the SAME service code is exact M/M/1/K.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
import random
import simpy


@dataclass
class Ticket:
    number: int
    arrived: object
    aircraft: str
    flight: str
    operation: str
    done: object
    blocked: bool = False
    started: object = None
    completed: object = None


class AircraftPadQueue:
    def __init__(self, env, K, service_rate, seed=1, time_scale=10):
        if type(K) is not int or K < 1:
            raise ValueError('K must be a positive integer including service.')
        if not all(isfinite(x) and x > 0 for x in (service_rate, time_scale)):
            raise ValueError('Positive finite service rate and time scale required.')
        self.env, self.capacity = env, K
        self.service_rate, self.time_scale = service_rate, time_scale
        self.server, self.rng = simpy.Resource(env, 1), random.Random(seed)
        self.jobs, self.reservations, self.tickets = set(), {}, []
        self.history = [(env.now, 0)]              # Actual N only
        self.reservation_history = [(env.now, 0)]
        self.combined_history = [(env.now, 0, 0)] # (time, N, R)

    def fits(self, count=1):
        return len(self.jobs) + len(self.reservations) + count <= self.capacity

    def _record(self):
        n, r = len(self.jobs), len(self.reservations)
        self.history.append((self.env.now, n))
        self.reservation_history.append((self.env.now, r))
        self.combined_history.append((self.env.now, n, r))

    def reserve(self, key, aircraft):
        if key in self.reservations:
            raise ValueError('Duplicate permit.')
        if not self.fits():
            return False
        self.reservations[key] = aircraft
        self._record()
        return True

    def arrive(self, key, aircraft, on_start=None):
        """A real aircraft reaches the queue and consumes its advance permit."""
        if self.reservations.get(key) != aircraft:
            raise ValueError('Aircraft must own this permit.')
        del self.reservations[key]
        flight, operation = key
        return self.offer(aircraft, flight, operation, on_start)

    def offer(self, aircraft='external', flight='', operation='takeoff', on_start=None):
        """Unreserved offers are lost if full; used by the Poisson test harness.

        Operating aircraft use arrive() with a permit, so are never discarded.
        """
        if operation not in ('takeoff', 'landing'):
            raise ValueError('Pad operation must be takeoff or landing.')
        t = Ticket(len(self.tickets)+1, self.env.now, aircraft, flight, operation, self.env.event())
        self.tickets.append(t)
        if not self.fits():
            t.blocked = True
            t.done.succeed(False)
            return t
        self.jobs.add(t.number)
        self._record()
        self.env.process(self._serve(t, on_start))
        return t

    def _serve(self, ticket, on_start):
        with self.server.request() as request:
            yield request
            ticket.started = self.env.now
            if on_start is not None:
                on_start()
            minutes = self.rng.expovariate(self.service_rate)
            duration = Fraction(str(minutes)) / Fraction(str(self.time_scale))
            yield self.env.timeout(duration)
            ticket.completed = self.env.now
            self.jobs.remove(ticket.number)
            self._record()
        ticket.done.succeed(ticket)

    def probabilities(self, start, end):
        if not 0 <= start < end <= self.env.now + 1e-9:
            raise ValueError('Observation window must be inside simulated time.')
        times = [0.0] * (self.capacity + 1)
        for i, (left, n) in enumerate(self.history):
            right = self.history[i+1][0] if i+1 < len(self.history) else end
            times[n] += float(max(0, min(right, end) - max(left, start)))
        return [value / float(end-start) for value in times]

    def statistics(self, start, end):
        """Time-weighted N; waits use observed starts and report censoring."""
        pn = self.probabilities(start, end)
        cohort = [t for t in self.tickets if start <= t.arrived < end]
        accepted = [t for t in cohort if not t.blocked]
        waits = [float((t.started-t.arrived)*self.time_scale) for t in accepted if t.started is not None]
        totals = [float((t.completed-t.arrived)*self.time_scale) for t in accepted if t.completed is not None]
        blocked = sum(t.blocked for t in cohort)
        completed = sum(t.completed is not None and start <= t.completed < end for t in self.tickets)
        minutes = float((end-start)*self.time_scale)
        reserved_area = committed_full_area = 0.0
        for i, (left, n, r) in enumerate(self.combined_history):
            right = self.combined_history[i+1][0] if i+1 < len(self.combined_history) else end
            dt = float(max(0, min(right,end)-max(left,start)))
            reserved_area += r*dt
            committed_full_area += (n+r == self.capacity)*dt
        return dict(K=self.capacity, pn=pn, offered=len(cohort), blocked=blocked,
                    admitted=len(accepted), observed_waits=len(waits),
                    not_started=len(accepted)-len(waits), not_completed=len(accepted)-len(totals),
                    blocking=blocked/len(cohort) if cohort else None,
                    full_fraction=pn[-1], busy=1-pn[0],
                    L=sum(n*p for n,p in enumerate(pn)),
                    Lq=sum(max(0,n-1)*p for n,p in enumerate(pn)),
                    Wq=sum(waits)/len(waits) if waits else None,
                    W=sum(totals)/len(totals) if totals else None,
                    effective_rate=completed/minutes, admitted_rate=len(accepted)/minutes,
                    mean_reservations=reserved_area/float(end-start),
                    committed_full_fraction=committed_full_area/float(end-start))

    def audit(self):
        accepted = [t for t in self.tickets if not t.blocked]
        assert len(accepted)-sum(t.completed is not None for t in accepted) == len(self.jobs)
        assert all(0 <= n and 0 <= r and n+r <= self.capacity for _,n,r in self.combined_history)
        previous_end, pending = 0, False
        for t in accepted:
            if t.started is None:
                pending = True
                continue
            assert not pending, 'FIFO violated'
            assert abs(t.started-max(t.arrived,previous_end)) < 1e-9, 'Overlap or idle server with queue'
            if t.completed is None:
                pending = True
            else:
                assert t.completed >= t.started
                previous_end = t.completed
        return True
