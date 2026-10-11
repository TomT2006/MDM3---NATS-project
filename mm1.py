"""Independent M/M/1 benchmark. It does not describe the closed aircraft network."""
import random
from collections import defaultdict
from math import isfinite
from statistics import mean
import simpy


def mm1(arrival_rate=2/3, service_rate=1, horizon_min=100000,
        warmup_min=10000, time_scale_min=10, seed=42):
    if (not all(isfinite(x) for x in (arrival_rate, service_rate, horizon_min, warmup_min, time_scale_min))
            or not 0 < arrival_rate < service_rate or not 0 <= warmup_min < horizon_min or time_scale_min <= 0):
        raise ValueError("Require 0 < lambda < mu, 0 <= warm-up < horizon, and a positive scale.")
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    arrivals, services = random.Random(seed), random.Random(seed + 1)
    lam, mu = arrival_rate * time_scale_min, service_rate * time_scale_min
    horizon, warmup = horizon_min / time_scale_min, warmup_min / time_scale_min
    waits = []
    population, last_change = 0, 0.0
    state_times = defaultdict(float)

    def observe(time):
        # Integrate N(t) over [warm-up, horizon), independently of customer waits.
        nonlocal last_change
        duration = max(0, min(time, horizon) - max(last_change, warmup))
        state_times[population] += duration
        last_change = time

    def customer():
        nonlocal population
        arrived = env.now
        observe(env.now)
        population += 1
        with server.request() as request:
            yield request
            if arrived >= warmup:
                waits.append((env.now - arrived) * time_scale_min)
            yield env.timeout(services.expovariate(mu))
            observe(env.now)
            population -= 1

    def source():
        while True:
            gap = arrivals.expovariate(lam)
            if env.now + gap >= horizon:
                return
            yield env.timeout(gap)
            env.process(customer())

    env.process(source())
    env.run()  # Finite source; drain so every retained arrival has a completed wait.
    observe(horizon)  # Includes the empty tail if the final service ended before the horizon.
    probabilities = {n: duration / (horizon - warmup) for n, duration in state_times.items()}
    rho = arrival_rate / service_rate
    return {"rho": rho,
            "Wq_theory_min": arrival_rate / (service_rate * (service_rate - arrival_rate)),
            "Wq_simulated_min": mean(waits) if waits else None, "arrivals_measured": len(waits),
            "Lq_theory": rho**2 / (1 - rho),
            "Lq_simulated": sum(max(n - 1, 0) * p for n, p in probabilities.items()),
            "busy_fraction": 1 - probabilities.get(0, 0),
            "state_probabilities": probabilities}
