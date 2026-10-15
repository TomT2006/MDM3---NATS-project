"""Theory and controlled-input verification of the actual aircraft pad queue."""
import math
import random
import simpy
from queues import AircraftPadQueue
from fractions import Fraction


def transient(arrival_rate, service_rate, K, times_min):
    """Exact finite-chain probabilities starting empty: p(t) = p(0) exp(Q t)."""
    import numpy as np
    from scipy.linalg import expm
    theory(arrival_rate, service_rate, K)  # Validate common parameters
    if any(not math.isfinite(t) or t < 0 for t in times_min):
        raise ValueError("Times must be finite and nonnegative.")
    Q = np.zeros((K+1,K+1))
    for n in range(K+1):
        if n < K: Q[n,n+1] = arrival_rate
        if n > 0: Q[n,n-1] = service_rate
        Q[n,n] = -sum(Q[n])
    return np.array([expm(Q*t)[0] for t in times_min])


def theory(arrival_rate, service_rate, K):
    """Rates in jobs/min; K includes the job in service. Times returned in min."""
    if (not all(math.isfinite(x) and x > 0 for x in (arrival_rate, service_rate))
            or type(K) is not int or K < 1):
        raise ValueError("Use positive finite rates and integer K >= 1.")
    rho = arrival_rate / service_rate
    logs = [n * math.log(rho) for n in range(K + 1)]
    largest = max(logs)
    weights = [math.exp(x - largest) for x in logs]
    pn = [w / sum(weights) for w in weights]
    busy = sum(pn[1:])
    L = sum(n * p for n, p in enumerate(pn))
    Lq = sum(max(0, n - 1) * p for n, p in enumerate(pn))
    # Equivalent to lambda*(1-pK), but avoids subtracting nearly equal numbers.
    effective_rate = service_rate * busy
    return dict(rho=rho, K=K, pn=pn, blocking=pn[-1], busy=busy, L=L, Lq=Lq,
                effective_rate=effective_rate, Wq=Lq / effective_rate, W=L / effective_rate)


def simulate(arrival_rate=0.9, service_rate=1, K=4, duration_min=12000,
             warmup_min=1200, seed=1, time_scale_min=10):
    """Exercise the SAME AircraftPadQueue used by every operating vertiport.

    Only the source/observation harness differs: independent Poisson offers,
    then stop and drain. No second service/admission implementation exists.
    """
    exact = theory(arrival_rate,service_rate,K)
    if not (math.isfinite(duration_min) and math.isfinite(warmup_min)
            and 0 <= warmup_min < duration_min and math.isfinite(time_scale_min) and time_scale_min > 0):
        raise ValueError('Invalid horizon, warm-up or time scale.')
    normalise = lambda minutes: Fraction(str(minutes))/Fraction(str(time_scale_min))
    env = simpy.Environment()
    queue = AircraftPadQueue(env,K,service_rate,seed+1000003,time_scale_min)
    rng = random.Random(seed)
    end = normalise(duration_min)
    def source():
        while True:
            gap = normalise(rng.expovariate(arrival_rate))
            if env.now+gap >= end:
                yield env.timeout(end-env.now)
                return
            yield env.timeout(gap)
            queue.offer()
    env.process(source())
    env.run()
    assert queue.audit()
    result = queue.statistics(normalise(warmup_min),end)
    result.update(rho=exact['rho'],seed=seed)
    return result
