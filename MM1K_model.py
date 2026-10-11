import random
from collections import defaultdict   
from math import isfinite            
from statistics import mean           
import simpy

#Horizon_min is the length of simulation in mins
#Warmup_min is the start of the simulation where the system is empty to remove bias
def mm1k(arrival_rate=2/3, service_rate=1, K=5, horizon_min=100000,
         warmup_min=10000, time_scale_min=10, seed=42):
    #Validate inputs 
    if (not all(isfinite(x) for x in (arrival_rate, service_rate, horizon_min, warmup_min, time_scale_min))
            or arrival_rate <= 0 or service_rate <= 0 or time_scale_min <= 0
            or not 0 <= warmup_min < horizon_min or int(K) != K or K < 1):
        raise ValueError("Require lambda, mu > 0, integer K >= 1, 0 <= warm-up < horizon, positive scale.")
    K = int(K)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)   
    #Separate seeded streams for arrivals and services keep the two processes independent
    arrivals, services = random.Random(seed), random.Random(seed + 1)
    #Convert rates and durations into simulation time units
    lam, mu = arrival_rate * time_scale_min, service_rate * time_scale_min
    horizon, warmup = horizon_min / time_scale_min, warmup_min / time_scale_min
    waits, tis = [], []     #wait in queue/ time in system
    offered = blocked = 0        #arrivals after warm-up and how many of them were blocked
    population, last_change = 0, 0.0   #current N(t) and the time it last changed
    state_times = defaultdict(float)   #total time spent with N(t) = n, after warm-up

    def observe(time):
        #Call just before population changes: credit the elapsed interval to the
        #current state, counting only the part inside [warm-up, horizon).
        nonlocal last_change
        state_times[population] += max(0, min(time, horizon) - max(last_change, warmup))
        last_change = time

    def customer():
        #Only accepted arrivals become customer processes
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
            if arrived >= warmup:
                tis.append((env.now - arrived) * time_scale_min) 

    def source():
        #Poisson arrival generator: exponential inter-arrival gaps
        nonlocal offered, blocked
        while True:
            gap = arrivals.expovariate(lam)
            if env.now + gap >= horizon:
                return                           #stop generating at the horizon
            yield env.timeout(gap)
            if env.now >= warmup:
                offered += 1
            if population >= K:                  #system full: arrival is lost
                if env.now >= warmup:
                    blocked += 1
            else:
                env.process(customer())          #admitted

    env.process(source())
    env.run()           #Finite source: run until every admitted customer has finished
    observe(horizon)    #Credit any remaining time up to the horizon (system empty after drain)

    #Time-average state probabilities over the measurement window
    probs = {n: d / (horizon - warmup) for n, d in state_times.items()}
    return {"rho": arrival_rate / service_rate, "K": K,
            #Blocking probability: fraction of arrivals lost vs. fraction of time full
            "P_block_simulated": blocked / offered if offered else None,
            "P_block_time_avg": probs.get(K, 0),
            #Throughput of admitted aircraft per minute
            "lam_eff_simulated": (offered - blocked) / (horizon_min - warmup_min),
            #Waiting time in queue and total time in system (accepted customers only)
            "Wq_simulated_min": mean(waits) if waits else None,
            "W_simulated_min": mean(tis) if tis else None,
            "arrivals_offered": offered, "arrivals_accepted": len(waits),
            #Mean number in system / in queue
            "L_simulated": sum(n * p for n, p in probs.items()),
            "Lq_simulated": sum(max(n - 1, 0) * p for n, p in probs.items()),
            "busy_fraction": 1 - probs.get(0, 0),    #utilisation = P(N > 0)
            "state_probabilities": probs}


if __name__ == "__main__":
    for key, val in mm1k().items():
        if not isinstance(val, dict):
            print(f"{key:>22}: {val}")