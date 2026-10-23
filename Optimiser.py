import itertools
import random
import statistics
import simpy
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# 1. SIMULATION PARAMETERS & CONSTANTS
# ----------------------------------------------------------------
P = dict(
    ports=('A', 'B', 'C', 'D'),
    seats=4,
    demand_per_hour=36,       # Total passengers per hour split across routes
    pad_mean=1.0,             # Mean takeoff/landing time on pad (minutes)
    cruise=9.0,               # Flight time between ports (minutes)
    board=2.0, unload=2.0,    # Boarding and unloading duration (minutes)
    turnaround=10.0,          # Charging and servicing time (minutes)
    max_delay=10.0,           # Max wait time for aircraft before service drop
    demand_stop=600,          # Time passenger arrivals stop (minutes)
    horizon=720,              # Total simulation duration (minutes)
    timely=60.0,              # Max wait threshold for "on-time" passenger (minutes)
)

# Optimization search space
SPACE = dict(
    fleet=[6, 8, 10, 12, 14, 16],    # Number of aircraft in the system
    interval=[15, 20, 30, 45],       # Minutes between flights per route
    K=[2, 4, 6, 8, 10, 12],          # Pad queue capacity
)

# Cost coefficients for objective function
PAX_WAIT_COST = 1.0       # Cost per passenger-minute spent waiting
UNSERVED_COST = 200.0     # Penalty per unserved passenger
FLIGHT_COST = 60.0        # Direct operating cost per flight
AIRCRAFT_HOUR_COST = 50.0 # Capital/operating cost per aircraft-hour
TARGET = 0.95             # Required service level target (95%)


# ----------------------------------------------------------------
# 2. SIMPY DES MODEL CLASSES & FUNCTIONS
# ----------------------------------------------------------------
class Pad:
    """Models a landing/takeoff pad with limited queue capacity K and single surface server."""
    def __init__(self, env, K, mean, rng):
        self.env, self.mean, self.rng = env, mean, rng
        self.space = simpy.Resource(env, K)   # Slots for waiting + in-service
        self.server = simpy.Resource(env, 1)  # Single physical landing pad
        self.waits = []                       # Log of queue waiting times

    def use(self):
        req = self.space.request()
        yield req
        t = self.env.now
        with self.server.request() as turn:
            yield turn
            self.waits.append(self.env.now - t)
            yield self.env.timeout(self.rng.expovariate(1.0 / self.mean))
        self.space.release(req)


def simulate(fleet, interval, K, seed, p=P):
    """Executes a single discrete-event simulation run for a design configuration."""
    env, rng = simpy.Environment(), random.Random(seed)
    names = p['ports']
    
    # Initialize pads and aircraft storage at each port
    pads = {n: Pad(env, K, p['pad_mean'], random.Random(rng.random())) for n in names}
    idle = {n: simpy.Store(env) for n in names}
    
    # Evenly distribute aircraft fleet across initial ports
    for a in range(fleet):
        idle[names[a % len(names)]].put(a)
        
    routes = [(a, b) for a in names for b in names if a != b]
    waiting = {r: [] for r in routes}
    everyone, flights = [], [0]

    # Process generating passenger demand per route
    def passengers(route, r):
        rate = p['demand_per_hour'] / 60.0 / len(routes)
        while env.now < p['demand_stop']:
            yield env.timeout(r.expovariate(rate))
            pax = dict(arrive=env.now, depart=None, done=None)
            everyone.append(pax)
            waiting[route].append(pax)

    # Process executing a single scheduled flight
    def flight(route):
        origin, dest = route
        if not waiting[route]:
            return # Cancel flight if no demand
            
        get = idle[origin].get()
        yield get | env.timeout(p['max_delay'])
        if not get.triggered:
            get.cancel() # Drop service if aircraft isn't available in time
            return
            
        aircraft = get.value
        yield env.timeout(p['board'])
        load = [waiting[route].pop(0) for _ in range(min(p['seats'], len(waiting[route])))]
        
        yield from pads[origin].use() # Takeoff
        flights[0] += 1
        for x in load:
            x['depart'] = env.now
            
        yield env.timeout(p['cruise'])
        yield from pads[dest].use()   # Landing
        yield env.timeout(p['unload'])
        
        for x in load:
            x['done'] = env.now
            
        yield env.timeout(p['turnaround'])
        idle[dest].put(aircraft)      # Return aircraft to idle pool

    # Process triggering regular timetable departures
    def timetable(route, offset):
        yield env.timeout(offset)
        while env.now < p['horizon']:
            env.process(flight(route))
            yield env.timeout(interval)

    # Launch all route processes
    for i, route in enumerate(routes):
        env.process(passengers(route, random.Random(rng.random())))
        env.process(timetable(route, i * interval / len(routes)))
        
    env.run(until=p['horizon'])

    # Calculate run performance metrics
    n = len(everyone)
    waits = sorted(x['depart'] - x['arrive'] for x in everyone if x['depart'] is not None)
    return dict(
        arrivals=n,
        completed=sum(x['done'] is not None for x in everyone),
        flights=flights[0],
        mean_wait=statistics.mean(waits) if waits else 0.0,
        p95_wait=waits[int(0.95 * (len(waits) - 1))] if waits else 0.0,
        served=sum(x['done'] is not None for x in everyone) / max(n, 1),
        timely=sum(w <= p['timely'] for w in waits) / max(n, 1),
        pad_wait=statistics.mean(w for pad in pads.values() for w in pad.waits) if flights[0] else 0.0,
        all_waits=waits
    )


# ----------------------------------------------------------------
# 3. QUEUE THEORY SCREENING & COST EVALUATION
# ----------------------------------------------------------------
def theory(lam, mu, K):
    """Calculates theoretical M/M/1/K queueing statistics."""
    rho = lam / mu
    if rho == 1.0:
        p = [1.0 / (K + 1)] * (K + 1)
    else:
        weights = [rho ** n for n in range(K + 1)]
        s = sum(weights)
        p = [w / s for w in weights]
        
    L = sum(n * pn for n, pn in enumerate(p))
    lam_eff = lam * (1.0 - p[K])
    Lq = L - (1.0 - p[0])
    return dict(blocking=p[K], Lq=Lq, Wq=Lq / lam_eff if lam_eff else 0.0)


def pad_ok(interval, K, max_blocking=0.25):
    """Screens out configurations with >25% theoretical pad queue blocking probability."""
    lam = 2.0 * (len(P['ports']) - 1) / interval
    return theory(lam, 1.0 / P['pad_mean'], K)['blocking'] <= max_blocking


def cost(m, fleet):
    """Calculates overall design cost based on delays, unserved passengers, and operations."""
    return (PAX_WAIT_COST * m['mean_wait'] * m['arrivals']
            + UNSERVED_COST * (m['arrivals'] - m['completed'])
            + FLIGHT_COST * m['flights']
            + AIRCRAFT_HOUR_COST * fleet * P['horizon'] / 60.0)


def optimise(seeds=(1, 2, 3)):
    """Grid search optimization loop across all parameters in SPACE."""
    rows = []
    runs_raw = {}
    for fleet, interval, K in itertools.product(*SPACE.values()):
        if not pad_ok(interval, K):
            continue # Skip configurations failing queue stability screen
            
        runs = [simulate(fleet, interval, K, s) for s in seeds]
        runs_raw[(fleet, interval, K)] = runs
        rows.append(dict(
            fleet=fleet, interval=interval, K=K,
            cost=statistics.mean(cost(m, fleet) for m in runs),
            served=statistics.mean(m['served'] for m in runs),
            timely=statistics.mean(m['timely'] for m in runs),
            p95_wait=statistics.mean(m['p95_wait'] for m in runs),
            mean_wait=statistics.mean(m['mean_wait'] for m in runs),
            feasible=all(m['served'] >= TARGET and m['timely'] >= TARGET for m in runs),
        ))
    ok = [r for r in rows if r['feasible']]
    best = min(ok or rows, key=lambda r: r['cost'])
    return best, sorted(rows, key=lambda r: r['cost']), runs_raw


# ----------------------------------------------------------------
# 4. TABLE GENERATION FUNCTION
# ----------------------------------------------------------------
def print_results_table(rows, best):
    """Prints a formatted summary table of all simulated configurations."""
    header = f"{'Rank':<5} | {'Fleet':<5} | {'Interval':<8} | {'K':<3} | {'Cost':<10} | {'Served (%)':<10} | {'Timely (%)':<10} | {'Mean Wait (m)':<13} | {'Feasible':<8}"
    divider = "-" * len(header)
    
    print("\n" + divider)
    print("                      CONFIGURATIONS EVALUATION TABLE")
    print(divider)
    print(header)
    print(divider)

    for i, r in enumerate(rows, 1):
        is_best = (r['fleet'] == best['fleet'] and 
                   r['interval'] == best['interval'] and 
                   r['K'] == best['K'])
        
        status = "YES" if r['feasible'] else "NO"
        line = f"{i:<5} | {r['fleet']:<5} | {r['interval']:<8} | {r['K']:<3} | {r['cost']:<10.2f} | {r['served']*100:<10.1f} | {r['timely']*100:<10.1f} | {r['mean_wait']:<13.1f} | {status:<8}"
        
        if is_best:
            print(f">>> {line}  <-- BEST CONFIGURATION")
        else:
            print(f"    {line}")
            
    print(divider + "\n")


# ----------------------------------------------------------------
# 5. PLOTTING FUNCTION
# ----------------------------------------------------------------
def plot_all_results(rows, best, runs_raw):
    """Plots all 3 analysis figures together in a single figure window."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- SUBPLOT 1: Cost vs. Service Level Trade-off ---
    costs = [r['cost'] for r in rows]
    serveds = [r['served'] * 100 for r in rows]
    feasibles = [r['feasible'] for r in rows]

    inf_x = [serveds[i] for i in range(len(rows)) if not feasibles[i]]
    inf_y = [costs[i] for i in range(len(rows)) if not feasibles[i]]
    axes[0].scatter(inf_x, inf_y, label='Infeasible', alpha=0.5)

    f_x = [serveds[i] for i in range(len(rows)) if feasibles[i]]
    f_y = [costs[i] for i in range(len(rows)) if feasibles[i]]
    axes[0].scatter(f_x, f_y, label='Feasible', alpha=0.7)

    axes[0].axvline(TARGET * 100, color='red', linestyle='--', label='Target')
    axes[0].scatter([best['served'] * 100], [best['cost']], color='magenta', marker='*', s=120, label='Optimal')

    axes[0].set_xlabel('Passengers Completed (%)')
    axes[0].set_ylabel('Total Cost')
    axes[0].set_title('Cost vs. Service Feasibility')
    axes[0].legend()
    axes[0].grid(True)

    # --- SUBPLOT 2: 2D Minimum Cost Matrix Heatmap ---
    fleets = sorted(list(set(r['fleet'] for r in rows)))
    intervals = sorted(list(set(r['interval'] for r in rows)))
    grid = np.full((len(fleets), len(intervals)), np.nan)

    for r in rows:
        i = fleets.index(r['fleet'])
        j = intervals.index(r['interval'])
        if np.isnan(grid[i, j]) or r['cost'] < grid[i, j]:
            grid[i, j] = r['cost']

    im = axes[1].imshow(grid, origin='lower', aspect='auto')

    for i in range(len(fleets)):
        for j in range(len(intervals)):
            val = grid[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f'{val:,.0f}', ha='center', va='center', color='white', fontsize=8)

    axes[1].set_xticks(range(len(intervals)))
    axes[1].set_xticklabels([f'{inv}m' for inv in intervals])
    axes[1].set_yticks(range(len(fleets)))
    axes[1].set_yticklabels([f'{f}' for f in fleets])

    axes[1].set_xlabel('Dispatch Interval')
    axes[1].set_ylabel('Fleet Size')
    axes[1].set_title('Minimum Cost Matrix')
    fig.colorbar(im, ax=axes[1], label='Cost')

    # --- SUBPLOT 3: Passenger Wait Time ECDF ---
    best_tuple = (best['fleet'], best['interval'], best['K'])
    runs = runs_raw[best_tuple]
    
    all_waits = []
    for r in runs:
        all_waits.extend(r['all_waits'])
        
    all_waits = np.sort(all_waits)
    ecdf = np.arange(1, len(all_waits) + 1) / len(all_waits)

    axes[2].plot(all_waits, ecdf, label='Optimal ECDF')
    axes[2].axvline(P['timely'], color='red', linestyle='--', label=f"SLA Target ({int(P['timely'])} min)")
    
    timely_pct = sum(all_waits <= P['timely']) / len(all_waits) * 100
    axes[2].scatter([P['timely']], [timely_pct / 100.0], color='red')

    axes[2].set_xlabel('Wait Time (min)')
    axes[2].set_ylabel('Cumulative Probability P(Wait <= w)')
    axes[2].set_title('Passenger Wait Time ECDF')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------
# 6. SCRIPT EXECUTION
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("Running optimization search loop...")
    best_config, sorted_rows, runs_raw = optimise()
    print("Optimization finished successfully.")
    
    # Print summary table to console
    print_results_table(sorted_rows, best_config)
    
    # Display graphics
    plot_all_results(sorted_rows, best_config, runs_raw)