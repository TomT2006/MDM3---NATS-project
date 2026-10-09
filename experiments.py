"""Step 6: run reproducible validation and sensitivity experiments separately from model.py.

Run this file to regenerate experiment_results.json. Edit the settings in run_experiments
or pass a ScenarioConfig. No aircraft-model randomness is introduced here.
"""
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t

from model import ScenarioConfig, Simulation
from results import STAND_STATES, summary, port_statistics
from mm1 import mm1


def interval95(values):
    """Student-t interval across independent replication means, not individual waits."""
    centre = mean(values)
    half = float(t.ppf(0.975, len(values) - 1)) * stdev(values) / sqrt(len(values))
    return {"mean": centre, "lower": centre - half, "upper": centre + half}


def audit(sim):
    """Check every recorded flight/resource interval in an experiment."""
    c, end = sim.config, sim.env.now
    slots, departures, usage, exposure = defaultdict(list), defaultdict(list), defaultdict(dict), defaultdict(float)
    actual = {(a, round(time, 8), state, port) for a, history in sim.log.states.items()
              for time, state, port in history}
    for f in sim.log.flights:
        slots[f.route.origin, f.departure_pad].append((f.takeoff_start, f.departure))
        slots[f.route.dest, f.arrival_pad].append((f.landing_start, f.arrival))
        departures[f.route.origin, f.route.dest].append(f.departure)
        for time, state, port in ((f.takeoff_start, "takeoff", f.route.origin),
                                  (f.departure, "cruise", None), (f.landing_start, "landing", f.route.dest)):
            if time < end - 1e-8:
                assert (f.aircraft, round(time, 8), state, port) in actual, "Aircraft missed its reserved slot"
    for windows in slots.values():
        windows.sort()
        assert all(a[1] <= b[0] + 1e-8 for a, b in zip(windows, windows[1:])), "Pad overlap"
    for times in departures.values():
        assert all(b - a >= c.time(c.headway_min) - 1e-8 for a, b in zip(times, times[1:])), "Headway violated"
    for aircraft, state, port, left, right in sim.log.intervals(end):
        exposure[aircraft] += right - left
        kinds = (["stands"] if state in STAND_STATES else [])
        kinds += (["chargers"] if state == "charging" else [])
        kinds += (["pads"] if state in {"takeoff", "landing"} else [])
        for kind in kinds:
            for time, change in ((left, 1), (right, -1)):
                key = round(time, 8)
                usage[port, kind][key] = usage[port, kind].get(key, 0) + change
    assert len(exposure) == c.fleet and all(abs(x - end) < 1e-7 for x in exposure.values()), "Lost aircraft time"
    for (port, kind), changes in usage.items():
        level = 0
        for time, change in sorted(changes.items()):
            level += change
            assert 0 <= level <= c.capacities(port)[kind], f"Capacity exceeded: {port} {kind}"
    for time, counts in sim.log.allocations:
        assert sum(counts.values()) == c.fleet, "Lost stand claim"
        assert all(0 <= n <= c.capacities(p)["stands"] for p, n in counts.items()), "Invalid stand allocation"
    return {"passed": True, "plans_checked": len(sim.log.flights)}


def measure(sim, warmup_min=0):
    r = summary(sim, warmup_min)
    ports = port_statistics(sim, warmup_min)
    return {"window_start_min": warmup_min, "window_end_min": sim.env.now * sim.config.time_scale_min,
            "flights_per_hour": r["flights_per_hour"], "completed_flights": r["completed_flights"],
            "time_fraction": r["time_fraction"], "ports": ports,
            "mean_dispatch_queue_total": sum(p["mean_dispatch_queue"] for p in ports.values()),
            "mean_charger_queue_total": sum(p["mean_charger_queue"] for p in ports.values())}


def two_port_bounds(c):
    """Necessary long-run bounds for the symmetric two-port experiments, not an optimiser."""
    assert c.port_names == ("A", "B") and len(c.network()) == 2
    flight = 60 * mean(c.network().values()) / c.speed_kmh
    handling = c.passengers * (c.unload_min_per_passenger + c.board_min_per_passenger)
    stand_time = handling + c.charge_min
    cycle = c.taxi_out_min + c.takeoff_min + flight + c.landing_min + c.taxi_in_min + stand_time
    caps = {p: c.capacities(p) for p in c.port_names}
    return {"fleet": c.fleet * 60 / cycle,
            "charging": 120 * min(x["chargers"] for x in caps.values()) / c.charge_min if c.charge_min else None,
            "pads": 120 * min(x["pads"] for x in caps.values()) / (c.takeoff_min + c.landing_min),
            "stands": 120 * min(x["stands"] for x in caps.values()) / stand_time if stand_time else None,
            "headway": 120 / c.headway_min if c.headway_min else None}


def run_experiments(base=None, replications=20, queue_horizon_min=100000, queue_warmup_min=10000,
                    horizon_min=10080, warmup_min=1440, verbose=True):
    base = base or ScenarioConfig()
    base.validate()
    if base.port_names != ("A", "B") or base.initial_counts is not None or base.port_overrides:
        raise ValueError("This experiment grid starts from uniform A/B ports with automatic initial counts; "
                         "use Simulation directly for custom networks or edit the grid below.")
    if replications < 2 or not 0 <= warmup_min < horizon_min:
        raise ValueError("Need at least two replications and 0 <= warm-up < horizon.")
    folder = Path(__file__).parent
    settings = asdict(base)
    settings["routes_km"] = [[a, b, d] for (a, b), d in base.network().items()]
    data = {"settings": settings, "replications": replications,
            "queue_horizon_min": queue_horizon_min, "queue_warmup_min": queue_warmup_min,
            "scenario_horizon_min": horizon_min, "scenario_warmup_min": warmup_min,
            "source_hashes": {name: sha256((folder / name).read_bytes()).hexdigest()
                              for name in ("model.py", "mm1.py", "results.py", "experiments.py")},
            "mm1_runs": [], "mm1_intervals": [], "sensitivity": [], "long_runs": []}
    for group, rho in enumerate((0.25, 2/3, 0.9)):
        runs = []
        for i in range(replications):
            seed = 100000 * group + 1000 + 2 * i  # No reused arrival/service seeds across runs
            result = mm1(rho, 1, queue_horizon_min, queue_warmup_min, base.time_scale_min, seed)
            runs.append(result)
            data["mm1_runs"].append({"seed": seed, **result})
        for metric, theory in (("Wq_simulated_min", rho/(1-rho)),
                               ("Lq_simulated", rho*rho/(1-rho)), ("busy_fraction", rho)):
            estimate = interval95([r[metric] for r in runs])
            data["mm1_intervals"].append({"rho": rho, "metric": metric, "theory": theory,
                                          **estimate, "relative_error": (estimate["mean"]-theory)/theory})
        if verbose:
            print(f"M/M/1: rho={rho:.3f}, {replications} replications complete", flush=True)

    scenarios = []
    for fleet in (1, 2, 4, 5, 6, 8):
        for headway in (1, 3, 5, 10):
            scenarios.append(("fleet_headway", replace(base, fleet=fleet, headway_min=headway)))
    for charge in (5, 10, 20, 30):
        for chargers in (1, 2, 4):
            scenarios.append(("charge_chargers", replace(base, charge_min=charge, chargers_per_port=chargers)))
    for pads in (1, 2):
        scenarios.append(("pads", replace(base, pads_per_port=pads)))
    # Diagnose pad-slot interactions at the longer stress-test headway.
    for fleet in (5, 8):
        scenarios.append(("headway_stress", replace(base, fleet=fleet, headway_min=10, pads_per_port=2)))
    for ports in (("A", "B"), ("A", "B", "C")):
        scenarios.append(("network", replace(base, port_names=ports, fleet=5, routes_km=None)))
    for counts in ({"A": 3, "B": 2}, {"A": 4, "B": 1}):
        scenarios.append(("initial_distribution", replace(base, fleet=5, initial_counts=counts)))
    for group, c in scenarios:
        c = replace(c, horizon_min=horizon_min)
        sim = Simulation(c).run()
        bounds = two_port_bounds(c) if c.port_names == ("A", "B") else {}
        row = {"group": group, "fleet": c.fleet, "headway_min": c.headway_min,
               "charge_min": c.charge_min, "chargers_per_port": c.chargers_per_port,
               "pads_per_port": c.pads_per_port, "port_names": c.port_names,
               "initial_counts": sim.log.allocations[0][1], "bounds": bounds,
               **measure(sim, warmup_min), "audit": audit(sim)}
        data["sensitivity"].append(row)
    if verbose:
        print(f"Sensitivity: {len(scenarios)} scenarios complete and audited", flush=True)

    for chargers in (4, 1):
        sim = Simulation(replace(base, chargers_per_port=chargers))
        snapshots = []
        for hours in (12, 24, 72, 168, 720):
            sim.run(hours * 60)
            for start in (0, 120, 720, 1440):
                if start < hours * 60:
                    snapshots.append(measure(sim, start))
        # Consecutive 24-hour windows, rather than cumulative averages.
        daily = []
        for day in range(30):
            left, right = sim.config.time(day * 1440), sim.config.time((day+1) * 1440)
            count = sum(left <= f.arrival < right - 1e-10 for f in sim.log.flights)
            daily.append({"day": day+1, "flights_per_hour": count/24})
        data["long_runs"].append({"chargers_per_port": chargers, "snapshots": snapshots,
                                   "daily": daily, "audit": audit(sim)})
        if verbose:
            print(f"30-day run: {chargers} charger(s) per port complete and audited", flush=True)
    return data


if __name__ == "__main__":
    results = run_experiments()
    destination = Path(__file__).with_name("experiment_results.json")
    destination.write_text(json.dumps(results, indent=2, allow_nan=False))
    print(f"Saved {destination.name}")
