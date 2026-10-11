"""Measurements kept separate from the simulator. All returned times are minutes."""
from collections import defaultdict


STAND_STATES = {"unloading", "charger_wait", "charging", "boarding", "dispatch_wait"}


def summary(sim, warmup_min=0):
    """Time fractions include ongoing activities, not just completed journeys."""
    c = sim.config
    start, end = c.time(warmup_min), sim.env.now
    if not 0 <= start < end:
        raise ValueError("Warm-up must precede the end of this run.")
    times = defaultdict(float)
    for aircraft, state, port, left, right in sim.log.intervals(end, start):
        times[state] += (right - left) * c.time_scale_min
    arrivals = sum(start <= f.arrival < end - 1e-10 for f in sim.log.flights)
    elapsed = (end - start) * c.time_scale_min
    return {"observation_min": elapsed, "completed_flights": arrivals,
            "flights_per_hour": arrivals * 60 / elapsed,
            "aircraft_minutes": dict(times),
            "time_fraction": {state: time / (c.fleet * elapsed) for state, time in times.items()}}


def port_statistics(sim, warmup_min=0):
    """Exact time averages from event intervals; not averages of occasional snapshots."""
    c, end = sim.config, sim.env.now
    start = c.time(warmup_min)
    if not 0 <= start < end:
        raise ValueError("Warm-up must precede the end of this run.")
    totals = {p: defaultdict(float) for p in sim.ports}
    for aircraft, state, port, left, right in sim.log.intervals(end, start):
        if port is None:
            continue
        duration = right - left
        if state in STAND_STATES:
            totals[port]["stand"] += duration
        if state in {"takeoff", "landing"}:
            totals[port]["pad"] += duration
        totals[port][state] += duration
    result = {}
    for port, times in totals.items():
        caps = c.capacities(port)
        length = end - start
        result[port] = {"stand_utilisation": times["stand"] / (length * caps["stands"]),
                        "charger_utilisation": times["charging"] / (length * caps["chargers"]),
                        "pad_utilisation": times["pad"] / (length * caps["pads"]),
                        "mean_dispatch_queue": times["dispatch_wait"] / length,
                        "mean_charger_queue": times["charger_wait"] / length}
    return result


def queue_trace(sim, port, state="dispatch_wait"):
    """Step-plot coordinates in minutes, with simultaneous changes combined."""
    changes = defaultdict(int)
    for aircraft, activity, location, left, right in sim.log.intervals(sim.env.now):
        if activity == state and location == port:
            # Merge numerically equal event times; avoid floating-point queue spikes.
            changes[round(left, 10)] += 1
            changes[round(right, 10)] -= 1
    times, lengths, count = [0.0], [0], 0
    for time, change in sorted(changes.items()):
        if time >= sim.env.now - 1e-10:  # Do not invent an empty queue at the cutoff
            continue
        count += change
        times.append(time * sim.config.time_scale_min)
        lengths.append(count)
    times.append(sim.env.now * sim.config.time_scale_min)
    lengths.append(count)
    return times, lengths


def run_until_wait_limit(sim, wait_limit_min=30, check_every_min=1):
    """Optional operational stop: inspect current waits at regular checkpoints.

    Detects a dispatch/charger wait reaching the threshold at a checkpoint;
    brief breaches between checkpoints can be missed. Not a physical safety test.
    """
    if wait_limit_min <= 0 or check_every_min <= 0:
        raise ValueError("Wait limit and check interval must be positive.")
    c = sim.config
    while sim.env.now < c.time(c.horizon_min) - 1e-10:
        now_min = sim.env.now * c.time_scale_min
        sim.run(min(now_min + check_every_min, c.horizon_min))
        for aircraft, history in sim.log.states.items():
            time, state, port = history[-1]
            waited = (sim.env.now - time) * c.time_scale_min
            if state in {"dispatch_wait", "charger_wait"} and waited >= wait_limit_min - 1e-8:
                return {"reason": state, "aircraft": aircraft, "port": port,
                        "detected_at_min": sim.env.now * c.time_scale_min, "wait_min": waited}
    return None
