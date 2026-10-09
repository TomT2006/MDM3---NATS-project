"""Run: python -m unittest -v test_model. Checks are separate from teaching code."""
from collections import defaultdict
from dataclasses import replace
import random
import unittest

from model import ScenarioConfig, Simulation
from results import STAND_STATES, summary, port_statistics, queue_trace, run_until_wait_limit
from mm1 import mm1


class ModelTests(unittest.TestCase):
    def check_invariants(self, sim):
        c, end = sim.config, sim.env.now
        self.assertEqual(len(sim.log.states), c.fleet)
        totals, usage, slots, departures = defaultdict(float), defaultdict(list), defaultdict(list), defaultdict(list)
        for aircraft, state, port, left, right in sim.log.intervals(end):
            totals[aircraft] += right - left
            kinds = []
            if state in STAND_STATES:
                kinds.append("stands")
            if state == "charging":
                kinds.append("chargers")
            if state in {"takeoff", "landing"}:
                kinds.append("pads")
            for kind in kinds:
                usage[port, kind].extend([(left, 1), (right, -1)])
        for duration in totals.values():
            self.assertAlmostEqual(duration, end)
        for (port, kind), events in usage.items():
            changes = defaultdict(int)
            for time, change in events:
                changes[round(time, 8)] += change
            count = 0
            for time, change in sorted(changes.items()):
                count += change
                self.assertGreaterEqual(count, 0)
                self.assertLessEqual(count, c.capacities(port)[kind])
        for time, counts in sim.log.allocations:
            self.assertEqual(sum(counts.values()), c.fleet)
            for port, count in counts.items():
                self.assertTrue(0 <= count <= c.capacities(port)["stands"])
        for f in sim.log.flights:
            slots[f.route.origin, f.departure_pad].append((f.takeoff_start, f.departure))
            slots[f.route.dest, f.arrival_pad].append((f.landing_start, f.arrival))
            departures[f.route.origin, f.route.dest].append(f.departure)
            # Match real process transitions to the reservation, not just the planned times.
            for expected, state, port in [(f.takeoff_start, "takeoff", f.route.origin),
                                          (f.departure, "cruise", None),
                                          (f.landing_start, "landing", f.route.dest)]:
                if expected < end - 1e-8:
                    self.assertTrue(any(abs(t - expected) < 1e-8 and s == state and p == port
                                        for t, s, p in sim.log.states[f.aircraft]))
        for windows in slots.values():
            windows.sort()
            for first, second in zip(windows, windows[1:]):
                self.assertLessEqual(first[1], second[0] + 1e-8)
        for times in departures.values():
            for first, second in zip(times, times[1:]):
                self.assertGreaterEqual(second - first + 1e-8, c.time(c.headway_min))
        self.assertAlmostEqual(sum(summary(sim)["time_fraction"].values()), 1)

    def test_one_aircraft_matches_hand_calculation(self):
        sim = Simulation(ScenarioConfig(fleet=1, horizon_min=65)).run()
        self.check_invariants(sim)
        starts = [p.transfer_start * 10 for p in sim.log.flights]
        for actual, expected in zip(starts, [0, 31, 62]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(starts), 3)
        f = sim.log.flights[0]
        self.assertAlmostEqual(f.departure * 10, 2)
        self.assertAlmostEqual(f.landing_start * 10, 11)
        self.assertAlmostEqual(f.arrival * 10, 12)

    def test_full_ports_progress_at_all_three_headways(self):
        for h in (1, 3, 5):
            sim = Simulation(ScenarioConfig(headway_min=h)).run()
            self.check_invariants(sim)
            self.assertGreater(summary(sim)["completed_flights"], 100)

    def test_full_three_port_cycle_progresses(self):
        sim = Simulation(ScenarioConfig(port_names=("A", "B", "C"), fleet=12)).run()
        self.check_invariants(sim)
        self.assertGreater(summary(sim)["completed_flights"], 200)

    def test_three_ports_five_aircraft(self):
        sim = Simulation(ScenarioConfig(port_names=("A", "B", "C"), fleet=5)).run()
        self.check_invariants(sim)
        self.assertEqual({f.route.dest for f in sim.log.flights}, {"A", "B", "C"})

    def test_scaling_changes_no_physical_predictions(self):
        base = ScenarioConfig(headway_min=5, horizon_min=1000)
        a = Simulation(base).run()
        b = Simulation(replace(base, time_scale_min=7, length_scale_km=11)).run()
        self.check_invariants(b)
        self.assertEqual(len(a.log.flights), len(b.log.flights))
        for x, y in zip(a.log.flights, b.log.flights):
            self.assertEqual(x.aircraft, y.aircraft)
            self.assertAlmostEqual(x.arrival * 10, y.arrival * 7)
        for state, fraction in summary(a)["time_fraction"].items():
            self.assertAlmostEqual(fraction, summary(b)["time_fraction"][state])

    def test_cutoff_in_cruise_and_charging_counts_partial_duration(self):
        sim = Simulation(ScenarioConfig(fleet=1)).run(5)
        self.assertAlmostEqual(summary(sim)["aircraft_minutes"]["cruise"], 3)
        sim.run(20)
        self.assertAlmostEqual(summary(sim)["aircraft_minutes"]["charging"], 3)
        self.assertAlmostEqual(summary(sim, warmup_min=18)["aircraft_minutes"]["charging"], 2)
        self.check_invariants(sim)

    def test_charger_scarcity_creates_a_real_queue(self):
        sim = Simulation(ScenarioConfig(chargers_per_port=1)).run()
        self.check_invariants(sim)
        self.assertGreater(port_statistics(sim)["A"]["mean_charger_queue"], 0)
        self.assertLessEqual(port_statistics(sim)["A"]["charger_utilisation"], 1)

    def test_queue_plot_has_no_floating_point_spikes(self):
        sim = Simulation(ScenarioConfig(horizon_min=100)).run()
        times, lengths = queue_trace(sim, "A")
        self.assertTrue(all(n == 0 for t, n in zip(times, lengths) if t >= 10))

    def test_ports_routes_capacities_can_differ(self):
        cfg = ScenarioConfig(port_names=("A", "B", "C"), fleet=7,
                             port_overrides={"A": {"stands": 2, "pads": 2, "chargers": 1}},
                             routes_km={("A", "B"): 22, ("A", "C"): 40,
                                        ("B", "A"): 22, ("B", "C"): 15, ("C", "A"): 40})
        sim = Simulation(cfg).run()
        self.check_invariants(sim)
        self.assertEqual({(f.route.origin, f.route.dest) for f in sim.log.flights}, set(cfg.routes_km))

    def test_several_network_sizes_and_fleets(self):
        rng = random.Random(4)
        for n in (2, 3, 4, 6):
            for full in (False, True):
                ports = tuple(f"P{i}" for i in range(n))
                distances = {(p, ports[(i + 1) % n]): rng.uniform(2, 60) for i, p in enumerate(ports)}
                cfg = ScenarioConfig(port_names=ports, fleet=n*4 if full else n+1,
                                     routes_km=distances, horizon_min=1200, headway_min=1.5,
                                     chargers_per_port=2, pads_per_port=2)
                sim = Simulation(cfg).run()
                self.check_invariants(sim)
                self.assertGreater(summary(sim)["completed_flights"], 0)

    def test_extending_run_matches_one_long_run(self):
        a, b = Simulation(), Simulation()
        a.run(20).run(40).run(720)
        b.run(720)
        self.assertEqual(summary(a), summary(b))

    def test_config_rejects_impossible_initial_conditions(self):
        for change in [dict(fleet=9), dict(fleet=0), dict(time_scale_min=0),
                       dict(port_names=("A", "A")), dict(stands_per_port=0)]:
            with self.assertRaises(ValueError):
                Simulation(ScenarioConfig(**change))
        with self.assertRaises(ValueError):
            Simulation(ScenarioConfig(initial_counts={"A": 5, "B": 3}))

    def test_optional_wait_threshold_stop(self):
        sim = Simulation(ScenarioConfig(chargers_per_port=1))
        failure = run_until_wait_limit(sim, wait_limit_min=10)
        self.assertEqual(failure["reason"], "charger_wait")
        self.assertLess(failure["detected_at_min"], sim.config.horizon_min)
        self.check_invariants(sim)

    def test_mm1_benchmark_and_dimensionless_equivalence(self):
        a = mm1(horizon_min=20000, warmup_min=2000)
        b = mm1(horizon_min=20000, warmup_min=2000, time_scale_min=7)
        self.assertAlmostEqual(a["Wq_theory_min"], 2)
        self.assertAlmostEqual(a["Wq_simulated_min"], b["Wq_simulated_min"], places=7)
        self.assertLess(abs(a["Wq_simulated_min"] - 2), 0.25)


if __name__ == "__main__":
    unittest.main()
