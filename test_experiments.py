"""Independent hand checks for the measurements added in step 6."""
import unittest
from unittest.mock import Mock, patch

from mm1 import mm1
from model import Simulation, ScenarioConfig
from experiments import audit, interval95, two_port_bounds


class ExperimentTests(unittest.TestCase):
    def controlled_queue(self, warmup):
        # Arrivals at 0.5 and 1.5, each service takes 2. Ends at 2.5 and 4.5.
        arrivals, services = Mock(), Mock()
        arrivals.expovariate.side_effect = [0.5, 1.0, 10.0]
        services.expovariate.side_effect = [2.0, 2.0]
        with patch("mm1.random.Random", side_effect=[arrivals, services]):
            return mm1(1, 2, horizon_min=3, warmup_min=warmup, time_scale_min=1)

    def test_time_integrals_match_hand_calculation(self):
        r = self.controlled_queue(0)
        self.assertAlmostEqual(r["Wq_simulated_min"], 0.5)
        self.assertAlmostEqual(r["Lq_simulated"], 1/3)
        self.assertAlmostEqual(r["busy_fraction"], 5/6)
        self.assertAlmostEqual(sum(r["state_probabilities"].values()), 1)

    def test_warmup_clips_existing_customers_correctly(self):
        r = self.controlled_queue(1)
        self.assertAlmostEqual(r["Wq_simulated_min"], 1)
        self.assertAlmostEqual(r["Lq_simulated"], 0.5)
        self.assertAlmostEqual(r["busy_fraction"], 1)

    def test_empty_tail_and_empty_cohort(self):
        r = mm1(horizon_min=1e-6, warmup_min=0, seed=42)
        self.assertIsNone(r["Wq_simulated_min"])
        self.assertEqual(r["state_probabilities"], {0: 1.0})
        self.assertEqual(r["busy_fraction"], 0)

    def test_confidence_interval_known_t_value(self):
        result = interval95([1, 3])
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["upper"] - 2, 12.706204736, places=7)

    def test_audit_detects_a_broken_reservation(self):
        sim = Simulation(ScenarioConfig(horizon_min=100)).run()
        self.assertTrue(audit(sim)["passed"])
        sim.log.flights[0].arrival = 1e6
        with self.assertRaises(AssertionError):
            audit(sim)

    def test_baseline_bounds_match_hand_calculation(self):
        b = two_port_bounds(ScenarioConfig())
        self.assertAlmostEqual(b["fleet"], 480/31)
        self.assertAlmostEqual(b["charging"], 48)
        self.assertAlmostEqual(b["pads"], 60)
        self.assertAlmostEqual(b["headway"], 40)
        unbounded = two_port_bounds(ScenarioConfig(headway_min=0, charge_min=0, passengers=0))
        self.assertIsNone(unbounded["headway"])
        self.assertIsNone(unbounded["charging"])
        self.assertIsNone(unbounded["stands"])


if __name__ == "__main__":
    unittest.main()
