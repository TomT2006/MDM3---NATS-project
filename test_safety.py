"""Energy gates, charger contention, reassignment and deliberately adverse tails."""
from dataclasses import replace
import unittest
from scipy.stats import erlang
from model import Simulation, example_config
from results import audit, measurements
from test_model import small


def holding_example(service_minutes=20, until=60):
    c = small(fleet=2, initial_counts={'A':1,'B':1}, holding_kw=30,
              mission_energy_margin=.1, charge_trigger_soc=0,
              demand_stop_min=200, horizon_min=250)
    s = Simulation(c, [(0,'A','B'),(0,'B','A')], [(10,'A','B'),(10,'B','A')])
    s.ports['A'].queue.rng.expovariate = lambda rate: 1
    s.ports['B'].queue.rng.expovariate = lambda rate: service_minutes
    return s.run(until)


class SafetyTests(unittest.TestCase):
    def test_budget_includes_hold_margin_and_multileg_charger_recovery(self):
        c = small(port_names=('A','B','C'), initial_counts={'A':1,'B':0,'C':0},
                  routes_km={('A','B'):30,('B','C'):30,('C','A'):30},
                  port_overrides={'B':{'chargers':0},'C':{'chargers':0}},
                  holding_kw=30, mission_energy_margin=.1)
        s = Simulation(c, [], [])
        allowance = erlang.ppf(.99,3)*30/(60*100)
        self.assertAlmostEqual(s.required(s.routes['A','B']),3*(.2*1.1+allowance)+.15)
        self.assertEqual(s.safety.recovery_path('B'),('B','C','A'))

    def test_K_one_has_zero_landing_wait_allowance(self):
        s = Simulation(small(queue_capacity=1,holding_kw=30),[],[])
        self.assertEqual(s.safety.hold_minutes,{'A':0,'B':0})

    def test_insufficient_budget_rejects_service_but_allows_recovery(self):
        c = example_config(fleet=1, initial_counts={'A':0,'B':1,'C':0,'D':0},
                           initial_soc=.5, total_demand_per_hour=0,
                           demand_stop_min=30,horizon_min=40)
        s = Simulation(c,[(0,'B','D')],[(10,'B','D')]).run(15)
        self.assertIsNone(s.flights[0].departed)
        self.assertTrue(any(f.kind=='reposition' and f.safety.allowed for f in s.flights))
        self.assertTrue(audit(s))

    def test_charger_workload_forecast_respects_fcfs_and_multiple_users(self):
        s = Simulation(small(fleet=3,initial_counts={'A':3,'B':0},initial_soc=.4),[],[]).run(5)
        wait, end = s.safety.charge_forecast('A',10,.4,'A3')
        self.assertAlmostEqual(wait,40)
        self.assertAlmostEqual(end,75)
        c = replace(s.config,port_overrides={'A':{'chargers':2}})
        t = Simulation(c,[],[]).run(5)
        wait, end = t.safety.charge_forecast('A',10,.4,'A3')
        self.assertAlmostEqual(wait,15)
        self.assertAlmostEqual(end,50)

    def test_long_charging_queue_prevents_dispatch(self):
        c = small(fleet=2,initial_counts={'A':1,'B':1},initial_soc=.4,
                  port_overrides={'B':{'charger_kw':10}},max_charger_wait_min=1)
        s = Simulation(c,[(0,'A','B')],[(30,'A','B')]).run()
        self.assertIsNone(s.flights[0].departed)
        self.assertTrue(any('charging queue' in d['reason'] for d in s.log.safety_checks))
        self.assertTrue(audit(s))

    def test_actual_holding_uses_energy_and_can_exceed_allowance(self):
        s = holding_example()
        a = s.aircraft[0]
        self.assertAlmostEqual(a.soc,.645) # 20 kWh mission + 11 min at 30 kW.
        self.assertAlmostEqual(measurements(s)['holding_kwh'],5.5)
        self.assertEqual(measurements(s)['holding_allowance_exceedances'],1)
        self.assertFalse(s.failure_event.triggered)
        self.assertTrue(audit(s))

    def test_partial_holding_is_counted_at_observation_end(self):
        s = holding_example(until=25)
        self.assertAlmostEqual(s.aircraft[0].soc,.675)
        self.assertAlmostEqual(measurements(s)['holding_kwh'],2.5)
        self.assertTrue(audit(s))

    def test_long_tail_stops_run_at_reserve_without_clipping_or_resuming(self):
        s = holding_example(service_minutes=200,until=250)
        self.assertTrue(s.failure_event.triggered)
        self.assertAlmostEqual(s.env.now*s.config.time_scale_min,130)
        self.assertAlmostEqual(s.aircraft[0].soc,.15)
        self.assertEqual(measurements(s)['safety_failures'],1)
        self.assertFalse(measurements(s)['passes_targets'])
        self.assertTrue(audit(s))
        with self.assertRaises(ValueError): s.run(250)

    def test_charging_aircraft_is_swapped_for_ready_aircraft(self):
        # The first aircraft uses more energy on A->B and has to charge at B.
        c = small(port_names=('A','B','C'),fleet=2,initial_counts={'A':1,'B':0,'C':1},
                  routes_km={('A','B'):70,('C','B'):10,('B','A'):30,('B','C'):30},
                  demand_stop_min=90,horizon_min=100,charger_kw=30)
        s = Simulation(c,[(0,'A','B'),(0,'C','B'),(0,'B','C')],
                       [(10,'A','B','A1'),(20,'C','B','A2'),(40,'B','C','A1')])
        for v in s.ports.values():v.queue.rng.expovariate=lambda rate: 1
        s.run()
        self.assertEqual(s.flights[2].aircraft,'A2')
        self.assertTrue(audit(s))

    def test_safety_parameters_validate(self):
        for field, value in [('holding_kw',-1),('holding_quantile',1),
                             ('holding_quantile',0),('mission_energy_margin',-1),
                             ('max_charger_wait_min',-1)]:
            with self.assertRaises(ValueError):Simulation(replace(small(),**{field:value}))

    def test_default_dispatches_approved_and_units_preserve_safety_metrics(self):
        c = example_config(horizon_min=180,demand_stop_min=120)
        a = Simulation(c).run()
        b = Simulation(replace(c,time_scale_min=7,length_scale_km=80)).run()
        x,y = measurements(a), measurements(b)
        for key in ('holding_kwh','minimum_soc','dispatches_approved','safety_failures',
                    'rejected_candidate_pairs','completed'):
            self.assertAlmostEqual(x[key],y[key],places=6,msg=key)
        self.assertTrue(x['all_dispatches_approved'])
        self.assertTrue(audit(a))
        self.assertTrue(audit(b))


if __name__ == '__main__': unittest.main()
