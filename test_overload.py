from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from model import example_config,Simulation
from overload import (OverloadRule,detect_overload,passenger_arrivals,demand_profile,
                      cost_components,score_run,timetable_ceiling,weighted_score)
from uncertainty import bootstrap_mean,exact_probability,paired_comparison,holm_adjust
from cost_optimiser_integrated import build_config,get_sweep_values,BASELINE_PARAMETERS
from test_safety import holding_example


class OverloadTests(unittest.TestCase):
    def test_exact_persistence_duration(self):
        rule=OverloadRule(queue_limit=10,persistence_min=3,growth_window_min=2,growth_per_hour=0)
        q=np.array([0,0,10,10,10,10,10])
        self.assertEqual(detect_overload(np.arange(7),q,rule,6),5)
        self.assertIsNone(detect_overload(np.arange(7),q,rule,4))

    def test_growth_and_size_are_both_required(self):
        rule=OverloadRule(queue_limit=10,persistence_min=2,growth_window_min=2,growth_per_hour=30)
        self.assertIsNone(detect_overload(np.arange(8),np.full(8,10),rule,7))
        self.assertEqual(detect_overload(np.arange(8),np.arange(8)*5,rule,7),4)
        self.assertIsNone(detect_overload(np.arange(8),np.array([0,0,0,np.nan,np.nan,np.nan,np.nan,np.nan]),rule,7))

    def test_rate_scaling_uses_common_unit_intensity(self):
        c=example_config(demand_stop_min=60,horizon_min=90)
        low=passenger_arrivals(c,36,1);high=passenger_arrivals(c,72,1)
        for route in c.network():
            a=[t for t,o,d in low if (o,d)==route];b=[t for t,o,d in high if (o,d)==route]
            self.assertTrue(np.allclose(np.array(a)/2,np.array(b[:len(a)])))
        self.assertEqual(low,passenger_arrivals(c,36,1))
        self.assertNotEqual(low,passenger_arrivals(c,36,2))

    def test_profile_mean_and_expected_counts(self):
        c=example_config(demand_stop_min=60,horizon_min=90)
        for profile in ('constant','two_peaks'):
            self.assertAlmostEqual(sum((b-a)*w for a,b,w in demand_profile(profile)),1)
            counts=[len(passenger_arrivals(c,60,s,profile)) for s in range(400)]
            self.assertLess(abs(np.mean(counts)-60),2)
            self.assertLess(abs(np.var(counts)-60),15)

    def test_wait_area_counts_unfinished_people(self):
        p=[SimpleNamespace(arrived=0,departed=2,completed=3),
           SimpleNamespace(arrived=1,departed=None,completed=None)]
        sim=SimpleNamespace(config=SimpleNamespace(time_scale_min=10,fleet=2,horizon_min=100),
                            env=SimpleNamespace(now=4),passengers=p,flights=[],ports={})
        x=cost_components(sim)
        self.assertEqual(x['total_wait_min'],50)
        self.assertEqual(x['in_transit_min'],10)
        self.assertAlmostEqual(x['aircraft_hours'],80/60)
        self.assertAlmostEqual(x['planned_aircraft_hours'],200/60)

    def test_early_failure_is_ineligible_and_tail_is_missing(self):
        s=holding_example(service_minutes=200,until=250)
        row,times,q=score_run(s)
        self.assertFalse(row['feasible'])
        self.assertTrue(row['safety_failure'])
        self.assertTrue(np.all(np.isnan(q[times>row['observed_min']+1e-7])))

    def test_bootstrap_and_pairing(self):
        ci=bootstrap_mean([1,2,3,4],resamples=1000)
        self.assertEqual(ci['mean'],2.5)
        a=[dict(seed=i,x=i) for i in range(30)]
        b=[dict(seed=i,x=i+5) for i in reversed(range(30))]
        result=paired_comparison(a,b,'x',resamples=1000)
        self.assertEqual((result['mean'],result['low'],result['high']),(5,5,5))
        with self.assertRaises(ValueError):paired_comparison(a,b[:-1],'x')

    def test_zero_failures_does_not_mean_zero_risk(self):
        p=exact_probability([False]*1000)
        self.assertEqual(p['probability'],0)
        self.assertGreater(p['high'],0)
        self.assertLess(p['high'],.005)
        self.assertTrue(np.allclose(holm_adjust([.01,.03,.2]),[.03,.06,.2]))

    def test_cost_weights_do_not_change_overload_event(self):
        q=np.arange(121)
        rule=OverloadRule()
        event=detect_overload(np.arange(121),q,rule,120)
        self.assertIsNotNone(event)
        x=dict(total_wait_min=10,uncompleted_passengers=2,flights=3,aircraft_hours=4)
        self.assertNotEqual(weighted_score(x),weighted_score(x,dict(passenger_wait_min=20)))
        self.assertEqual(event,detect_overload(np.arange(121),q,rule,120))

    def test_timetable_and_compatibility_validation(self):
        self.assertEqual(timetable_ceiling(example_config()),96)
        c=build_config(BASELINE_PARAMETERS,1)
        self.assertEqual(c.capacities('A')['chargers'],2)
        bad={**BASELINE_PARAMETERS,'stands_per_port':1}
        with self.assertRaises(ValueError):Simulation(build_config(bad,1))
        with self.assertRaises(ValueError):get_sweep_values('fleet',[3.5],[],BASELINE_PARAMETERS)

    def test_replication_streams_do_not_overlap_between_ports(self):
        a=Simulation(example_config(seed=1),passenger_arrivals=[])
        b=Simulation(example_config(seed=2),passenger_arrivals=[])
        states=[v.queue.rng.getstate() for s in (a,b) for v in s.ports.values()]
        self.assertEqual(len(states),len(set(states)))
        delays=[r.getstate() for s in (a,b) for r in s.delay_rng.values()]
        self.assertEqual(len(delays),len(set(delays)))
        c=Simulation(example_config(seed=1,fleet=5),passenger_arrivals=[])
        self.assertEqual(a.ports['A'].queue.rng.getstate(),c.ports['A'].queue.rng.getstate())

    def test_dimensionless_units_preserve_external_demand_results(self):
        c=example_config(demand_stop_min=60,horizon_min=90)
        arrivals=passenger_arrivals(c,60,123)
        x=score_run(Simulation(c,passenger_arrivals=arrivals).run())[0]
        y=score_run(Simulation(replace(c,time_scale_min=3),passenger_arrivals=arrivals).run())[0]
        for key in ('score','mean_observed_wait_min','completed','monitored_peak_queue'):
            self.assertAlmostEqual(x[key],y[key],places=6)


if __name__=='__main__':unittest.main()
