"""Small hand-checkable examples plus conservation checks on a whole network."""
from dataclasses import replace
import unittest
import simpy
from model import ScenarioConfig, Simulation, example_config
from queues import AircraftPadQueue
from unittest.mock import patch
from results import measurements, audit


def small(**changes):
    values = dict(port_names=('A','B'), fleet=1, seats=1,
                  initial_counts={'A':1,'B':0}, total_demand_per_hour=0,
                  holding_kw=0, mission_energy_margin=0, # Isolate original timing checks.
                  demand_stop_min=50, horizon_min=60)
    values.update(changes)
    return ScenarioConfig(**values)


class NetworkTests(unittest.TestCase):
    @patch('random.Random.expovariate', return_value=1.0)
    def test_one_trip_clock_and_energy(self, mocked):
        s = Simulation(small(), [(0,'A','B')], [(10,'A','B')]).run()
        p = s.passengers[0]
        self.assertAlmostEqual(p.departed*10,11)
        self.assertAlmostEqual(p.completed*10,23) # 9 cruise, 1 landing, 1 taxi, 1 unload
        self.assertAlmostEqual(s.aircraft[0].soc,.7)
        self.assertTrue(audit(s))

    def test_passenger_capacity_and_censoring(self):
        s = Simulation(small(), [(0,'A','B')]*3, [(10,'A','B')]).run()
        r = measurements(s)
        self.assertEqual((r['arrivals'],r['completed'],r['not_completed']),(3,1,2))
        self.assertEqual(r['load_factor'],1)
        self.assertAlmostEqual(r['timely_fraction'],1/3)

    def test_inflight_passengers_are_not_completed(self):
        s = Simulation(small(), [(0,'A','B')], [(10,'A','B')]).run(15)
        r = measurements(s)
        self.assertEqual(r['departed_not_completed'],1)
        self.assertEqual(r['completed'],0)
        self.assertTrue(audit(s))

    def test_aircraft_reassignment(self):
        c = small(fleet=2,initial_counts={'A':2,'B':0})
        s = Simulation(c,[(0,'A','B'),(11,'A','B')],[(10,'A','B','A1'),(15,'A','B','A1')]).run()
        self.assertEqual(s.flights[0].aircraft,'A1')
        self.assertEqual(s.flights[1].aircraft,'A2')
        self.assertEqual(measurements(s)['reassignments'],1)

    def test_three_port_rotation(self):
        c = small(port_names=('A','B','C'), initial_counts={'A':1,'B':0,'C':0},
                  routes_km={('A','B'):30,('B','C'):30,('C','A'):30},
                  port_overrides={'B':{'chargers':0},'C':{'chargers':0}},charge_trigger_soc=0)
        s = Simulation(c,[(0,'A','B'),(0,'B','C'),(0,'C','A')],
                       [(10,'A','B'),(25,'B','C'),(40,'C','A')]).run()
        self.assertEqual([f.aircraft for f in s.flights],['A1']*3)
        self.assertEqual(measurements(s)['completed'],3)
        self.assertTrue(audit(s))

    def test_full_stand_swap(self):
        c = small(fleet=2,stands_per_port=1,initial_counts={'A':1,'B':1},queue_capacity=2)
        s = Simulation(c,[(0,'A','B'),(0,'B','A')],[(10,'A','B'),(10,'B','A')]).run()
        self.assertEqual(measurements(s)['completed'],2)
        self.assertTrue(audit(s))

    def test_K_one_can_prevent_a_full_stand_swap(self):
        c = small(fleet=2,stands_per_port=1,initial_counts={'A':1,'B':1},queue_capacity=1)
        s = Simulation(c,[(0,'A','B'),(0,'B','A')],[(10,'A','B'),(10,'B','A')]).run()
        self.assertEqual(measurements(s)['completed'],0)
        self.assertGreater(len(s.log.deferrals),0)
        self.assertTrue(audit(s))

    def test_partial_charge_and_charger_queue(self):
        c = small(fleet=2,initial_counts={'A':2,'B':0},initial_soc=.4)
        s = Simulation(c,[],[]).run(5)
        self.assertAlmostEqual(s.aircraft[0].soc,.5)
        self.assertAlmostEqual(s.aircraft[1].soc,.4)
        self.assertGreater(measurements(s)['state_fraction']['charger_wait'],0)
        self.assertAlmostEqual(measurements(s)['grid_kwh'],10)
        self.assertTrue(audit(s))

    def test_unequipped_port_recovery(self):
        c = small(initial_counts={'A':0,'B':1},initial_soc=.4,port_overrides={'B':{'chargers':0}})
        s = Simulation(c,[],[]).run()
        self.assertEqual(s.aircraft[0].port,'A')
        self.assertAlmostEqual(s.aircraft[0].soc,.9)
        self.assertEqual(s.flights[0].kind,'reposition')
        self.assertTrue(audit(s))

    def test_battery_too_low_for_initial_escape(self):
        c = small(initial_counts={'A':0,'B':1},initial_soc=.3,port_overrides={'B':{'chargers':0}})
        with self.assertRaises(ValueError): Simulation(c)

    def test_no_charging_path_rejected(self):
        with self.assertRaises(ValueError): Simulation(small(chargers_per_port=0))

    def test_infeasible_route_rejected(self):
        with self.assertRaises(ValueError): Simulation(small(default_distance_km=300))

    def test_dimensionless_reference_does_not_change_physics(self):
        a = Simulation(example_config(horizon_min=180,demand_stop_min=120)).run()
        b = Simulation(replace(a.config,time_scale_min=3,length_scale_km=80)).run()
        ra, rb = measurements(a), measurements(b)
        for key in ('arrivals','completed','flights','reassignments','grid_kwh','mean_wait_min'):
            self.assertAlmostEqual(ra[key],rb[key],places=6,msg=key)
        self.assertTrue(audit(b))

    def test_same_demand_when_fleet_changes(self):
        c = example_config(horizon_min=180,demand_stop_min=120)
        a,b = Simulation(c).run(),Simulation(replace(c,fleet=5)).run()
        self.assertEqual([(p.arrived,p.origin,p.dest) for p in a.passengers],
                         [(p.arrived,p.origin,p.dest) for p in b.passengers])

    def test_four_port_network_with_delay(self):
        s = Simulation(example_config(ground_delay_mean_min=5)).run()
        self.assertTrue(audit(s))
        r = measurements(s)
        self.assertEqual(r['arrivals'],r['completed']+r['unassigned']+
                         r['assigned_not_departed']+r['departed_not_completed'])
        self.assertAlmostEqual(sum(r['state_fraction'].values()),1)

    def test_zero_demand_not_a_success(self):
        s = Simulation(small(),[],[]).run()
        self.assertFalse(measurements(s)['passes_targets'])
        self.assertIsNone(measurements(s)['served_fraction'])

    def test_assignment_deadline_does_not_truncate_random_service(self):
        c=small(max_assignment_delay_min=0)
        s=Simulation(c,[(0,'A','B')],[(10,'A','B')]).run()
        self.assertEqual(s.flights[0].status,'completed')
        self.assertGreater(s.flights[0].departed, c.time(10))
        self.assertTrue(audit(s))

    def test_configuration_changes_three_ports_five_aircraft(self):
        s = Simulation(example_config(port_names=('A','B','C'),fleet=5)).run()
        self.assertEqual(len(s.ports),3)
        self.assertEqual(len(s.aircraft),5)
        self.assertTrue(audit(s))

    def test_optional_costs(self):
        c = small(costs=dict(fare=50,per_flight=10,per_grid_kwh=.2,per_aircraft_hour=5))
        s = Simulation(c,[(0,'A','B')],[(10,'A','B')]).run()
        self.assertAlmostEqual(measurements(s)['scenario_contribution'],35)


class QueueTests(unittest.TestCase):
    def test_finite_aircraft_loss_and_fifo(self):
        e=simpy.Environment(); q=AircraftPadQueue(e,2,1,time_scale=1)
        q.rng.expovariate=lambda rate: 2
        first=q.offer('A1','F1','landing'); second=q.offer('A2','F2','takeoff')
        rejected=q.offer('A3','F3','landing')
        e.run()
        self.assertEqual((first.started,first.completed,second.started,second.completed),(0,2,2,4))
        self.assertTrue(rejected.blocked)
        self.assertTrue(q.audit())
        self.assertEqual(q.probabilities(0,4),[0,.5,.5])

    def test_future_permit_is_not_a_queueing_aircraft(self):
        e=simpy.Environment();q=AircraftPadQueue(e,1,1,time_scale=1)
        self.assertTrue(q.reserve(('F1','landing'),'A1'))
        e.run(until=3)
        self.assertEqual(q.probabilities(0,3),[1,0])
        self.assertFalse(q.fits())
        t=q.arrive(('F1','landing'),'A1');e.run()
        self.assertEqual(t.arrived,3)
        self.assertEqual(q.reservations,{})
        self.assertTrue(q.audit())

    def test_wrong_aircraft_cannot_consume_permit(self):
        e=simpy.Environment();q=AircraftPadQueue(e,2,1)
        q.reserve(('F1','landing'),'A1')
        with self.assertRaises(ValueError):q.arrive(('F1','landing'),'A2')

    def test_actual_network_aircraft_wait_for_pad(self):
        # Two different routes from A share its one pad. Long service creates a real queue.
        c=small(port_names=('A','B','C'),fleet=2,initial_counts={'A':2,'B':0,'C':0},
                headway_min=0,charge_trigger_soc=0,horizon_min=100,
                port_overrides={'A':{'pad_service_mean_min':5}})
        s=Simulation(c,[(0,'A','B'),(0,'A','C')],[(10,'A','B'),(10,'A','C')])
        for v in s.ports.values():v.queue.rng.expovariate=lambda rate: 5
        s.run()
        tickets=[t for t in s.ports['A'].queue.tickets if t.operation=='takeoff']
        self.assertEqual(len(tickets),2)
        self.assertGreater(tickets[1].started-tickets[1].arrived,0)
        self.assertEqual({t.aircraft for t in tickets},{'A1','A2'})
        self.assertEqual(measurements(s)['completed'],2)
        self.assertTrue(audit(s))

    def test_landing_queue_arrival_is_after_cruise(self):
        s=Simulation(small(),[(0,'A','B')],[(10,'A','B')]).run(15)
        self.assertEqual(s.ports['B'].queue.tickets,[])
        self.assertEqual(len(s.ports['B'].queue.reservations),1)
        s.run(60)
        f=s.flights[0];t=s.ports['B'].queue.tickets[0]
        self.assertEqual(t.arrived,f.departed+f.route.flight_time)
        self.assertTrue(audit(s))

    def test_one_pad_and_valid_K(self):
        with self.assertRaises(ValueError):Simulation(small(pads_per_port=2))
        for value in (0,-1,1.5):
            with self.assertRaises(ValueError): AircraftPadQueue(simpy.Environment(),value,1)


if __name__ == '__main__': unittest.main()
