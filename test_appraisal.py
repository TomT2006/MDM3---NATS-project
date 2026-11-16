"""Hand-checkable valuation, accounting and missing-outcome checks."""
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from appraisal import *


class AppraisalTests(unittest.TestCase):
    def test_published_values_and_units(self):
        i=load_inputs();c=AppraisalCase()
        self.assertAlmostEqual(time_value(i,c,2026),7.207272161207625)
        self.assertAlmostEqual(energy_value(i,c,2026),.15412714753577278)

    def test_discount_clock_starts_at_appraisal_not_opening(self):
        c=AppraisalCase(opening_year=2030)
        self.assertAlmostEqual(discount_factor(2030,c),1/1.03**7)
        self.assertAlmostEqual(discount_factor(2057,c),1/(1.03**33*1.025))

    def test_hand_calculated_time_and_energy(self):
        p=[dict(wait_hours_saved=1,other_hours_saved=.5,extra_energy_kwh=10)]*2
        i=load_inputs();c=AppraisalCase();r=daily_components(p,i,c)
        np.testing.assert_allclose(r['time_benefit'],[18.01818040301906]*2)
        np.testing.assert_allclose(r['extra_energy_cost'],[1.5412714753577278]*2)
        np.testing.assert_allclose(r['partial_net_benefit'],[16.47690892766133]*2)

    def test_no_difference_and_negative_benefit(self):
        i=load_inputs();c=AppraisalCase()
        zero=[dict(wait_hours_saved=0,other_hours_saved=0,extra_energy_kwh=0)]*2
        np.testing.assert_allclose(benefit_budget(zero,i,c),0)
        harm=[dict(wait_hours_saved=-1,other_hours_saved=0,extra_energy_kwh=1)]*2
        self.assertTrue(np.all(benefit_budget(harm,i,c)<0))

    def test_extra_unfinished_has_a_cost_not_a_benefit(self):
        r=residual_time_adjustment([{'extra_unfinished':1},{'extra_unfinished':-1}],load_inputs(),AppraisalCase(),2026,60)
        np.testing.assert_allclose(r,[-7.207272161207625,7.207272161207625])

    def test_capital_cost_at_switching_value_has_zero_mean_npv(self):
        pairs,_=load_pairs();i=load_inputs();c=AppraisalCase()
        capital=float(upfront_budget(pairs,i,c,annual_additional_cost=12345).mean())
        result=net_present_value(pairs,i,c,upfront_cost=capital,annual_additional_cost=12345,renewals={},residual_value=0)
        self.assertAlmostEqual(float(result.mean()),0,places=7)

    def test_missing_cost_is_not_zero(self):
        with self.assertRaises(ValueError):
            net_present_value([],load_inputs(),AppraisalCase(),upfront_cost=None,annual_additional_cost=0,renewals={},residual_value=0)

    def test_invalid_and_out_of_source_years(self):
        with self.assertRaises(ValueError):AppraisalCase(purpose='business').validate()
        with self.assertRaises(ValueError):AppraisalCase(operating_days=400).validate()
        with self.assertRaises(ValueError):time_value(load_inputs(),AppraisalCase(),2100)

    def test_all_pairs_and_unfinished_people_retained(self):
        pairs,meta=load_pairs()
        self.assertEqual(len(pairs),80)
        self.assertEqual(sum(p['unfinished_12h_a'] for p in pairs),61)
        self.assertEqual(sum(p['unfinished_12h_b'] for p in pairs),77)
        self.assertTrue(all(p['arrivals_a']==p['arrivals_b'] for p in pairs))

    def test_energy_accounting_matches_consumption(self):
        data=json.loads((HERE/'data/simulation_results.json').read_text())
        for run in data['runs']:
            r=run['original_12h']
            self.assertAlmostEqual(sum(r['grid_kwh_by_port'].values())+r['terminal_replenishment_kwh'],r['mission_kwh']+r['holding_kwh'],places=7)

    def test_duplicate_seed_and_safety_stop_are_rejected(self):
        source=json.loads((HERE/'data/simulation_results.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'
            duplicate={**source,'runs':source['runs']+[source['runs'][0]]}
            p.write_text(json.dumps(duplicate))
            with self.assertRaises(ValueError):load_pairs(p)
            source['runs'][0]['original_12h']['safety_failure']=True
            p.write_text(json.dumps(source))
            with self.assertRaises(ValueError):load_pairs(p)


if __name__=='__main__':unittest.main()
