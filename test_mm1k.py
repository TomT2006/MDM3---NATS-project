import unittest
from mm1k import theory, simulate, transient


class BenchmarkTests(unittest.TestCase):
    def test_uniform_at_equal_rates(self):
        r=theory(1,1,4)
        self.assertEqual(r['pn'],[.2]*5)
        self.assertAlmostEqual(r['Lq'],1.2)
        self.assertAlmostEqual(r['Wq'],1.5)

    def test_loss_system_K_one(self):
        r=theory(2,1,1)
        self.assertAlmostEqual(r['blocking'],2/3)
        self.assertEqual(r['Wq'],0)
        self.assertAlmostEqual(r['W'],1)

    def test_flow_and_little_law(self):
        for rho in (.2,.999999,1,1.000001,2,1000):
            r=theory(rho,1,8)
            self.assertAlmostEqual(sum(r['pn']),1)
            self.assertAlmostEqual(rho*(1-r['blocking']),r['effective_rate'])
            self.assertAlmostEqual(r['Lq'],r['effective_rate']*r['Wq'])
            self.assertAlmostEqual(r['L'],r['effective_rate']*r['W'])

    def test_detailed_balance(self):
        r=theory(1.4,2,5)
        for a,b in zip(r['pn'],r['pn'][1:]): self.assertAlmostEqual(1.4*a,2*b)

    def test_simulated_loss_has_no_wait(self):
        r=simulate(2,1,1,duration_min=2000,warmup_min=100,seed=3)
        self.assertEqual(r['Wq'],0)
        self.assertEqual(r['admitted'],r['observed_waits'])
        self.assertGreater(r['blocked'],0)

    def test_benchmark_scale_invariance(self):
        a=simulate(duration_min=1000,warmup_min=100,time_scale_min=1)
        b=simulate(duration_min=1000,warmup_min=100,time_scale_min=10)
        for k in ('blocking','Wq','Lq','effective_rate'):
            self.assertAlmostEqual(a[k],b[k],places=9)

    def test_invalid_rates(self):
        for lam,mu,k in [(0,1,4),(1,0,4),(1,1,0),(float('nan'),1,4)]:
            with self.assertRaises(ValueError): theory(lam,mu,k)

    def test_transient_converges_to_stationary_distribution(self):
        p=transient(1.5,1,4,[0,100])
        self.assertEqual(list(p[0]),[1,0,0,0,0])
        for a,b in zip(p[1],theory(1.5,1,4)['pn']): self.assertAlmostEqual(a,b,places=9)


if __name__ == '__main__': unittest.main()
