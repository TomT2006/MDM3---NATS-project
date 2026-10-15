from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from model import example_config
from experiments import interval, fingerprint, save_results, load_results, capacity_summary


class ExperimentTests(unittest.TestCase):
    def test_cache_rejects_changed_configuration(self):
        c=example_config()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.json'
            save_results({'fingerprint':fingerprint(c)},path)
            load_results(path,c)
            with self.assertRaises(ValueError): load_results(path,replace(c,fleet=5))

    def test_interval_is_across_replications(self):
        r=interval([1,2,3,4,5])
        self.assertEqual(r['mean'],3)
        self.assertEqual(r['n'],5)
        self.assertLess(r['low'],3)
        self.assertGreater(r['high'],3)

    def test_capacity_requires_each_seed_to_pass(self):
        rows=[dict(group='demand',level=20,metrics={'passes_targets':v}) for v in (True,False)]
        self.assertIsNone(capacity_summary(rows,example_config())['demand']['best_tested'])


if __name__=='__main__': unittest.main()
