import unittest
import numpy as np
from collections import Counter
from armor_native.sampling import GroupSelector

class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.keys=[(band, k, 'normal', 128+32*k)
                   for band,n in enumerate([29,22,11,7,8,8,8,8])
                   for k in range(n)]

    def test_distance_draw_does_not_inherit_crop_size_imbalance(self):
        s=GroupSelector(self.keys,'distance');rng=np.random.default_rng(9041)
        counts=Counter(s.draw(rng)[0] for _ in range(80000))
        for band in range(8):
            self.assertLess(abs(counts[band]/80000-.125),.005)
            self.assertEqual(s.range_probabilities()[str(band)],.125)
        # Historical flat-group selection would assign >50% to ranges below2m.
        legacy=GroupSelector(self.keys,'groups').range_probabilities()
        self.assertGreater(legacy['0']+legacy['1'],.5)

    def test_historical_rng_draw_order_preserved(self):
        s=GroupSelector(self.keys,'groups');a=np.random.default_rng(62);b=np.random.default_rng(62)
        keys=sorted(self.keys)
        for _ in range(100):self.assertEqual(s.draw(a),keys[int(b.integers(len(keys)))])

    def test_empty_ranges_are_not_synthesized(self):
        s=GroupSelector([(2,0,'normal',128),(7,0,'dark',128)],'distance')
        self.assertEqual(s.range_probabilities(),{'2':.5,'7':.5})
        with self.assertRaises(ValueError):GroupSelector([],'distance')

if __name__=='__main__':unittest.main()
