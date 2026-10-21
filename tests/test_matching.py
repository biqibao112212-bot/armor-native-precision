import unittest
from armor_native.matching import match_centers
class MatchingTests(unittest.TestCase):
 def test_refusal_cannot_steal_valid(self):
  p=[dict(center=[0,0],xyz=None),dict(center=[1,0],xyz=[0,0,4])];g=[dict(center=[0,0],extent_px=20)]
  self.assertEqual(match_centers(p,g),{0:1})
 def test_gated_assignment_and_two_distinct_targets(self):
  p=[dict(center=[2,0],xyz=[0,0,4]),dict(center=[50,0],xyz=[0,0,4]),dict(center=[90,0],xyz=None)];g=[dict(center=[0,0]),dict(center=[50,0]),dict(center=[90,0])]
  self.assertEqual(match_centers(p,g),{0:0,1:1,2:2})
 def test_far_candidate_not_match(self):
  self.assertEqual(match_centers([dict(center=[20,0],xyz=[0,0,4])],[dict(center=[0,0])]),{})
