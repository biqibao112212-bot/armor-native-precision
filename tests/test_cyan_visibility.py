import unittest
import numpy as np
from armor_native.observability import lamp_evidence,RULE,CYAN_RULE

class CyanVisibilityTest(unittest.TestCase):
 def fixture(self,left,right):
  image=np.full((100,100,3),45,np.uint8)
  image[18:73,18:23]=left;image[18:73,78:83]=right
  return image,np.array([[20,70],[20,20],[80,20],[80,70]],float)
 def test_real_bright_cyan_led_color_remains_visible(self):
  im,c=self.fixture([114,235,245],[113,236,245])
  self.assertFalse(lamp_evidence(im,c,rule=RULE)['accepted'])
  self.assertTrue(lamp_evidence(im,c,rule=CYAN_RULE)['accepted'])
  self.assertEqual(RULE['chroma_min'],CYAN_RULE['chroma_min'])
 def test_color_and_two_strip_negatives(self):
  for color,expected in [([0,0,255],True),([255,0,0],True),([255,255,255],False),([0,255,0],False),([255,255,0],False),([45,45,45],False)]:
   im,c=self.fixture(color,color)
   self.assertEqual(lamp_evidence(im,c,rule=RULE)['accepted'],expected)
   self.assertEqual(lamp_evidence(im,c,rule=CYAN_RULE)['accepted'],expected)
  im,c=self.fixture([114,235,245],[45,45,45])
  self.assertFalse(lamp_evidence(im,c,rule=CYAN_RULE)['accepted'])

if __name__=='__main__':unittest.main()
