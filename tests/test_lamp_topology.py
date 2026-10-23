import unittest
import numpy as np
from armor_native.observability import lamp_evidence, CYAN_RULE
from armor_native.lamp_topology import separated_lamp_evidence

class DistinctStripTests(unittest.TestCase):
    def test_two_red_or_cyan_strips(self):
        for color in [(255, 40, 20), (20, 230, 255)]:
            image=np.full((64,64,3),40,np.uint8)
            image[8:56,20:24]=color;image[8:56,44:48]=color
            c=[[21.5,54],[21.5,9],[45.5,9],[45.5,54]]
            self.assertTrue(lamp_evidence(image,c,CYAN_RULE)['accepted'])
            self.assertTrue(separated_lamp_evidence(image,c)['accepted'])
    def test_single_strip_edges_and_white_core_are_not_two_lamps(self):
        for white in [False,True]:
            image=np.full((64,64,3),40,np.uint8)
            image[8:56,20:31]=[20,230,255]
            if white:image[8:56,23:28]=255
            c=[[21,54],[21,9],[29,9],[29,54]]
            self.assertTrue(lamp_evidence(image,c,CYAN_RULE)['accepted'])
            self.assertFalse(separated_lamp_evidence(image,c)['accepted'])

if __name__=='__main__':unittest.main()
