import numpy as np
from armor_native.observability import lamp_evidence

def test_two_supported_lamps_and_missing_lamp_refusal():
 c=np.array([[10,30],[10,10],[40,10],[40,30]],float)
 for color in [[10,150,250],[250,20,10]]:
  im=np.zeros((48,64,3),np.uint8);im[10:31,9:12]=color;im[10:31,39:42]=color
  assert lamp_evidence(im,c)['accepted']
  im[:,39:42]=0
  assert not lamp_evidence(im,c)['accepted']

def test_neutral_body_and_collapsed_lamp_do_not_establish_four_corners():
 c=np.array([[10,30],[10,10],[40,10],[40,30]],float);im=np.full((48,64,3),180,np.uint8)
 assert not lamp_evidence(im,c)['accepted']
 im[10:31,9:12]=[10,150,250]
 assert not lamp_evidence(im,np.array([[10,30],[10,10],[10,10],[10,30]],float))['accepted']
 assert not lamp_evidence(im,np.full((4,2),np.nan))['accepted']

def test_half_pixel_rule_keeps_two_lamps_and_refuses_single_lamp():
 from armor_native.observability import CYAN_PIXEL_CENTER_RULE
 c=np.array([[10.2,30.2],[10.2,10.2],[40.2,10.2],[40.2,30.2]])
 for color in [[10,240,250],[250,20,10]]:
  im=np.zeros((48,64,3),np.uint8);im[10:31,9:12]=color;im[10:31,39:42]=color
  assert lamp_evidence(im,c,CYAN_PIXEL_CENTER_RULE)['accepted']
  im[:,39:42]=0
  assert not lamp_evidence(im,c,CYAN_PIXEL_CENTER_RULE)['accepted']
 for color in [[180,180,180],[10,240,10],[240,240,10]]:
  im=np.full((48,64,3),color,np.uint8)
  assert not lamp_evidence(im,c,CYAN_PIXEL_CENTER_RULE)['accepted']
