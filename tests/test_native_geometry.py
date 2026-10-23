"""Parity regression for the optional compiled backend (set ARMOR_NATIVE_LIBRARY)."""
import os
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from armor_native.native_geometry import NativeGeometry
from armor_native.glyph import fit_glyph_pose
from armor_native.observability import lamp_evidence,RULES
from test_glyph import template,scene

@pytest.fixture(scope="module")
def native():
    path=os.environ.get("ARMOR_NATIVE_LIBRARY")
    if not path:
        pytest.skip("Build native geometry and set ARMOR_NATIVE_LIBRARY")
    return NativeGeometry(path)

@pytest.mark.parametrize("distance",[.4,1.,3.,5.,8.])
def test_reference_geometry_parity(native,distance):
    K,q,rv,tv,_=scene()
    tv=np.array([.03*distance,.025*distance,distance])
    points=cv2.projectPoints(template(),rv,tv,K,np.zeros(5))[0].reshape(-1,2)
    rng=np.random.default_rng(42)
    for kind in ["exact","noise","outlier","missing"]:
        p=points.copy();prob=np.full(16,.9)
        if kind!="exact":p+=rng.normal(0,.12,p.shape)
        if kind=="outlier":p[8]+=[4.,-2.]
        if kind=="missing":prob[[4,7,11,14]]=0
        ref=fit_glyph_pose(p,prob,K,template(),q)
        got=native.fit_glyph_pose(p,prob,K,template(),q)
        assert (ref is None)==(got is None)
        if ref:
            np.testing.assert_allclose(got["corners"],ref["corners"],atol=1e-3,rtol=0)
            np.testing.assert_allclose(got["xyz_m"],ref["xyz_m"],atol=1e-5,rtol=0)
            assert got["used_points"]==ref["used_points"]

@pytest.mark.parametrize("version",list(RULES))
def test_lamp_pixel_and_boundary_parity(native,version):
    rng=np.random.default_rng(9);image=rng.integers(0,256,(130,170,3),dtype=np.uint8)
    quads=[np.array([[20.,95.],[20.,20.],[100.,20.],[100.,95.]]),
           np.array([[-2.,95.],[-2.,20.],[100.,20.],[100.,95.]]),
           np.zeros((4,2)),np.full((4,2),np.nan)]
    quads += [rng.uniform([-40,-40],[210,170],(4,2)) for _ in range(80)]
    for c in quads:
        ref=lamp_evidence(image,c,RULES[version]);got=native.lamp_evidence(image,c,RULES[version])
        assert ref.keys()==got.keys()
        for k in ref:
            if isinstance(ref[k],(list,float)):
                np.testing.assert_allclose(got[k],ref[k],rtol=0,atol=1e-10)
            else:assert got[k]==ref[k]

def test_fallback_and_low_support(native):
    K,q,rv,tv,p=scene()
    assert native.fit_glyph_pose(p,np.zeros(16),K,template(),q) is None
    for mode in ["direct","lamp_level"]:
        ref=fit_glyph_pose(p,np.ones(16),K,template(),q,mode)
        got=native.fit_glyph_pose(p,np.ones(16),K,template(),q,mode)
        assert ref==got
