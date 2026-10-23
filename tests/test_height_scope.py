import numpy as np
import pytest
from collections import Counter
from armor_native.height_scope import height_region, DistanceHeightSelector

def test_signed_scope_and_boundaries():
    assert [height_region(h) for h in [-.4,-.15001,0,.1,.15,.15001,.4]]==[2,2,0,0,0,1,1]
    with pytest.raises(ValueError):height_region(.40001)
    with pytest.raises(ValueError):height_region(-.40001)

def test_height_selector_preserves_distance_and_signed_tail_coverage():
    keys=[(d,0,"normal",128,h) for d in range(8) for h in range(3)]
    sampler=DistanceHeightSelector(keys);rng=np.random.default_rng(19274)
    counts=Counter((lambda k:(k[0],k[-1]))(sampler.draw(rng)) for _ in range(64000))
    for d in range(8):
        total=sum(counts[d,h] for h in range(3))
        assert abs(total/64000-1/8)<.01
        assert abs(counts[d,0]/total-.7)<.02
        assert abs(counts[d,1]/total-.15)<.02
        assert abs(counts[d,2]/total-.15)<.02

def test_missing_height_is_explicitly_renormalized():
    sampler=DistanceHeightSelector([(0,0,"dark",128,0),(0,0,"dark",128,2)])
    assert sampler.probabilities()["0"]=={"0":.7/.85,"2":.15/.85}
    rng=np.random.default_rng(19)
    assert {sampler.draw(rng)[-1] for _ in range(100)}=={0,2}
