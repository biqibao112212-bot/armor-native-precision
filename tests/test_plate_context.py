import numpy as np
import torch
from armor_native.plate import NativePlatePrecision
from armor_native.plate_context import DilatedContext

def test_identity_initialization_and_learning_path():
    torch.manual_seed(19);x=torch.randn(2,64,24,24,requires_grad=True)
    model=DilatedContext(64,4)
    y=model(x)
    assert torch.equal(y,x)
    y.square().mean().backward()
    assert model.branch[-1].weight.grad.abs().sum()>0
    assert torch.isfinite(x.grad).all()

def test_transfer_preserves_native_outputs():
    torch.manual_seed(20);base=NativePlatePrecision().eval();wide=NativePlatePrecision(context_dilations=(2,4,8)).eval()
    missing,extra=wide.load_state_dict(base.state_dict(),strict=False)
    assert not extra and missing and all(k.startswith('wide_context.') for k in missing)
    with torch.inference_mode():
        for size in [128,192]:
            x=torch.rand(1,3,size,size);a=base(x);b=wide(x)
            for k in a:assert torch.equal(a[k],b[k]),(size,k,(a[k]-b[k]).abs().max())
