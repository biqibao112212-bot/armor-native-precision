import numpy as np,torch
from armor_native.plate import dense_targets,fit_dense_quad,NativePlatePrecision

def test_exact_projective_field_recovers_native_corners():
    for corners in [np.array([[32,91],[35,40],[97,44],[102,89.]],np.float32),np.array([[40.17,70.23],[40.91,53.18],[83.72,55.57],[85.33,72.99]],np.float32)]:
        uv,mask=dense_targets(corners,128)
        got=fit_dense_quad(uv,mask[0],corners+.5)
        assert got is not None
        assert np.max(np.abs(got-corners))<.0001

def test_degenerate_foreground_has_no_solution():
    assert fit_dense_quad(np.zeros((2,128,128)),np.zeros((128,128)),np.zeros((4,2))) is None

def test_native_shapes_and_gradient_paths():
    torch.set_num_threads(2)
    model=NativePlatePrecision()
    for size in [128,192]:
        x=torch.rand(1,3,size,size);pred=model(x)
        assert pred['uv'].shape==(1,2,size,size)
        assert pred['corners'].shape==(1,4,2)
        loss=pred['uv'].square().mean()+pred['foreground'].square().mean()+pred['corners'].square().mean()/100
        loss.backward()
        assert torch.isfinite(loss)
        assert model.dense[-1].weight.grad.abs().sum()>0
        assert model.heatmap.weight.grad.abs().sum()>0
        model.zero_grad()
