import torch
from armor_native.quad import continuous_points
from armor_native.dense_vote import dense_vote_loss

def test_consistent_vectors_decode_fractional_points_even_at_borders():
    torch.manual_seed(1)
    corners=torch.tensor([[[.2,.7],[15.8,.1],[4.99,5.01],[9.01,8.99]]])
    y,x=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    field=(corners[:,:,:,None,None]-torch.stack([x,y]).float()[None,None]).reshape(1,8,16,16)
    heatmap=torch.randn(1,4,16,16)
    decoded=continuous_points(heatmap,field,offsets_are_vectors=True)
    assert torch.allclose(decoded,corners,atol=3e-6,rtol=0)
    assert dense_vote_loss(field,corners)<1e-10
    assert field.min()<0 and field.max()>1

def test_dense_labels_train_neighbors_not_only_the_floor_pixel():
    corners=torch.tensor([[[4.2,5.4],[7.6,7.3],[9.5,9.9],[10.1,4.2]]])
    field=torch.zeros(1,8,16,16,requires_grad=True)
    loss=dense_vote_loss(field,corners);loss.backward()
    assert torch.isfinite(field.grad).all()
    assert (field.grad!=0).sum()>4*2
    assert field.grad.abs().sum()>0

def test_legacy_decoder_matches_zero_residual_parameterization():
    torch.manual_seed(2);heatmap=torch.randn(2,4,16,16);offset=torch.randn(2,8,16,16)
    old=continuous_points(heatmap,offset)
    zero=continuous_points(heatmap,offset.sigmoid()+torch.zeros_like(offset),offsets_are_vectors=True)
    assert torch.equal(old,zero)
