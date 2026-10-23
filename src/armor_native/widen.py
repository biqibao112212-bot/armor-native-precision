"""Function-preserving widening for the custom native four-crop network.

Based on the channel-duplication idea in Net2Net (Chen et al., arXiv:1511.05641).
GroupNorm, depthwise layers and the FiLM feature/anchor split are mapped explicitly.
This implementation supports only the controlled 1x -> 2x experiment.
"""
import torch


def widen_quad_state(source, target, seed=2026102221, symmetry_noise=.01):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    target_state = target.state_dict()

    def split_inputs(value, axis=1):
        assert axis == 1
        out = value.repeat_interleave(2, 1)/2
        if symmetry_noise:
            # Equal-and-opposite outgoing changes cancel for the duplicated
            # input features but allow distinct gradients after initialization.
            epsilon = torch.randn(out[:, 0::2].shape, generator=generator,
                                  dtype=out.dtype) * value.std() * symmetry_noise / 2
            out[:, 0::2] += epsilon
            out[:, 1::2] -= epsilon
        return out

    assert source.keys() == target_state.keys()
    for key, tensor in source.items():
        value = tensor.detach().cpu()
        shape = target_state[key].shape
        if value.shape == shape:
            widened = value.clone()
        elif key == "condition.0.weight":
            assert value.shape[1] == 4*64+8 and shape[1] == 4*128+8
            widened = torch.cat([split_inputs(value[:, :256]), value[:, 256:]], 1)
        elif key == "condition.2.weight":
            widened = value.repeat_interleave(2, 0)
        elif value.ndim == 1:
            widened = value.repeat_interleave(2)
        elif value.ndim == 4:
            assert shape[0] in [value.shape[0], value.shape[0]*2]
            widened = value.repeat_interleave(shape[0]//value.shape[0], 0)
            if shape[1] == value.shape[1]*2:
                widened = split_inputs(widened)
            else:
                assert shape[1] == value.shape[1]
        else:
            raise ValueError(f"Unmapped widening tensor: {key}, {value.shape} -> {shape}")
        assert widened.shape == shape, (key, widened.shape, shape)
        result[key] = widened
    return result
