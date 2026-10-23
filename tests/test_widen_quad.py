"""Function preservation and learnable extra capacity after native widening."""
import unittest
import torch
from armor_native.quad import NativeQuadPrecision
from armor_native.widen import widen_quad_state


class WideningContracts(unittest.TestCase):
    def test_initial_outputs_and_new_gradient_directions(self):
        torch.set_num_threads(2)
        torch.manual_seed(427)
        small = NativeQuadPrecision(True, "legacy", 1).eval()
        # Nonzero FiLM exercises the feature-versus-anchor mapping.
        with torch.no_grad():
            small.condition[-1].weight.normal_(0, .01)
            small.condition[-1].bias.normal_(0, .01)
        large = NativeQuadPrecision(True, "legacy", 2).eval()
        large.load_state_dict(widen_quad_state(small.state_dict(), large))
        image = torch.rand(1, 4, 3, 128, 128)
        anchors = torch.tensor([[[700., 560.], [700., 520.],
                                 [740., 520.], [740., 560.]]])
        with torch.no_grad():
            expected = small(image, anchors)
            actual = large(image, anchors)
        for key in ["heatmap", "offset", "corners"]:
            torch.testing.assert_close(actual[key], expected[key], atol=5e-4, rtol=1e-5)
        gradient = torch.autograd.grad(
            large(image, anchors)["corners"].sum(),
            large.backbone.native[0][0].weight)[0]
        self.assertGreater(float((gradient[0::2]-gradient[1::2]).abs().max()), 1e-7)


if __name__ == "__main__":
    unittest.main()
