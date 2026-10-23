"""Continuous-coordinate numerical controls, independent of trained weights."""
import unittest
import torch
from armor_native.quad import gaussian_coordinate_points, gaussian_coordinate_loss


class ContinuousCoordinates(unittest.TestCase):
    def test_subpixel_targets_without_integer_rounding(self):
        phase = torch.tensor([.01, .1, .25, .49, .5, .75, .9, .99])
        points = torch.stack([40 + phase, 80 + phase.flip(0)], -1)[None]
        yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
        logits = -((xx - points[..., 0, None, None])**2 +
                   (yy - points[..., 1, None, None])**2) / (2 * 1.25**2)
        actual = gaussian_coordinate_points(logits)
        self.assertLess(float((actual - points).abs().max()), .006)
        self.assertLess(abs(float(gaussian_coordinate_loss(logits, points))), 1e-6)

    def test_distribution_gradient_moves_towards_exact_subpixel_label(self):
        center = torch.tensor([[[30.8, 31.7]]], requires_grad=True)
        target = torch.tensor([[[30.2, 31.25]]])
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        def logits(point):
            return -((xx-point[..., 0, None, None])**2 +
                     (yy-point[..., 1, None, None])**2)/(2*1.25**2)
        before = (gaussian_coordinate_points(logits(center))-target).norm()
        loss = gaussian_coordinate_loss(logits(center), target)
        loss.backward()
        self.assertTrue(torch.isfinite(center.grad).all())
        after_center = center.detach() - .2 * center.grad
        after = (gaussian_coordinate_points(logits(after_center))-target).norm()
        self.assertLess(float(after), float(before))


if __name__ == "__main__":
    unittest.main()
