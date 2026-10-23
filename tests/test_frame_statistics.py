"""Frame-normalization state and batching contracts."""
import copy
import unittest
import torch
from torch import nn
from armor_native.proposals import use_frame_statistics


class FrameStatistics(unittest.TestCase):
    def model(self):
        torch.manual_seed(53)
        return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1),
                             nn.BatchNorm2d(8, eps=.001),
                             nn.SiLU()).eval()

    def test_matches_batch_one_training_statistics_without_changing_parameters(self):
        reference = self.model()
        actual = copy.deepcopy(reference)
        reference[1].training = True
        reference[1].track_running_stats = False
        before = {key: value.clone() for key, value in actual.named_parameters()}
        self.assertEqual(use_frame_statistics(actual), 1)
        actual.eval()
        image = torch.randn(1, 3, 32, 48) * .03 + .01
        with torch.inference_mode():
            torch.testing.assert_close(actual(image), reference(image), rtol=0, atol=0)
        for key, value in actual.named_parameters():
            self.assertTrue(torch.equal(value, before[key]))

    def test_no_frame_history_or_batch_cross_talk(self):
        model = self.model()
        use_frame_statistics(model)
        model.eval()
        dark = torch.randn(1, 3, 32, 48) * .01
        bright = torch.randn(1, 3, 32, 48) * 5 + 10
        with torch.inference_mode():
            original = model(dark).clone()
            model(bright)
            self.assertTrue(torch.equal(model(dark), original))
            together = model(torch.cat([dark, bright]))[:1]
            torch.testing.assert_close(together, original, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
