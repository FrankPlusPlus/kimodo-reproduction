from __future__ import annotations

import unittest

import torch

from kimodo.training.optim import AdamAtan2


def _step(optimizer: AdamAtan2, parameter: torch.nn.Parameter, grad: float) -> None:
    parameter.grad = torch.full_like(parameter, grad)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


class AdamAtan2UpdateStatsTests(unittest.TestCase):
    def test_update_math_matches_with_or_without_stat_tracking(self) -> None:
        tracked = torch.nn.Parameter(torch.tensor([1.0]))
        skipped = torch.nn.Parameter(torch.tensor([1.0]))
        tracked_opt = AdamAtan2([tracked], lr=0.1, betas=(0.0, 0.0), weight_decay=0.3)
        skipped_opt = AdamAtan2([skipped], lr=0.1, betas=(0.0, 0.0), weight_decay=0.3)
        tracked_opt.track_update_stats = True
        skipped_opt.track_update_stats = False
        _step(tracked_opt, tracked, 2.0)
        _step(skipped_opt, skipped, 2.0)
        self.assertTrue(torch.equal(tracked.detach(), skipped.detach()))
        self.assertGreater(tracked_opt.last_update_norm, 0.0)
        self.assertEqual(skipped_opt.last_update_norm, 0.0)

    def test_stat_reduction_stays_on_parameter_device(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
        optimizer = AdamAtan2([parameter], lr=0.05, betas=(0.0, 0.0))
        optimizer.track_update_stats = True
        _step(optimizer, parameter, 1.25)
        self.assertGreater(optimizer.last_update_norm, 0.0)
        self.assertGreater(optimizer.last_update_param_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
