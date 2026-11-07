from unittest.mock import Mock

import pytest
import torch

from rsl_rl.hard_pact_logging import log_hard_pact_policy_std


@pytest.mark.parametrize('coupled', [False, True])
def test_policy_std_logging_channels_every_iteration_without_mutation(coupled):
    std = torch.nn.Parameter(torch.tensor([.4]*12 + ([.8]*12 if coupled else [])))
    original = std.detach().clone()
    version = std._version
    writer = Mock()
    for iteration in (1, 2):  # No diagnostic cadence gate.
        log_hard_pact_policy_std(writer, std, iteration)
        calls = writer.add_scalar.call_args_list
        metrics = {c.args[0]: c.args[1] for c in calls if c.args[2] == iteration}
        assert metrics['Policy/mean_noise_std'] == pytest.approx(.6 if coupled else .4)
        assert metrics['Policy/mean_noise_std_position'] == pytest.approx(.4)
        if coupled:
            assert metrics['Policy/mean_noise_std_feedforward'] == pytest.approx(.8)
        assert len(metrics) == (3 if coupled else 2)
        assert all(isinstance(v, float) for v in metrics.values())
    assert writer.add_scalar.call_count == (6 if coupled else 4)
    assert std._version == version and std.grad is None
    torch.testing.assert_close(std, original, rtol=0, atol=0)
