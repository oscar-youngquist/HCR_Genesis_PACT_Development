"""Selection is pre-solver; two rollout solves still yield one PPO anchor."""
import torch
import pytest

from rsl_rl.algorithms.hard_pact_qp import (
    balanced_substep_indices, qp_interval_selection, qp_substep_mask,
)
from rsl_rl.algorithms.ppo_hard_pact import disjoint_qp_epoch_mask
from test_hard_pact_qp_modes import fixture


def test_balanced_later_solve_and_independent_fair_replay_coin():
    generator = torch.Generator().manual_seed(41)
    later, replay = qp_interval_selection("first_plus_random_substep", 30000, 4, "cpu", generator=generator)
    assert torch.bincount(later.long()).tolist() == [0, 10000, 10000, 10000]
    for k in (1, 2, 3):
        # A separate coin, not correlated with the chosen later phase.
        fraction = (replay[later == k] == 0).float().mean().item()
        assert abs(fraction-.5) < .025
    assert ((replay == 0) | (replay == later)).all()
    masks = torch.stack([qp_substep_mask("first_plus_random_substep", k, later) for k in range(4)])
    assert masks[0].all() and masks.sum(0).eq(2).all()
    assert masks.gather(0, replay.long()[None]).all()
    epochs = [disjoint_qp_epoch_mask(torch.arange(len(replay)), replay, epoch=e, num_epochs=5)
              for e in range(5)]
    assert torch.stack(epochs).sum(0).eq(1).all()
    again = qp_interval_selection("first_plus_random_substep", 30000, 4, "cpu",
                                  generator=torch.Generator().manual_seed(41))
    assert all(torch.equal(a,b) for a,b in zip((later,replay),again))


@pytest.mark.parametrize("n", [0, 1, 2, 7, 11])
def test_small_batches_are_balanced(n):
    later, replay = qp_interval_selection("first_plus_random_substep", n, 4, "cpu")
    counts = torch.bincount(later.long(), minlength=4)[1:]
    assert counts.max()-counts.min() <= 1
    assert replay.shape == later.shape == (n,)


@pytest.mark.parametrize("mode", ["every_substep", "random_one_substep"])
def test_old_selection_and_rng_consumption_are_unchanged(mode):
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    old = balanced_substep_indices(17, 4, "cpu", generator=g1)
    rollout, replay = qp_interval_selection(mode, 17, 4, "cpu", generator=g2)
    assert torch.equal(old, rollout) and torch.equal(old, replay)
    assert torch.equal(g1.get_state(), g2.get_state())


def test_failed_first_solve_does_not_reselect_replay_anchor():
    torch.manual_seed(12)
    task, _, qp, real, counts, q, v, quat, _ = fixture("first_plus_random_substep", n=12)
    task._begin_qp_interval()
    anchor = task._qp_sampled_substep_index.clone()
    assert (anchor == 0).any() and (anchor > 0).any()
    original = qp.solve
    def fail_first(**kwargs):
        result = original(**kwargs)
        if task._qp_substep == 0:
            result.stage.fill_(2)
            result.differentiated_mask.zero_()
            result.recovery_mask.zero_()
        return result
    qp.solve = fail_first
    for k in range(4):
        q.fill_(.001*k)
        task._solve_hard_pact_rollout_qp_substep(quat, torch.zeros(12,6))
    packet = task._qp_sampled_transition
    assert counts.eq(2).all() and packet["sampled_qp_valid"].all()
    assert torch.equal(packet["sampled_qp_substep_index"].flatten(), anchor)
    assert packet["sampled_qp_stage"][anchor == 0].eq(2).all()
    assert not packet["sampled_qp_differentiated"][anchor == 0].any()
    torch.testing.assert_close(packet["sampled_qp_q"][:,7:], (.001*anchor.float())[:,None].expand(-1,12))
