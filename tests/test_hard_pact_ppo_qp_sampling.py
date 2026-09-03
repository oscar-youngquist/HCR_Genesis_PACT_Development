"""Focused coverage for deterministic PPO-side HardPACT QP partitioning."""

from types import SimpleNamespace

import torch

from rsl_rl.algorithms.ppo_hard_pact import (
    PPO_HardPACT,
    disjoint_qp_epoch_mask,
)
from rsl_rl.hard_pact_ablations import resolve_hard_pact_features


def test_disjoint_anchor_stratified_qp_sampling_and_zero_pinn_weight():
    rows = 20
    epochs = 5
    rollout_indices = torch.arange(rows)
    anchors = torch.tensor([0, 2] * (rows // 2))

    masks = [
        disjoint_qp_epoch_mask(
            rollout_indices,
            anchors,
            epoch=epoch,
            num_epochs=epochs,
            passes_per_iteration=1,
            shard_percentage=20.0,
            stratify_by_anchor=True,
            seed=37,
            iteration=11,
        )
        for epoch in range(epochs)
    ]
    coverage = torch.stack(masks).sum(dim=0)
    assert torch.equal(coverage, torch.ones_like(coverage))
    for mask in masks:
        assert int(mask.sum()) == 4
        assert int((anchors[mask] == 0).sum()) == 2
        assert int((anchors[mask] == 2).sum()) == 2
    # Stateless sampling is exactly reproducible at resume iteration 11.
    repeated = disjoint_qp_epoch_mask(
        rollout_indices,
        anchors,
        epoch=3,
        num_epochs=epochs,
        seed=37,
        iteration=11,
    )
    assert torch.equal(repeated, masks[3])

    class CountingQP:
        def __init__(self):
            self.calls = 0

        def solve(self, value):
            self.calls += 1
            return 1.1 * value

    algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
    algorithm.device = torch.device("cpu")
    algorithm.num_learning_epochs = epochs
    algorithm.ppo_qp_sampling = "disjoint_epoch_partition"
    algorithm.ppo_qp_passes_per_iteration = 1
    algorithm.ppo_qp_shard_percentage = 20.0
    algorithm.ppo_qp_stratify_by_anchor = True
    algorithm.ppo_qp_sampling_seed = 37
    algorithm.lambda_inverse = 0.7
    algorithm.lambda_rollout = 0.8
    algorithm.lambda_projection = 1.0
    algorithm.lambda_soft_constraint = 0.0
    algorithm.hard_pact_features = resolve_hard_pact_features("hard")
    solver = CountingQP()
    algorithm.hard_pact_qp = solver
    algorithm.storage = SimpleNamespace(
        current_batch_indices=rollout_indices,
        current_hard_pact_batch={
            "sampled_qp_substep_index": anchors[:, None],
        },
    )

    # One QP is triggered in every epoch/minibatch, each row participates
    # once, and lambda_proj remains active although w_PINN is exactly zero.
    learned = torch.linspace(-1.0, 1.0, rows, requires_grad=True)
    losses = []
    selected_rows = []
    for epoch in range(epochs):
        shard = algorithm._qp_rows_for_epoch(epoch, iteration=11)
        selected_rows.append(shard)
        projected = solver.solve(learned[shard])
        projection = (projected - learned[shard]).square().mean()
        losses.append(algorithm._combine_bard_losses(
            learned.sum() * 0.0,
            learned.sum() * 0.0,
            projection,
            pinn_weight=0.0,
        ))
    torch.stack(losses).mean().backward()
    assert solver.calls == epochs
    assert torch.equal(
        torch.cat(selected_rows).sort().values, rollout_indices
    )
    assert learned.grad is not None
    assert torch.isfinite(learned.grad).all()
    assert (learned.grad.abs() > 0).all()

    # A QP-disabled ablation returns no rows and therefore invokes no solver.
    disabled = CountingQP()
    algorithm.hard_pact_qp = disabled
    algorithm.hard_pact_features = resolve_hard_pact_features("soft")
    for epoch in range(epochs):
        shard = algorithm._qp_rows_for_epoch(epoch, iteration=11)
        if shard is not None:
            disabled.solve(learned[shard])
    assert disabled.calls == 0
