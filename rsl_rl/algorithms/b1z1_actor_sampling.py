"""Coverage-balanced actor-only sampling; independent RNG leaves PPO shuffles intact."""
import math

import torch


def validate_fraction(cfg):
    fraction = float(cfg.get("actor_phys_sample_fraction", 1.0))
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("actor_phys_sample_fraction must be finite and in [0, 1]")
    return fraction


def epoch_indices(active, fraction, epochs, seed):
    """Randomized cyclic windows of round(f*N); repetition counts differ by at most one.

    Evenly interleave shuffled force strata before taking windows, keeping each
    window's active count within one sample of its proportional allocation.
    """
    active = active.detach().flatten().cpu().bool()
    n = active.numel()
    size = round(fraction * n)
    once = math.isclose(fraction * epochs, 1., rel_tol=0., abs_tol=1e-12)
    if not n or (not size and not once):
        return [torch.empty(0, dtype=torch.long) for _ in range(epochs)]
    if size == n:
        return [torch.arange(n) for _ in range(epochs)]
    rng = torch.Generator().manual_seed(seed)
    yes, no = active.nonzero().flatten(), (~active).nonzero().flatten()
    yes = yes[torch.randperm(len(yes), generator=rng)]
    no = no[torch.randperm(len(no), generator=rng)]
    positions = torch.arange(n)
    slots = ((positions + 1) * len(yes) // n) != (positions * len(yes) // n)
    order = torch.empty(n, dtype=torch.long)
    order[slots], order[~slots] = yes, no
    offset = int(torch.randint(n, (), generator=rng))
    # At f=1/E, rounding every size independently cannot cover an indivisible N.
    # Rounded cumulative boundaries give exact coverage with sizes differing by one.
    bounds = ([round(epoch * n / epochs) for epoch in range(epochs + 1)] if once
              else [epoch * size for epoch in range(epochs + 1)])
    return [order[(torch.arange(bounds[e], bounds[e + 1]) + offset) % n] for e in range(epochs)]


def start(a, iteration):
    """Build only small index metadata here; mechanics are allocated per epoch."""
    a.actor_physics_selection = None
    a.actor_physics_index_map = None
    a.actor_sampling_metrics = {}
    fraction = validate_fraction(a.cfg)
    if not a.cfg.get("actor_phys_enabled", False):
        return
    n = a.storage.steps * a.storage.num_envs
    from .b1z1_actor_physics import scheduled_coefficient
    running = scheduled_coefficient(a) > 0
    if running and fraction < 1 and n % a.mini_batches:
        raise ValueError("Actor-PINN subsampling requires rollout size divisible by num_mini_batches; PPO drops remainder rows")
    active = a.storage.actor_physics.get("sampling_active_force")
    # Synthetic/older snapshots may omit the diagnostic stratum; use one stratum.
    active = torch.zeros(n, dtype=torch.bool) if active is None else active.flatten().cpu().bool()
    if running and fraction == 1:
        # No masks/permutations needed for the prepare-once legacy path.
        counts = torch.full((n,), a.epochs, dtype=torch.long)
        selections = None
    elif running:
        selections = epoch_indices(active, fraction, a.epochs,
                                   (torch.initial_seed() + int(iteration)) % (2**63 - 1))
    else:
        selections = [torch.empty(0, dtype=torch.long) for _ in range(a.epochs)]
    if selections is not None:
        counts = torch.zeros(n, dtype=torch.long)
        for indices in selections:
            counts.index_add_(0, indices, torch.ones_like(indices))
    total = n * a.epochs
    selected = int(counts.sum())
    a.actor_sampling_metrics = dict(
        sample_fraction=fraction, sample_fraction_realized=selected / max(total, 1),
        selected_transitions=selected, total_transitions=total,
        unique_transitions=int((counts > 0).sum()), unique_coverage=float((counts > 0).float().mean()) if n else 0.,
        mean_repetitions=float(counts.float().mean()) if n else 0.,
        max_repetitions=int(counts.max()) if n else 0,
        selected_active_force_count=int(counts[active].sum()))
    if running and 0 < fraction < 1:
        a.actor_physics_selection = selections


def select(a, batch, actions, context):
    """Subset before any actor physics; preserve a differentiable action view."""
    mapping = getattr(a, "actor_physics_index_map", None)
    if mapping is None:
        return batch, actions, context
    rows = mapping[batch["indices"]] >= 0
    return ({k: v[rows] for k, v in batch.items()}, actions[rows],
            {k: v[rows] for k, v in context.items()})
