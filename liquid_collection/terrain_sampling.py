"""Seeded terrain coverage across partial episode resets (CPU bookkeeping only)."""
import torch


class TerrainCoverage:
    def __init__(self, initial_tiles, num_tiles):
        self.previous = torch.as_tensor(initial_tiles, dtype=torch.long, device='cpu').clone()
        self.visits = torch.zeros(len(self.previous), num_tiles, dtype=torch.long)
        self.visits[torch.arange(len(self.previous)), self.previous] = 1

    def select(self, env_ids):
        """Prefer each robot's least-visited tiles, then globally least visited.

        Random ties use the worker's seeded Torch RNG. Never repeat the previous
        tile when a second tile exists; other environments remain untouched.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device='cpu')
        selected = []
        for env in ids.tolist():
            candidates = torch.arange(self.visits.shape[1])
            if len(candidates) > 1:
                candidates = candidates[candidates != self.previous[env]]
            local = self.visits[env, candidates]
            candidates = candidates[local == local.min()]
            totals = self.visits.sum(0)[candidates]
            candidates = candidates[totals == totals.min()]
            tile = int(candidates[torch.randint(len(candidates), ())])
            self.previous[env] = tile
            self.visits[env, tile] += 1
            selected.append(tile)
        return torch.tensor(selected, dtype=torch.long)


def relocate_reset(sim, env_ids, base_pos):
    """Move reset origins and preserve sampled spawn jitter, including terrain Z."""
    if not getattr(sim._cfg.liquid, 'resample_terrain_on_reset', False) or not sim._custom_origins or len(env_ids) == 0:
        return base_pos
    old = sim._env_origins[env_ids].clone()
    tiles = sim._terrain_coverage.select(env_ids.cpu()).to(sim._device)
    cols = sim._cfg.terrain.num_cols
    sim._terrain_levels[env_ids] = tiles // cols
    sim._terrain_types[env_ids] = tiles % cols
    sim._env_origins[env_ids] = sim._terrain_origins[tiles // cols, tiles % cols]
    return base_pos + sim._env_origins[env_ids] - old
