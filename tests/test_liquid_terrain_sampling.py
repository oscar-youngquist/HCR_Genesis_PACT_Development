from types import SimpleNamespace as NS
import torch
from liquid_collection.terrain_sampling import TerrainCoverage, relocate_reset


def test_each_robot_visits_every_tile_without_immediate_repeats():
    torch.manual_seed(7)
    sampling=TerrainCoverage([0,1,2],9)
    previous=sampling.previous.clone()
    for _ in range(8):
        current=sampling.select(torch.arange(3))
        assert (current!=previous).all()
        previous=current
    assert (sampling.visits==1).all()
    assert sampling.select([]).numel()==0


def test_partial_reset_preserves_other_env_and_moves_xyz_with_jitter():
    origins=torch.tensor([[[0.,0.,0.],[0.,8.,-.2]],[[8.,0.,.3],[8.,8.,.1]]])
    sim=NS(_cfg=NS(liquid=NS(resample_terrain_on_reset=True),terrain=NS(num_cols=2)),
        _custom_origins=True,_device='cpu',_terrain_levels=torch.tensor([0,1]),
        _terrain_types=torch.tensor([0,1]),_terrain_origins=origins,
        _env_origins=torch.stack([origins[0,0],origins[1,1]]),
        _terrain_coverage=TerrainCoverage([0,3],4))
    other=sim._env_origins[1].clone();visits=sim._terrain_coverage.visits[1].clone()
    jitter=torch.tensor([[.2,-.3,.42]])
    position=relocate_reset(sim,torch.tensor([0]),sim._env_origins[:1]+jitter)
    torch.testing.assert_close(position,sim._env_origins[:1]+jitter)
    torch.testing.assert_close(sim._env_origins[1],other)
    torch.testing.assert_close(sim._terrain_coverage.visits[1],visits)
    assert sim._terrain_coverage.previous[0]!=0
    sim._cfg.liquid.resample_terrain_on_reset=False
    assert relocate_reset(sim,torch.tensor([0]),position) is position


def test_seeded_ties_and_single_tile_fallback():
    torch.manual_seed(42);a=TerrainCoverage([0],5).select([0])
    torch.manual_seed(42);b=TerrainCoverage([0],5).select([0])
    torch.testing.assert_close(a,b)
    assert TerrainCoverage([0],1).select([0]).item()==0
