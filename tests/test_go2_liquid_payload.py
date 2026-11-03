"""Liquid task preserves Go2 contracts and resets only the requested fluid."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg, GO2PACTCfgPPO
from legged_gym.envs.go2.go2_pact_water_eval.go2_pact_water_config import GO2PACTWaterCfg, GO2PACTWaterCfgPPO
from legged_gym.simulator.genesis_simulator_pact_liquid import GenesisSimulator_PACT_Liquid, tank_geometry
from legged_gym.simulator.genesis_simulator_pact_nopinn import GenesisSimulator_PACT_NoPINN
from legged_gym.scripts.play_go2_pact_water import configure, parser


def test_task_preserves_checkpoint_and_observation_contract():
    from legged_gym.utils import class_to_dict, task_registry
    from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
    assert task_registry.get_task_class('go2_pact_water') is Go2PACT
    assert class_to_dict(GO2PACTWaterCfgPPO.policy)==class_to_dict(GO2PACTCfgPPO.policy)
    for key in ('num_observations','num_privileged_obs','num_explicit_recon_obs',
                'num_obs_hist','num_actions','num_priv_stack'):
        assert getattr(GO2PACTWaterCfg.env,key)==getattr(GO2PACTCfg.env,key)
    assert GO2PACTWaterCfg.asset.dof_names==GO2PACTCfg.asset.dof_names
    assert not hasattr(GO2PACTCfg.env,'liquid_payload')


def test_mount_fill_and_cli_are_configurable():
    cfg=GO2PACTWaterCfg()
    args=parser().parse_args(['--num_envs','3','--liquid_type','water','--liquid_volume','10',
                             '--liquid_tank','wide','--ckpt','7000'])
    configure(cfg,args)
    preset,scale,xy,size,bottom,top=tank_geometry(cfg.liquid)
    assert args.ckpt==7000 and cfg.env.num_envs==3
    assert cfg.liquid.liquid_type=='water' and cfg.liquid.liquid_volume==10
    np.testing.assert_allclose(scale,[1.4,2.6,1.2])
    assert (size>0).all() and top>bottom>cfg.liquid.mount_height>.055
    assert np.all(xy==0)
    cfg.liquid.liquid_volume=3
    with pytest.raises(ValueError,match='Unsupported'):
        cfg.liquid.liquid_tank='default'
        tank_geometry(cfg.liquid)


def test_liquid_partial_reset_rotates_and_clears_momentum():
    sim=GenesisSimulator_PACT_Liquid.__new__(GenesisSimulator_PACT_Liquid)
    sim._liquid_rest_local=torch.tensor([[[1.,0.,.3],[0.,1.,.3]],[[1.,0.,.3],[0.,1.,.3]]])
    world=sim._liquid_rest_local.clone();velocity=torch.ones_like(world)
    sim._liquid=SimpleNamespace(
        set_particles_pos=lambda p,envs_idx:world.index_copy_(0,envs_idx,p),
        set_particles_vel=lambda v,envs_idx:velocity.index_copy_(0,envs_idx,v))
    ids=torch.tensor([1]);pos=torch.tensor([[3.,4.,1.]])
    quat=torch.tensor([[0.,0.,2**-.5,2**-.5]])
    with patch.object(GenesisSimulator_PACT_NoPINN,'reset_root_states'):
        sim.reset_root_states(ids,pos,quat,torch.zeros(1,3),torch.zeros(1,3))
    torch.testing.assert_close(world[0],sim._liquid_rest_local[0])
    torch.testing.assert_close(world[1],torch.tensor([[3.,5.,1.3],[2.,4.,1.3]]))
    assert velocity[1].eq(0).all() and velocity[0].eq(1).all()
