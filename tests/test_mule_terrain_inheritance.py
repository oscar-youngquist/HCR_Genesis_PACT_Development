"""Live HardPACT terrain inheritance, including the actual initial sampler."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
from legged_gym.simulator.isaaclab_simulator_mule import IsaacLabSimulatorMULE
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT
from legged_gym.scripts.train_mule import parser, resolve_run
from rsl_rl.beyond_robustness_config import settings_dict
from legged_gym.utils.helpers import class_to_dict


def shared():
    E,T=make_hard_pact_backend_configs('isaaclab')
    return E(),T()


def test_full_terrain_is_inherited_and_owned():
    e,t=shared(); source=class_to_dict(e.terrain)
    c=resolve_go2_mule(e,t)
    assert settings_dict(c.shared.terrain)==source
    assert c.manifest['shared']['terrain']==source
    c.shared.terrain.terrain_proportions[0]=.99
    c.shared.terrain.measured_points_x[0]=99
    assert class_to_dict(e.terrain)==source
    assert settings_dict(resolve_go2_mule(e,t).shared.terrain)==source


@pytest.mark.parametrize('curriculum',[True,False])
def test_changed_initial_distribution_propagates_to_actual_sampler(curriculum):
    e,t=shared()
    e.terrain.num_rows=7; e.terrain.num_cols=12; e.terrain.num_subterrains=84
    e.terrain.max_init_terrain_level=3; e.terrain.curriculum=curriculum
    e.terrain.terrain_proportions=[.2,.1,.1,.2,.1,.3]
    e.terrain.terrain_length=6.; e.terrain.terrain_width=9.
    e.terrain.curriculum_upward_delay_iterations=123
    c=resolve_go2_mule(e,t)
    assert settings_dict(c.shared.terrain)==class_to_dict(e.terrain)
    assert IsaacLabSimulatorMULE._get_env_origins is IsaacLabSimulator_PACT._get_env_origins
    origins=np.arange(7*12*3).reshape(7,12,3).astype(float)
    def sample(cls,cfg):
        s=object.__new__(cls); s._cfg=cfg; s._device='cpu'; s._num_envs=4096
        s._terrain=NS(env_origins=origins)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7); s._get_env_origins()
        return s
    mule=sample(IsaacLabSimulatorMULE,c.shared)
    pact=sample(IsaacLabSimulator_PACT,e)
    for name in ('_terrain_levels','_terrain_types','_env_origins'):
        torch.testing.assert_close(getattr(mule,name),getattr(pact,name))
    upper=3 if curriculum else 6
    assert set(mule._terrain_levels.tolist())==set(range(upper+1))
    counts=torch.bincount(mule._terrain_levels,minlength=upper+1)
    assert ((counts.float()/4096-1/(upper+1)).abs()<.04).all()
    assert torch.bincount(mule._terrain_types).max()-torch.bincount(mule._terrain_types).min()<=1


@pytest.mark.parametrize('phase',['nominal_only','joint_adaptation'])
def test_production_entrypoint_preserves_terrain_in_both_phases(phase):
    e,_=shared()
    c,_,_=resolve_run(parser().parse_args(['--num-envs','64','--optimization-phase',phase]))
    assert settings_dict(c.shared.terrain)==class_to_dict(e.terrain)
    assert c.manifest['overrides']['environment'].get('terrain.max_init_terrain_level') is None
