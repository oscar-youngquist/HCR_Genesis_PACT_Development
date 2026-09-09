"""Exercise the environment's actual label/critic assembly without Genesis."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from rsl_rl.algorithms.ppo_pact import PPO_PACT

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('task', ['pact', 'abl3', 'pact_pos'])
def test_payload_scaling_matches_labels_critic_and_reconstruction(task):
    directory=ROOT/f'legged_gym/envs/go1/go1_{task}'
    config=ast.parse((directory/f'go1_{task}_config.py').read_text())
    scales_class=next(n for n in ast.walk(config) if isinstance(n,ast.ClassDef) and n.name=='obs_scales')
    scales={n.targets[0].id:ast.literal_eval(n.value) for n in scales_class.body if isinstance(n,ast.Assign)}
    env=NS(num_envs=3, obs_buf=torch.zeros(3,57), obs_scales=NS(**scales),
           friction_value_offset=0., kp_scale_offset=0., kd_scale_offset=0.,
           cfg=NS(rewards=NS(foot_height_offset=0.,base_height_target=0.)))
    mass=torch.tensor([[-1.],[0.],[8.]])
    com=torch.tensor([[-.2,-.15,-.15],[0.,0.,0.],[.2,.15,.15]])
    sim=NS(_added_base_mass=mass.clone(),_base_com_bias=com.clone(),
           base_lin_vel=torch.zeros(3,3),base_pos=torch.zeros(3,3),
           feet_indices=torch.arange(4),link_contact_states=torch.zeros(3,4),
           feet_pos=torch.zeros(3,4,3),height_around_feet=torch.zeros(3,4,1),
           measured_heights=torch.zeros(3,143),normal_vector_around_feet=torch.zeros(3,4,3),
           _grfs_buf=torch.zeros(3,12))
    for name,width in (('_friction_values',1),('_rand_push_vels',3),('_rand_wrench_vels',3),
                       ('_kp_scale',12),('_kd_scale',12),('_motor_strength',12),
                       ('_joint_armature',1),('_joint_friction',1),('_joint_damping',1),
                       ('feedforward_tau_weight',1),('feedback_tau_weight',1)):
        setattr(sim,name,torch.ones(3,width))
    env.simulator=sim
    tree=ast.parse((directory/f'go1_{task}.py').read_text())
    body=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_observations').body
    # Execute the three original assignment expressions, including their original ordering.
    selected=[n for n in body if isinstance(n,ast.Assign) and
              ast.unparse(n.targets[0]) in ('self.explicit_labels_buf','domain_randomization_info','critic_obs')]
    assert len(selected)==3
    scope={'self':env,'torch':torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'observations','exec'),scope)
    critic=scope['critic_obs']
    expected=torch.cat((mass/8.,com/.2),-1)
    torch.testing.assert_close(env.explicit_labels_buf[:,12:16],expected)
    torch.testing.assert_close(critic[:,96:100],expected)
    algorithm=PPO_PACT.__new__(PPO_PACT);algorithm.privileged_grf_start_index=61
    target=algorithm._privileged_decode_target(critic,12)
    torch.testing.assert_close(target[:,84:88],expected)
    assert expected.abs().max()<=1
    # The 8 kg mass error now contributes 1 instead of 64 to squared error.
    assert target[-1,84].square()==1
    torch.testing.assert_close(sim._added_base_mass,mass)
    torch.testing.assert_close(sim._base_com_bias,com)
