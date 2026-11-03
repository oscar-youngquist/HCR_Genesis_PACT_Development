"""Checkpoint-shaped policies and saved semantics for the shared liquid CLI."""
from pathlib import Path
import pytest
import torch
from legged_gym.envs.go1.go1_pact_water_eval.go1_pact_water_config import GO1PACTWaterCfg, GO1PACTWaterCfgPPO
from legged_gym.envs.go1.go1_pact.go1_pact import Go1PACT
from legged_gym.scripts.play_pact_water import parser, load_policy, configure, PLAY_RESET_TERMINATION
from legged_gym.scripts.pact_liquid_checkpoint import infer_policy_contract
from rsl_rl.modules import ActorCritic_PACT


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'logs/pact_corl/go1_pact_rough/May17_21-20-40_pact_100hz_spec_smartcurr_original_jointrand_hightau'


def test_go1_registered_shared_task_and_default_run():
    from legged_gym.utils import task_registry
    assert task_registry.get_task_class('go1_pact_water') is Go1PACT
    cfg,train=GO1PACTWaterCfg(),GO1PACTWaterCfgPPO()
    assert cfg.env.liquid_payload and cfg.env.num_explicit_recon_obs==16
    assert train.runner.load_run==RUN.name
    assert cfg.liquid.mount_height==.057
    assert not cfg.domain_rand.push_robots and not cfg.commands.curriculum
    assert parser().parse_args(['--robot','go1','--liquid_volume','6']).liquid_volume==6


@pytest.mark.skipif(not (RUN/'model_8000.pt').is_file(),reason='Local May17 checkpoint unavailable')
def test_requested_go1_checkpoint_strict_load_and_saved_configuration():
    cfg,train=GO1PACTWaterCfg(),GO1PACTWaterCfgPPO()
    # Start with deliberately wrong architecture and gain to prove restoration.
    train.policy.cenet_velo_dim=11
    cfg.control.action_scale=99.
    cfg.init_state.default_joint_angles={name:99. for name in cfg.asset.dof_names}
    args=parser().parse_args(['--robot','go1','--checkpoint',str(RUN/'model_8000.pt')])
    model,path=load_policy(cfg,train,args,'cpu')
    assert path==RUN/'model_8000.pt'
    assert train.policy.cenet_velo_dim==16 and cfg.env.num_obs_hist==20
    assert cfg.control.action_scale==.25 and train.policy.activation=='elu'
    assert cfg.normalization.obs_scales.dof_tau==.05
    assert cfg.init_state.default_joint_angles['FR_hip_joint']==-.1
    assert cfg.init_state.default_joint_angles['FL_thigh_joint']==.8
    configure(cfg,args)
    assert cfg.liquid.joint_dynamics['armature']['midpoint']==.015
    assert cfg.domain_rand.joint_friction_range_start==[.03,.03]
    assert cfg.domain_rand.joint_stiffness_range_end==[.01,.01]
    assert cfg.domain_rand.joint_damping_range_end==[.4,.4]
    with torch.no_grad():
        out=model.act_inference(torch.zeros(2,57),torch.zeros(2,1140))
    assert out.shape==(2,24) and out.isfinite().all()
    assert not any(p.requires_grad for p in model.parameters())


def test_alternative_checkpoint_shapes_and_literal_config_do_not_execute_python(tmp_path):
    reference=ActorCritic_PACT(57,17,12,[8,6],[8,4],57*3,5,11,[8,6],'tanh',1.).eval()
    checkpoint=tmp_path/'model_1.pt'
    torch.save({'model_state_dict':reference.state_dict()},checkpoint)
    (tmp_path/'go1_pact_config.py').write_text('''
raise RuntimeError("Saved source must never execute")
class GO1PACTCfg:
    class asset:
        name = 'go1'
    class control:
        dt = 0.020
        decimation = 4
        action_scale = 0.30
    class normalization:
        class obs_scales:
            ang_vel = 0.125
class GO1PACTCfgPPO:
    class policy:
        activation = 'tanh'
''')
    cfg,train=GO1PACTWaterCfg(),GO1PACTWaterCfgPPO()
    model,_=load_policy(cfg,train,parser().parse_args(['--checkpoint',str(checkpoint)]),'cpu')
    assert cfg.env.num_obs_hist==3 and cfg.control.dt==.020 and cfg.sim.dt==.005
    assert cfg.control.action_scale==.30 and cfg.normalization.obs_scales.ang_vel==.125
    assert train.policy.cenet_enc_latent_dim==5 and train.policy.cenet_velo_dim==11
    obs,hist=torch.randn(2,57),torch.randn(2,171)
    with torch.no_grad():
        torch.testing.assert_close(model.act_inference(obs,hist),reference.act_inference(obs,hist),rtol=0,atol=0)
    bad=reference.state_dict();bad.pop('act_tau_out.weight')
    with pytest.raises(ValueError,match='Unsupported'):
        infer_policy_contract(bad)


def test_saved_joint_midpoints_and_manual_reset_precedence(tmp_path,monkeypatch):
    from legged_gym.scripts.pact_liquid_checkpoint import restore_inference_config
    source=tmp_path/'go1_pact_config.py'
    source.write_text('''
class GO1PACTCfg:
    class domain_rand:
        randomize_joint_armature = True
        joint_armature_range = [0.02,0.04]
        joint_friction_range_start = [0.0,0.1]
        joint_friction_range_end = [0.2,0.4]
        joint_stiffness_range = [0.04,0.06]
        joint_damping_range_end = [0.6,1.0]
    class termination:
        roll_threshold = 0.9
    class env:
        episode_length_s = 12.0
''')
    cfg,train=GO1PACTWaterCfg(),GO1PACTWaterCfgPPO()
    restore_inference_config(cfg,train,tmp_path/'model.pt','go1')
    args=parser().parse_args(['--robot','go1'])
    monkeypatch.setitem(PLAY_RESET_TERMINATION['termination'],'pitch_threshold',.75)
    configure(cfg,args)
    assert cfg.domain_rand.joint_armature_range==[.03,.03]
    assert cfg.domain_rand.joint_friction_range_start==pytest.approx([.3,.3])
    assert cfg.domain_rand.joint_stiffness_range_start==[.05,.05]
    assert cfg.domain_rand.joint_damping_range_end==[.8,.8]
    assert cfg.termination.roll_threshold==.9 and cfg.env.episode_length_s==12.
    assert cfg.termination.pitch_threshold==.75
    assert not cfg.domain_rand.use_domainrand_curriculum
