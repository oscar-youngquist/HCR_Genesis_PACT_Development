"""RSS selectors, real PPO paths, gradients and shell dispatch (no simulator)."""
import os
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rsl_rl.hard_pact_ablations import RSS_HARD_PACT_ABLATIONS as RSS, resolve_hard_pact_features
from rsl_rl.algorithms.hard_pact_actor_regularizers import ActorRegularizerConfig, detached_torque_imitation
from rsl_rl.modules.actor_critic_hard_pact import ActorCritic_HardPACT
from rsl_rl.modules.hard_pact_action_std import initialize_action_std
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_rss_hard_pact_configs
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO
from legged_gym.envs.go2.go2_hard_pact.deployment import calculate_physics_head_gains, build_deployment_contract
from legged_gym.utils.task_registry import task_registry
from test_hard_pact_auxiliary import make_algorithm
from test_hard_pact_actor_regularizers import allocation_inputs
from test_hard_pact_reduced_qp import inputs, solver

ROOT = Path(__file__).resolve().parents[1]


def test_registry_matrix_and_isolation():
    expected = {
        'pos_ppo': ('position', False, False, False, 'off'),
        'coupled_ppo': ('coupled', False, False, False, 'off'),
        'guided_no_qp': ('coupled', True, True, False, 'off'),
        'qp_no_post': ('coupled', True, True, True, 'metric'),
        'qp_imitation': ('coupled', True, True, True, 'imitation'),
        'full': ('coupled', True, True, True, 'full'),
        'no_fk': ('coupled', False, True, True, 'full'),
        'no_ff': ('coupled', True, False, True, 'full'),
        'no_channels': ('coupled', False, False, True, 'full'),
        'no_qp_wrench': ('coupled', True, True, True, 'full'),
    }
    import legged_gym.envs
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
    for name, feature in RSS.items():
        assert (feature.action_mode, feature.fk, feature.ff_support, feature.execution_qp, feature.post_qp) == expected[name]
        assert not feature.inverse_loss and not feature.rollout_loss
        assert feature.qp_wrench == (name != 'no_qp_wrench')
        assert resolve_hard_pact_features(feature.variant_id) is feature
        E, P = make_rss_hard_pact_configs(name)
        e, p = E(), P()
        assert not p.algorithm.bard_inverse_enabled and not p.algorithm.bard_rollout_enabled
        assert e.env.num_observations == GO2HardPACTCfg.env.num_observations
        assert p.runner.max_iterations == GO2HardPACTCfgPPO.runner.max_iterations
        assert task_registry.task_classes[f'go2_hard_pact_rss_{name}_isaaclab'] is Go2HardPACT
        assert p.algorithm.actor_regularizers['lambda_fk'] == (GO2HardPACTCfgPPO.algorithm.actor_regularizers['lambda_fk'] if feature.fk else 0)
        assert p.algorithm.actor_regularizers['lambda_load'] == (GO2HardPACTCfgPPO.algorithm.actor_regularizers['lambda_load'] if feature.ff_support else 0)
        p.algorithm.hard_pact_qp['torque_tracking_weight'] = -99
        e.domain_rand.added_mass_range[0] = -99
        assert GO2HardPACTCfg.domain_rand.added_mass_range[0] != -99
        assert make_rss_hard_pact_configs(name)[1]().algorithm.hard_pact_qp['torque_tracking_weight'] != -99
    with pytest.raises(FrozenInstanceError): RSS['full'].inverse_loss = True
    assert resolve_hard_pact_features('full').inverse_loss  # legacy retained


def test_position_distribution_and_deployment():
    cfg = GO2HardPACTCfg()
    gains = calculate_physics_head_gains(cfg)
    actor = ActorCritic_HardPACT(actor_layers=[32,16], critic_layers=[32,16],
        cenet_in_dim=1140, cenet_enc_layers=[32,16], ablation_features='rss_pos_ppo',
        grf_scale_n=gains.grf_scale_n, wrench_scale=gains.wrench_scale_n_nm,
        wrench_qp_clip=gains.wrench_qp_clip_n_nm)
    obs, hist = torch.randn(3,57), torch.randn(3,1140)
    actions = actor.act(obs, hist)
    assert actions.shape == actor.std.expand(3,-1).shape == (3,12)
    torch.testing.assert_close(actor.get_actions_log_prob(actions), actor.distribution.log_prob(actions).sum(-1))
    torch.testing.assert_close(actor.entropy, actor.distribution.entropy().sum(-1))
    assert actor.act_inference(obs,hist).shape == (3,12)
    actor.get_actions_log_prob(actions).sum().backward()
    assert actor.act_pos_out.weight.grad.abs().sum() > 0
    assert actor.act_tau_out.weight.grad is None
    c = build_deployment_contract(cfg,actor,gains)
    assert c['policy_action_dimension'] == 12 and c['policy_action_mode'] == 'position'


def test_imitation_masks_and_gradients():
    nominal = torch.zeros(3,12,requires_grad=True)
    target = torch.ones(3,12,requires_grad=True)
    with torch.no_grad(): target[2] = float('nan')
    result = SimpleNamespace(tau_safe=target, differentiated_mask=torch.tensor([1,0,0]).bool(),
                             recovery_mask=torch.tensor([0,1,0]).bool())
    loss = detached_torque_imitation(result, nominal, torch.ones(12), torch.ones(3).bool(), .5)
    assert loss == 18
    loss.backward()
    assert target.grad is None and nominal.grad.isfinite().all() and nominal.grad[2].eq(0).all()
    torch.testing.assert_close(nominal.grad[:2,0], torch.tensor([-2.,-1.]))


def test_full_imitation_and_metric_derivative_routes():
    from rsl_rl.algorithms.hard_pact_qp import projection_loss
    data=inputs(2)
    data['foot_jacobians'][:,:,:,:3]=torch.eye(3,dtype=torch.float64)
    for i in range(4): data['foot_jacobians'][:,i,:,6+3*i:9+3*i]=torch.eye(3,dtype=torch.float64)
    refs=[data[k].requires_grad_() for k in ('tau_nom','force_pred_world','wrench_pred_world')]
    qp=solver()
    result=qp.solve(differentiable=True,**data)
    valid=torch.ones(2,dtype=torch.bool)
    assert result.differentiated_mask.all()
    full=projection_loss(result.tau_safe,refs[0],qp.torque_limits,valid,result.differentiated_mask,
        qdd=result.qdd,foot_jacobians=data['foot_jacobians'],foot_acceleration_bias=data['foot_acceleration_bias'],
        stance_mask=valid[:,None].expand(-1,4),contact_weight=1.,contact_scale=50.)
    gradients=torch.autograd.grad(full,refs,retain_graph=True)
    assert all(g.isfinite().all() and g.abs().sum()>0 for g in gradients)
    imitation=detached_torque_imitation(result,refs[0],qp.torque_limits,valid,.5)
    gradients=torch.autograd.grad(imitation,refs,allow_unused=True)
    assert gradients[0].abs().sum()>0 and gradients[1:] == (None,None)
    assert not full.detach().requires_grad  # metric-only contribution


@pytest.mark.parametrize('backend', ['qpth', pytest.param('cupiqp', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='cuPIQP CUDA unavailable'))])
def test_matched_qp_forward_and_wrench_omission(backend):
    data = inputs(2, device='cuda' if backend == 'cupiqp' else 'cpu')
    data['foot_jacobians'][:,:,:,:3] = torch.eye(3,dtype=torch.float64)
    data['wrench_pred_world'][:,2] = -10
    data['tau_nom'].requires_grad_()
    full = solver(qp_solver=backend).solve(differentiable=True, diagnostics_phase='ppo', **data)
    with torch.no_grad():
        metric = solver(qp_solver=backend).solve(differentiable=True, diagnostics_phase='ppo', **data)
    torch.testing.assert_close(full.tau_safe, metric.tau_safe, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(full.qdd, metric.qdd, atol=1e-10, rtol=1e-10)
    assert full.tau_safe.requires_grad and not metric.tau_safe.requires_grad
    assert full.differentiated_mask.all()
    full.tau_safe.square().sum().backward()
    assert data['tau_nom'].grad.isfinite().all() and data['tau_nom'].grad.abs().sum() > 0
    omitted = solver(qp_solver=backend,use_external_wrench=False).solve(differentiable=True, **data)
    zero = solver(qp_solver=backend).solve(differentiable=True, **dict(data,wrench_pred_world=torch.zeros_like(data['wrench_pred_world'])))
    torch.testing.assert_close(omitted.tau_safe,zero.tau_safe,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(omitted.qdd,zero.qdd,atol=1e-10,rtol=1e-10)
    assert data['wrench_pred_world'][:,2].eq(-10).all()


@pytest.mark.parametrize('variant', ['guided_no_qp','qp_no_post','qp_imitation','full','no_qp_wrench'])
def test_channel_losses_are_actor_only_and_do_not_require_qp(variant):
    alg = make_algorithm(ablation_variant='rss_'+variant, bard_batch_capacity=4)
    alg.actor_regularizers = ActorRegularizerConfig(lambda_fk=.1,lambda_load=.1)
    actor=alg.actor_critic
    _,_,latent,explicit=actor.cenet_enc_forward(torch.randn(3,1140))
    explicit=torch.cat((explicit[:,:3],torch.full_like(explicit[:,3:7],.9),explicit[:,7:]),-1)
    pos,ff=actor.actor_forward(torch.cat((torch.randn(3,57),latent,explicit),-1))
    q=torch.zeros(3,19);q[:,6]=1
    batch=dict(pre_q=q,pre_v=torch.zeros(3,18),physics_valid_mask=torch.ones(3,1,dtype=torch.bool),
        delayed_action_source_valid=torch.ones(3,1,dtype=torch.bool),
        control_kp=torch.ones(3,12)*3,control_kd=torch.ones(3,12)*.2,
        control_motor_strength=torch.ones(3,1)*1.2,control_feedback_weight=torch.ones(3,1)*.9,
        control_feedforward_weight=torch.ones(3,1)*.7,control_torque_limits=torch.ones(3,12)*23.5,
        actor_fk_stance=torch.ones(3,4,dtype=torch.bool),actor_fk_next_feet_body=torch.zeros(3,4,3))
    J,B,W,g,stance=allocation_inputs();J,B,g=J.float(),B.float(),g.float()
    alg.physics_dynamics=SimpleNamespace(batch_capacity=4,
        build_context=lambda *a,**kw:SimpleNamespace(foot_jacobians=J,base_jacobian=B),
        foot_positions_body=lambda command:command.reshape(-1,4,3))
    from rsl_rl.algorithms.hard_pact_actor_regularizers import compensation_demand
    seen=[]
    def demand(*a,**kw):
        seen.append(a[2].clone());assert not torch.is_grad_enabled()
        return compensation_demand(*a,**kw)
    raw_wrench=torch.zeros(3,6);raw_wrench[:,2]=-.2
    replay=dict(desired_position=pos,feedforward_torque=ff)
    with patch('rsl_rl.algorithms.ppo_hard_pact.nominal_support_gravity',return_value=g), \
         patch('rsl_rl.algorithms.ppo_hard_pact.compensation_demand',side_effect=demand), \
         patch.object(actor.physics_estimator,'predict_wrench',return_value=raw_wrench):
        load=alg._compute_rss_support_loss(replay,batch,(latent,explicit))
        fk=alg._compute_actor_fk_loss(replay,batch)
    assert load > 0 and fk > 0 and seen[0][:,2].lt(0).all()  # preserved even in no_qp_wrench
    ff_grads=torch.autograd.grad(load,[actor.act_tau_out.weight,actor.act_pos_out.weight,
        actor.context_encoder.ce_out_mean.weight,actor.physics_estimator.wrench_head[-1].weight],
        allow_unused=True,retain_graph=True)
    assert ff_grads[0].abs().sum()>0 and all(g is None for g in ff_grads[1:])
    fk_grads=torch.autograd.grad(fk,[actor.act_pos_out.weight,actor.act_tau_out.weight,
        actor.context_encoder.ce_out_mean.weight],allow_unused=True)
    assert fk_grads[0].abs().sum()>0 and all(g is None for g in fk_grads[1:])


@pytest.mark.parametrize('variant', tuple(RSS))
def test_tiny_real_ppo_update(variant):
    """Shared update/PCGrad/optimizer loop; synthetic mechanics, real CPU QP."""
    feat = RSS[variant]
    alg = make_algorithm(ablation_variant=feat.variant_id, num_learning_epochs=1,
                         num_mini_batches=1, bard_batch_capacity=4)
    actor = alg.actor_critic
    actor.hard_pact_features = feat
    if feat.action_mode == 'position':
        # Constructor/likelihood tested above; reconstruct owned std before the
        # test optimizer is rebuilt rather than silently use a 24-D draw.
        initialize_action_std(actor,12,1.,.2,5.)
        for group in alg.act_optimizer.optimizer.param_groups:
            group['params'] = [actor.std if p.shape == (24,) else p for p in group['params']]
        alg.ppo_parameters = [actor.std if p.shape == (24,) else p for p in alg.ppo_parameters]
    n, steps, width = 3, 2, actor.std.numel()
    alg.init_storage(n,steps,[57],[95],[133],[1140],[width],[11],[12],[18])
    s=alg.storage; s.max_action_delay=0
    z=lambda d: torch.zeros(steps,n,d)
    q=z(19);q[...,6]=1; q[...,7:]=torch.tensor([0.,.8,-1.5]*4)
    fields=dict(pre_q=q,pre_v=z(18),post_v=z(18),control_dt=z(1)+.02,
        interval_executed_torque=z(12),standardized_action_noise=z(width),
        delayed_action_source_valid=torch.ones(steps,n,1,dtype=torch.bool),
        physics_valid_mask=torch.ones(steps,n,1,dtype=torch.bool),
        control_kp=z(12)+3,control_kd=z(12)+.2,control_motor_strength=z(1)+1,
        control_feedback_weight=z(1)+1,control_feedforward_weight=z(1)+(width==24),
        control_torque_limits=z(12)+23.5,
        actor_fk_stance=torch.ones(steps,n,4,dtype=torch.bool),actor_fk_next_feet_body=torch.zeros(steps,n,4,3),
        total_external_wrench_label_yaw_normalized=z(6),sustained_wrench_active_mask=z(1).bool(),
        **{k:z(1).bool() for k in ('push_event_mask','reset_mask','timeout_mask','teleport_mask')})
    s.hard_pact_fields=fields
    J,B,W,g,stance=allocation_inputs()
    J,B,g=J.float(),B.float(),g.float()
    fake=SimpleNamespace(batch_capacity=4,
        build_context=lambda *a,**kw:SimpleNamespace(foot_jacobians=J,base_jacobian=B),
        foot_positions_body=lambda command: command.reshape(-1,4,3))
    alg.physics_dynamics=fake
    alg.actor_regularizers=ActorRegularizerConfig(lambda_fk=.1 if feat.fk else 0,lambda_load=.1 if feat.ff_support else 0)
    if feat.execution_qp:
        qp=solver(use_external_wrench=feat.qp_wrench)
        alg.hard_pact_qp=qp;alg.qp_config=qp.cfg
        fields.update(sampled_qp_q=q.clone(),sampled_qp_v=z(18),physics_dt=z(1)+.005,
            sampled_qp_velocity_command=z(3)+.2,sampled_qp_grf_conditioning_q=q[...,7:].clone(),
            sampled_qp_grf_conditioning_v=z(12),sampled_qp_previous_torque=z(12),
            sampled_qp_valid=torch.ones(steps,n,1,dtype=torch.bool),sampled_qp_substep_index=z(1).long())
        def mechanics(q,v,batch):
            d=inputs(q.shape[0],q.dtype)
            return SimpleNamespace(**{k:d[k] for k in ('mass_matrix','bias','foot_jacobians','base_jacobian','foot_acceleration_bias')})
        alg._deployment_replay_mechanics=mechanics
    for t in range(steps):
        obs,hist,crit=torch.randn(n,57),torch.randn(n,1140),torch.randn(n,95)
        with torch.no_grad(): alg.act(obs,crit,hist,obs,hist,obs,hist)
        tr=alg.transition
        for name,value in (('observations',obs),('observation_history',hist),('critic_observations',crit),
            ('actions',tr.actions),('actions_log_prob',tr.actions_log_prob[:,None]),('mu',tr.action_mean),
            ('sigma',tr.action_sigma),('latent_noise',tr.latent_noise),('latent_boot_mask',tr.latent_boot_mask)):
            getattr(s,name)[t].copy_(value)
    s.advantages.normal_()
    # Full batch = six rows; support builder is bounded to three-row fake J.
    alg.qp_config=replace(alg.qp_config,ppo_chunk_size=3)
    seen=[]
    real_loss=alg._compute_bard_loss
    def checked(*a,**kw):
        value=real_loss(*a,**kw); seen.append(value.requires_grad)
        if feat.post_qp == 'metric' and value.requires_grad:
            gradient=torch.autograd.grad(value,a[0],retain_graph=True,allow_unused=True)[0]
            assert gradient is None or gradient.eq(0).all()
        return value
    with patch.object(alg,'_compute_bard_loss',side_effect=checked), \
         patch('rsl_rl.algorithms.ppo_hard_pact.nominal_support_gravity',return_value=g), \
         patch('torch.cuda.synchronize'),patch.object(alg,'spectral_normalization'):
        alg.update(lambda a:(a[:,:12],a[:,12:]),lambda d,p,v:3*(d-p)-.2*v,
                   .02,1,torch.zeros(12),1.)
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    assert bool(seen) == feat.execution_qp
    if feat.post_qp in ('full','imitation'): assert all(seen)
    if feat.fk: assert alg.last_actor_loss_metrics['Loss/actor_pinn/fk'] > 0
    assert not alg.bard_inverse_enabled and not alg.bard_rollout_enabled


@pytest.mark.parametrize('variant', tuple(RSS))
def test_wrappers_dry_run(variant):
    env=dict(os.environ,HARD_PACT_PYTHON=os.sys.executable,CUDA_VISIBLE_DEVICES='GPU-assigned')
    for location in ('lab','unity'):
        script=ROOT/f'legged_gym/scripts/go2_hard_pact_rss_{variant}_{location}.sh'
        subprocess.run(['bash','-n',str(script)],check=True)
        output=subprocess.check_output(['bash',str(script),'--dry-run','--seed','7','--num_envs','3',
            '--max_iterations','2','--checkpoint','100'],env=env,text=True)
        assert f'go2_hard_pact_rss_{variant}_isaaclab' in output and '--seed 7' in output
        assert '--checkpoint 100' in output
        if not RSS[variant].execution_qp: assert 'solvers=\n' in output
        if location=='unity': assert 'skip_conda=1' in output and '--gpu cuda:0' in output


def test_container_allowlist_and_dry_run(tmp_path):
    env=dict(os.environ,IMAGE='/not-executed.sif',REPO=str(ROOT),RUN_DIR=str(tmp_path/'output'),
             CUDA_VISIBLE_DEVICES='GPU-scheduler-assigned',OMNI_KIT_ACCEPT_EULA='YES')
    run=ROOT/'containers/isaaclab/run.sh'
    for variant in RSS:
        out=subprocess.check_output(['bash',str(run),'script',f'go2_hard_pact_rss_{variant}_unity.sh',
                                     '--dry-run','--seed','8'],env=env,text=True)
        assert 'GPU-scheduler-assigned' in out and '--seed 8' in out
    assert not (tmp_path/'output').exists()  # no mounts, files, jobs or simulator
    bad=subprocess.run(['bash',str(run),'script','../../untrusted.sh','--dry-run'],env=env,capture_output=True)
    assert bad.returncode==2
