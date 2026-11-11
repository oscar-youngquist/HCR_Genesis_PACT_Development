from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from rsl_rl.hard_pact_ablations import resolve_hard_pact_features
from rsl_rl.hard_pact_logging import compact_hard_pact_metrics
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT


def runner(hard=True, active=False):
    obj = OnPolicyRunnerPACT.__new__(OnPolicyRunnerPACT)
    obj.is_hard_pact = hard
    obj.writer = Mock()
    obj.device = 'cpu'
    obj.tot_timesteps = 0
    obj.tot_time = 0.
    obj.num_steps_per_env = 2
    obj.current_learning_iteration = 0
    for key in ('console_reward_terms', 'console_detailed_losses', 'console_pinn_timing',
                'console_qp_timing', 'console_iteration'):
        setattr(obj, key, False)
    zero = torch.tensor(0.)
    obj.env = NS(num_envs=2, reward_scales={'tracking_lin_vel': 1.})
    obj.alg = NS(actor_critic=NS(std=torch.ones(24)), learning_rate=.001,
        hard_pact_qp=None, qp_enabled_at_iteration=lambda i: active,
        hard_pact_features=resolve_hard_pact_features('full'),
        pinn_weight=float(active), bard_inverse_enabled=True, bard_rollout_enabled=True,
        actor_physics_warmup_factor=float(active),
        actor_regularizers=NS(lambda_fk=.1, lambda_load=.05),
        lambda_projection=.25, lambda_qp_velocity_xy=1., lambda_qp_velocity_yaw=0.,
        qp_config=NS(soft_joint_recovery_enabled=True), current_vae_beta=.1,
        last_auxiliary_metrics={'grf': zero, 'encoder_pinn_unweighted': zero},
        last_physics_loss_metrics={'physics/loss/inverse': zero, 'physics/loss/pinn_unweighted': zero},
        last_inverse_dynamics_metrics={'inverse_residual/all_mae_physical': zero},
        last_rollout_dynamics_metrics={'rollout_velocity/all_mae_physical': zero},
        last_actor_loss_metrics={f'Loss/actor_pinn/{k}': zero for k in (
            'projection_primary','projection_recovery','velocity_xy','velocity_yaw',
            'fk','load','total_weighted','total_warmed')},
        last_qp_metrics={}, last_physics_gradient_metrics={})
    if hard:
        obj.hard_pact_features = obj.alg.hard_pact_features
    return obj


def emit(obj):
    locs = dict(it=1, num_learning_iterations=2, collection_time=.1, learn_time=.2,
                ep_infos=[{'rew_tracking_lin_vel': torch.tensor(.6), 'success': torch.tensor(1.)}],
                rewbuffer=[1.], lenbuffer=[2.])
    for name in ('autoenc','vel','recon','kld','decoder','value','surrogate','pinn'):
        locs[f'mean_{name}_loss'] = 0.
    obj.log(locs)
    return [call.args[0] for call in obj.writer.add_scalar.call_args_list]


@pytest.mark.parametrize('active', [False, True])
def test_compact_runner_omits_duplicates_and_inactive_losses(active):
    obj = runner(active=active)
    keys = emit(obj)
    assert len(keys) == len(set(keys))
    assert not any(k.startswith(('reward/','rewards/','tracking/','system/','domain_rand/')) for k in keys)
    assert {'train/return', 'train/episode_length', 'train/success',
            'Train/mean_reward/time', 'Train/mean_episode_length/time'} <= set(keys)
    assert 'Episode/rew_tracking_lin_vel' in keys
    assert 'Episode/success' in keys
    assert 'Loss/auxiliary_grf' in keys
    assert 'physics/grf/loss' not in keys
    assert ('Loss/pinn_loss' in keys) == active
    assert ('Loss/actor_pinn/projection_primary' in keys) == active
    assert ('Loss/actor_pinn/fk' in keys) == active
    assert ('Loss/auxiliary_encoder_pinn_unweighted' in keys) == active
    assert ('inverse_residual/all_mae_physical' in keys) == active
    assert 'Loss/actor_pinn/velocity_yaw' not in keys  # coefficient zero
    assert len(keys) < 45


def test_legacy_logging_untouched():
    keys = emit(runner(hard=False))
    assert 'Episode/rew_tracking_lin_vel' in keys
    assert 'tracking/rew_tracking_lin_vel' in keys
    assert 'reward/tracking_lin_vel' in keys
    assert 'train/return' in keys
    assert 'Train/mean_reward/time' in keys


@pytest.mark.parametrize('active', [False, True])
def test_physical_actor_errors_follow_parent_loss_activation(active):
    obj = runner(active=active)
    for name in ('velocity_xy/error_m_s', 'velocity_yaw/error_rad_s', 'load/error_nm'):
        obj.alg.last_actor_loss_metrics['Loss/actor_pinn/'+name] = torch.tensor(1.)
    keys = emit(obj)
    assert ('Loss/actor_pinn/velocity_xy/error_m_s' in keys) == active
    assert ('Loss/actor_pinn/load/error_nm' in keys) == active
    assert 'Loss/actor_pinn/velocity_yaw/error_rad_s' not in keys
    assert len(keys) == len(set(keys))


def test_compact_metrics_do_not_create_unavailable_schema_entries():
    alg = runner().alg
    assert compact_hard_pact_metrics(alg, 0) == {'physics/actor_loss_warmup_factor': 0.}


def test_provenance_saved_to_file_not_tensorboard(tmp_path):
    import json
    obj = runner()
    obj.cfg = {'task_backend': 'isaaclab'}
    obj.train_cfg = {'seed': 7}
    obj.log_dir = str(tmp_path)
    obj.alg.actor_critic.parameters = lambda: iter([obj.alg.actor_critic.std])
    obj._log_hard_pact_run_metadata()
    metadata = json.loads((tmp_path/'hard_pact_run_metadata.json').read_text())
    assert metadata['seed'] == '7' and metadata['backend'] == 'isaaclab'
    assert metadata['features']['execution_qp'] == 1.
    assert metadata['config'] == {'seed': 7}
    obj.writer.add_text.assert_not_called()
