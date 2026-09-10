"""PINN config/CLI precedence through the real parser and runner registry."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import legged_gym.envs  # Register tasks first, matching train.py import order.
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry, runner_registry
from test_hard_pact_auxiliary import make_algorithm


class CapturedRunner:
    def __init__(self, env, train_cfg, log_dir, device):
        self.train_cfg = train_cfg


def resolved_runner(task, options=(), configured_weight=-1.0):
    with patch('sys.argv', ['train.py', '--task', task, '--cpu', *options]):
        args = get_args()
    config = SimpleNamespace(
        runner_class_name='PACTRunner',
        runner=SimpleNamespace(resume=False),
        policy=SimpleNamespace(pinn_loss_weight=configured_weight, pinn_init_steps=0),
        algorithm=SimpleNamespace(**(
            {'hard_pact_qp': {}} if task.startswith('go2_hard_pact') else {}
        )),
    )
    with patch.object(runner_registry, 'get_runner_class', return_value=CapturedRunner):
        runner, _ = TaskRegistry().make_alg_runner(
            object(), name=task, args=args, train_cfg=config, log_root=None,
        )
    return args, runner.train_cfg['policy']


@pytest.mark.parametrize('task', [
    'go2_hard_pact_full_isaaclab', 'go2_hard_pact_genesis',
    'go2_hard_pact_soft_isaaclab', 'go2_hard_pact_inverse_isaaclab',
])
@pytest.mark.parametrize('configured', [-1.0, 0.125])
def test_omitted_cli_preserves_hard_pact_config(task, configured):
    args, policy = resolved_runner(task, configured_weight=configured)
    assert args.pinn_loss_weight is None
    assert policy['pinn_loss_weight'] == configured


@pytest.mark.parametrize('options,expected', [
    (['--pinn_loss_weight', '-1.0'], -1.0),
    (['--pinn_loss_weight=0.01'], 0.01),
    (['--pinn_loss_weight', '0'], 0.0),
])
def test_explicit_cli_override_reaches_runner(options, expected):
    args, policy = resolved_runner('go2_hard_pact_full_isaaclab', options)
    assert args.pinn_loss_weight == policy['pinn_loss_weight'] == expected


def test_preserved_negative_weight_reaches_algorithm_and_balanced_mode():
    _, policy = resolved_runner('go2_hard_pact_full_isaaclab')
    alg = make_algorithm(pinn_lambda=policy['pinn_loss_weight'], pinn_init_steps=0)
    assert alg.pinn_weight_final < 0  # update() selects pc_backward_ppgrad.
    assert alg._update_pinn_weight_for_iteration(1) == 1.0


@pytest.mark.parametrize('task', ['go1_pact', 'go2_pact'])
def test_legacy_default_and_explicit_override_are_unchanged(task):
    args, policy = resolved_runner(task)
    assert args.pinn_loss_weight == policy['pinn_loss_weight'] == 0.01
    _, policy = resolved_runner(task, ['--pinn_loss_weight=-1.0'])
    assert policy['pinn_loss_weight'] == -1.0


@pytest.mark.parametrize('task', ['go1_pact_pos', 'go2_pact_pos', 'go2_hard_pact_pos_isaaclab'])
def test_pos_retains_existing_config_only_semantics(task):
    _, policy = resolved_runner(task, ['--pinn_loss_weight=0.2'])
    assert policy['pinn_loss_weight'] == -1.0


def test_benchmark_override_survives_omitted_cli_weight():
    _, policy = resolved_runner(
        'go2_hard_pact_full_isaaclab', ['--benchmark_bard_active'], configured_weight=.01,
    )
    assert policy['pinn_loss_weight'] == -1.0
    assert policy['pinn_init_steps'] == -1
