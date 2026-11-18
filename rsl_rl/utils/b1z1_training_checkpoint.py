"""PACT learning continuation at completed PPO boundaries, not PhysX replay.

Legacy top-level network/optimizer keys remain usable by existing evaluators.
Physical episodes, FiLM history, action delays and rollout caches start fresh.
"""
from copy import deepcopy
import os
from pathlib import Path
import random
import subprocess
import tempfile

import numpy as np
import torch


VERSION = 1
RUNTIME = {'max_iterations', 'save_interval', 'resume', 'load_run', 'checkpoint',
           'run_name', 'experiment_name', 'pretrained_path', 'enable_additional_diagnostics'}
ADAPTIVE = ('learning_rate', 'pinn_weight', 'pinn_updates', '_spec_u', 'rejection_progress')
ENV_PROGRESS = ('command_ranges', 'reward_scales', 'tradeoff_step_ctr',
                'common_step_counter', 'global_steps')
SIM_PROGRESS = ('feedforward_tau_weight_clean', 'feedback_tau_weight_clean',
                '_terrain_levels', '_terrain_types', '_env_origins')


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (str, int, float, bool, type(None))): return value
    return {k: plain(getattr(value, k)) for k in dir(value)
            if not k.startswith('_') and not callable(getattr(value, k))}


def optimizers(runner):
    return dict(actor_optimizer=runner.alg.actor_optimizer.optimizer,
                auxiliary_optimizer=runner.alg.auxiliary_optimizer,
                decoder_optimizer=runner.alg.decoder_optimizer)


def contract(runner):
    names = {}
    for prefix, model in (('actor', runner.actor_critic), ('decoder', runner.privileged_decoder)):
        for name, p in model.named_parameters(): names.setdefault(id(p), prefix+'.'+name)
    config = deepcopy(plain(runner.cfg))
    config['runner'] = {k: v for k, v in config.get('runner', {}).items() if k not in RUNTIME}
    environment = plain(runner.env.cfg)
    return dict(training=config, environment=environment,
        grf_semantics=plain(getattr(runner.env.simulator, 'grf_semantics', None)),
        ownership={key: [[names[id(p)] for p in group['params']] for group in opt.param_groups]
                   for key, opt in optimizers(runner).items()})


def boundary(runner):
    if (getattr(runner, '_update_in_progress', False)
            or getattr(runner.alg.storage, 'step', 0)
            or getattr(runner.alg.transition, 'actions', None) is not None):
        raise RuntimeError('Save/resume requires a completed rollout and PPO/auxiliary update')


def progress(runner):
    return {owner: {name: deepcopy(getattr(obj, name)) for name in fields if hasattr(obj, name)}
            for owner, obj, fields in (('environment', runner.env, ENV_PROGRESS),
                                      ('simulator', runner.env.simulator, SIM_PROGRESS))}


def restore_progress(runner, state):
    for owner, obj in (('environment', runner.env), ('simulator', runner.env.simulator)):
        for name, value in state[owner].items():
            current = getattr(obj, name)
            if torch.is_tensor(current): current.copy_(value.to(current.device))
            else: setattr(obj, name, deepcopy(value))


def rng(runner):
    device = torch.device(runner.device)
    # Only the training device is owned here; unrelated GPUs/jobs are untouched.
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(runner, state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), torch.device(runner.device))


def capture(runner, iteration):
    boundary(runner)
    if getattr(runner, '_resume_state', None) is not None:
        raise RuntimeError('Start fresh resumed episodes before saving another checkpoint')
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                           'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    return dict(version=VERSION, contract=deepcopy(runner._training_contract),
        source_revision=revision,
        next_iteration=iteration, total_timesteps=runner.total_timesteps, total_time=runner.total_time,
        completed_rewards=list(getattr(runner, '_completed_rewards', ())),
        completed_lengths=list(getattr(runner, '_completed_lengths', ())),
        adaptive={name: deepcopy(getattr(runner.alg, name)) for name in ADAPTIVE if hasattr(runner.alg, name)},
        progress=progress(runner), rng=rng(runner),
        continuation='learning_state_with_fresh_episodes', exact_physics=False)


def validate(runner, checkpoint):
    """Preflight before touching live weights, moments, curricula or RNG."""
    boundary(runner)
    state = checkpoint['training_state']
    if state['version'] != VERSION: raise ValueError('Unsupported B1Z1 training-state version')
    if state['contract'] != runner._training_contract:
        differing = [k for k in state['contract'] if state['contract'][k] != runner._training_contract.get(k)]
        raise ValueError('Incompatible B1Z1 training contract: '+', '.join(differing)+
                         '; use load_optimizer=False for an intentional weights-only initialization')
    if state['next_iteration'] != checkpoint['iteration']:
        raise ValueError('Checkpoint iteration fields disagree')
    if checkpoint.get('optimizer_partition_version') != 3:
        raise ValueError('Full continuation requires optimizer partition version 3')
    for model, key in ((runner.actor_critic, 'model_state_dict'),
                       (runner.privileged_decoder, 'privileged_decoder_state_dict')):
        current, saved = model.state_dict(), checkpoint[key]
        if current.keys() != saved.keys() or any(current[k].shape != saved[k].shape for k in current):
            raise ValueError('Incompatible network state: '+key)
    for key, opt in optimizers(runner).items():
        saved = checkpoint[key]
        if len(saved['param_groups']) != len(opt.param_groups): raise ValueError('Optimizer groups differ')
        for old, new in zip(saved['param_groups'], opt.param_groups):
            if len(old['params']) != len(new['params']): raise ValueError('Optimizer ownership differs')
            for index, p in zip(old['params'], new['params']):
                for name, value in saved['state'].get(index, {}).items():
                    if torch.is_tensor(value) and name != 'step' and value.shape != p.shape:
                        raise ValueError('Optimizer moment shape differs: '+key)
    for owner, obj in (('environment', runner.env), ('simulator', runner.env.simulator)):
        for name, value in state['progress'][owner].items():
            current = getattr(obj, name)
            if torch.is_tensor(current) and current.shape != value.shape:
                raise ValueError('Curriculum shape differs: '+name)
    random.Random().setstate(state['rng']['python'])
    np.random.RandomState().set_state(state['rng']['numpy'])
    torch.Generator().set_state(state['rng']['torch'].cpu())
    if state['rng']['cuda'] is not None:
        if torch.device(runner.device).type != 'cuda':
            raise ValueError('CUDA training continuation needs a CUDA device; weights-only supports CPU')
        torch.Generator(device=runner.device).set_state(state['rng']['cuda'].cpu())


def atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(data, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
