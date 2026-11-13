"""Versioned learning-state envelope. No simulator state is inferred from weights.

Adapters retain their legacy weight keys. Transfer is opt-in at the target
environment, with a pure validation hook followed by explicit progress import
and fresh payload initialization. MULE deliberately has no adapter yet.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import warnings

import numpy as np
import torch

VERSION = 1
KEY = 'training_state'
OPERATIONS = ('resume', 'stage_transfer', 'weights_only')
RUNTIME_KEYS = {'max_iterations', 'save_interval', 'log_dir', 'log_root',
                'run_name', 'experiment_name', 'resume', 'load_run', 'checkpoint',
                'pretrained_path', 'export_hard_pact_start', 'hard_pact_start_filename'}
ADAPTIVE_FIELDS = ('learning_rate', 'current_entropy_coef', 'entropy_coef',
    'boot_mult', 'use_boot', 'pinn_weight', 'num_pinn_updates', 'current_vae_beta',
    '_qp_training_iteration', '_last_completed_iteration', 'actor_physics_warmup_factor')


def plain(value):
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool, type(None))): return value
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    return {k: plain(getattr(value, k)) for k in dir(value)
            if not k.startswith('_') and not callable(getattr(value, k))}


def structural(value):
    if isinstance(value, dict):
        return {k: structural(v) for k, v in value.items() if k not in RUNTIME_KEYS}
    if isinstance(value, list): return [structural(v) for v in value]
    return value


def _kind(obj):
    return type(obj).__module__ + '.' + type(obj).__qualname__


def adapter(runner):
    if _kind(runner.alg).endswith('PPOBeyondRobustness'): return 'beyond_robustness'
    if getattr(runner, 'is_hard_pact', False): return 'hard_pact'
    if getattr(runner, 'is_hard_pact_pos', False): return 'hard_pact_pos'
    raise ValueError('No complete-training-state adapter for this method (MULE deferred)')


def modules(runner):
    result = {'actor_critic': runner.alg.actor_critic}
    if hasattr(runner.alg, 'decoder'): result['decoder'] = runner.alg.decoder
    # Includes future explicitly attached target/EMA/normalization modules.
    for name, obj in vars(runner.alg).items():
        if isinstance(obj, torch.nn.Module) and all(obj is not m for m in result.values()):
            result[name] = obj
    return result


def optimizers(runner):
    names = (('optimizer', 'history_optimizer', 'load_optimizer')
             if adapter(runner) == 'beyond_robustness' else
             ('act_optimizer', 'enc_optimizer', 'decoder_optimizer'))
    return {n: getattr(getattr(runner.alg, n), 'optimizer', getattr(runner.alg, n)) for n in names}


def ownership(runner):
    names = {}
    for owner, module in modules(runner).items():
        for name, p in module.named_parameters(): names.setdefault(id(p), owner+'.'+name)
    result = {}
    for name, opt in optimizers(runner).items():
        result[name] = dict(type=_kind(opt), groups=[
            [dict(name=names[id(p)], shape=list(p.shape), dtype=str(p.dtype)) for p in g['params']]
            for g in opt.param_groups])
    return result


def wrappers(runner):
    return {name: dict(type=_kind(obj), reduction=obj._reduction, owned_only=obj._owned_only)
            for name, obj in vars(runner.alg).items()
            if hasattr(obj, '_optim') and hasattr(obj, '_owned_only')}


def boundary(runner):
    if getattr(runner, '_checkpoint_update_in_progress', False):
        raise RuntimeError('Checkpoint/load requires a completed rollout AND auxiliary update')
    alg = runner.alg
    if getattr(alg, 'pending', None) is not None or getattr(getattr(alg, 'storage', None), 'step', 0):
        raise RuntimeError('Checkpoint/load requires empty rollout storage and no pending action')
    transition = getattr(alg, 'transition', None)
    if transition is not None and getattr(transition, 'actions', None) is not None:
        raise RuntimeError('Checkpoint/load has a pending transition')


def contract(runner):
    method = adapter(runner)
    cfg = plain(runner.cfg if method == 'beyond_robustness' else runner.train_cfg)
    if method == 'beyond_robustness':
        # The SAVED active phase is restored; it is not inferred from constructor defaults.
        cfg.pop('training_phase', None)
        semantic = runner.alg.actor_critic.input_contract()
    else:
        semantic = getattr(runner, 'deployment_contract', None)
        if semantic is None: raise ValueError('HardPACT requires its semantic deployment contract')
    env = runner.env
    environment = (env.training_environment_contract() if hasattr(env, 'training_environment_contract')
                   else env.physical_contract() if hasattr(env, 'physical_contract')
                   else {'adapter': _kind(env), 'input_schema': plain(getattr(env, 'input_schema', None))})
    env_cfg = plain(getattr(env, 'cfg', {}))
    shared = {name: env_cfg[name] for name in ('control','normalization','noise','init_state',
              'commands','terrain','rewards','termination','asset') if name in env_cfg}
    shared['observation_dimensions'] = {name:value for name,value in env_cfg.get('env',{}).items()
        if name.startswith(('num_obs','num_priv','num_crit','num_exp','num_action'))}
    return dict(method=method, algorithm=_kind(runner.alg), config=structural(cfg),
        features=plain(getattr(runner, 'hard_pact_features', None)),
        semantics=plain(semantic), shared_environment=shared, environment=plain(environment),
        optimizer_ownership=ownership(runner), wrappers=wrappers(runner))


def rng_state(runner):
    local = {}
    # Explicit owner scopes; no traversal of solver workspaces or simulator internals.
    for prefix, owner in (('runner', runner), ('algorithm', runner.alg), ('environment', runner.env),
                          ('domain', getattr(runner.env, 'domain_rand_curriculum', None))):
        for name, value in vars(owner).items() if owner is not None else ():
            if isinstance(value, torch.Generator): local[prefix+'.'+name] = value.get_state()
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None, local=local)


def restore_rng(runner, state, transfer=False):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device topology differs; explicit migration required')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
    owners = dict(runner=runner, algorithm=runner.alg, environment=runner.env,
                  domain=getattr(runner.env, 'domain_rand_curriculum', None))
    for path, value in state['local'].items():
        prefix, name = path.split('.', 1)
        if transfer and prefix == 'environment': continue  # never import payload sampling RNG
        getattr(owners[prefix], name).set_state(value.cpu())


def _extra_state(runner):
    # Schedulers/scalers, if present, have their own state, not guessed scalar fields.
    return {n: deepcopy(v.state_dict()) for n, v in vars(runner.alg).items()
            if ('scheduler' in n or 'scaler' in n) and hasattr(v, 'state_dict')}


def envelope(runner, *, stage_end=False):
    boundary(runner)
    env = runner.env
    method = adapter(runner)
    try: sha = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                       'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): sha = 'unavailable'
    progress = (env.export_training_progress() if hasattr(env, 'export_training_progress') else None)
    return dict(version=VERSION, contract=contract(runner),
        models={n: m.state_dict() for n, m in modules(runner).items()},
        optimizers={n: o.state_dict() for n, o in optimizers(runner).items()},
        adaptive={n: deepcopy(getattr(runner.alg, n)) for n in ADAPTIVE_FIELDS if hasattr(runner.alg, n)},
        extra=_extra_state(runner), rng=rng_state(runner), transferable_progress=progress,
        next_iteration=runner.current_learning_iteration,
        interactions=getattr(runner, 'interactions', getattr(runner, 'tot_timesteps', 0)),
        total_time=getattr(runner, 'total_time', getattr(runner, 'tot_time', 0.)),
        phase=getattr(runner.alg, 'training_phase', 'ppo'),
        stage=deepcopy(getattr(runner, 'training_stage', {
            'name': 'stage'+str(getattr(getattr(env,'stage_state',None),'stage',2)), 'origin_iteration': 0})),
        qp_curriculum=(runner.qp_curriculum.state_dict() if hasattr(runner, 'qp_curriculum') else None),
        resolved_config=plain(getattr(env, 'resolved_config', getattr(env, 'cfg', None))),
        resolved_training=plain(runner.cfg if method == 'beyond_robustness' else runner.train_cfg),
        source_commit=sha, parent=deepcopy(getattr(runner, '_checkpoint_parent', None)),
        stage_end=bool(stage_end), exact_physics_supported=bool(getattr(env, 'supports_exact_resume', False)),
        unavailable=[] if progress is not None else ['portable environment curriculum adapter'])


def atomic_save(data, path, source=None):
    path = Path(path).resolve()
    if source and path.parent == Path(source).resolve().parent:
        raise ValueError('Write continuation checkpoints in a new directory; source runs are protected')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(data, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def validate(runner, data, operation='resume', *, learning_rate=None):
    """Pure preflight. Never load a module/optimizer to discover incompatibility."""
    boundary(runner)
    if operation not in OPERATIONS: raise ValueError('Unknown checkpoint operation')
    if operation == 'weights_only' and learning_rate is not None:
        raise ValueError('Fine-tuning LR override requires optimizer restoration, not weights_only')
    if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Fine-tuning learning rate must be finite and positive')
    state = data.get(KEY)
    if state is None: raise ValueError('Legacy partial checkpoint: use explicit allow_legacy or weights_only')
    if state['version'] != VERSION: raise ValueError('Unsupported training checkpoint version')
    current = contract(runner); saved = state['contract']
    for key in current:
        if operation == 'weights_only':
            # Playback/bootstrap may change terrain, command ranges, batch size,
            # training hyperparameters or disable QP. Learned input/head/action
            # semantics still have to match, not merely tensor dimensions.
            if key not in ('method','semantics','shared_environment'): continue
            if key == 'semantics':
                skip = {'qp_update','qp_objective','qp_inputs','grf_swing_gating','rss_features','actuator_execution'}
                if {k:v for k,v in current[key].items() if k not in skip} != {k:v for k,v in saved[key].items() if k not in skip}:
                    raise ValueError('Incompatible weight/input semantics')
                continue
            if key == 'shared_environment':
                for field in ('normalization','observation_dimensions'):
                    if current[key].get(field) != saved[key].get(field):
                        raise ValueError('Incompatible weight/input '+field)
                continue
        if key == 'environment' and operation == 'stage_transfer': continue
        if current[key] != saved[key]: raise ValueError('Incompatible training contract: '+key)
    for owner, model in modules(runner).items():
        own = model.state_dict(); old = state['models'][owner]
        if own.keys() != old.keys() or any(own[k].shape != old[k].shape or own[k].dtype != old[k].dtype for k in own):
            raise ValueError('Incompatible named model state: '+owner)
    if operation == 'weights_only': return state
    for name, opt in optimizers(runner).items():
        old = state['optimizers'][name]
        if len(old['param_groups']) != len(opt.param_groups): raise ValueError('Optimizer groups differ')
        for group, target in zip(old['param_groups'], opt.param_groups):
            if len(group['params']) != len(target['params']): raise ValueError('Optimizer ownership differs')
            for index, p in zip(group['params'], target['params']):
                for k, v in old['state'].get(index, {}).items():
                    if torch.is_tensor(v) and k != 'step' and v.shape != p.shape:
                        raise ValueError('Optimizer moment shape mismatch: '+name+'/'+k)
    if set(state['extra']) != set(_extra_state(runner)): raise ValueError('Scheduler/scaler inventory differs')
    if state['rng']['cuda'] is not None and torch.cuda.is_available() and len(state['rng']['cuda']) != torch.cuda.device_count():
        raise ValueError('CUDA RNG topology mismatch')
    owners = dict(runner=runner, algorithm=runner.alg, environment=runner.env,
                  domain=getattr(runner.env, 'domain_rand_curriculum', None))
    # Validate RNG payloads on independent generators, without consuming or
    # mutating the live streams before the rest of preflight succeeds.
    try:
        random.Random().setstate(state['rng']['python'])
        np.random.RandomState().set_state(state['rng']['numpy'])
        torch.Generator().set_state(state['rng']['torch'].cpu())
        if state['rng']['cuda'] is not None and torch.cuda.is_available():
            for index, value in enumerate(state['rng']['cuda']):
                torch.Generator(device=f'cuda:{index}').set_state(value.cpu())
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Invalid checkpoint RNG state') from error
    for path in state['rng']['local']:
        prefix, name = path.split('.', 1)
        if operation == 'stage_transfer' and prefix == 'environment': continue
        if not isinstance(getattr(owners.get(prefix), name, None), torch.Generator):
            raise ValueError('Missing local RNG owner: '+path)
        try:
            torch.Generator(device=getattr(owners[prefix], name).device).set_state(state['rng']['local'][path].cpu())
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError('Invalid local RNG state: '+path) from error
    if state['qp_curriculum'] is not None:
        from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum
        check = QPCurriculum(runner.alg.qp_config)
        check.load_state_dict(deepcopy(state['qp_curriculum']))
    if operation == 'stage_transfer':
        for hook in ('validate_stage_transfer', 'import_training_progress', 'initialize_transferred_payload'):
            if not hasattr(runner.env, hook): raise ValueError('Target environment has no '+hook+' adapter; liquid transfer is not implemented')
        if state['transferable_progress'] is None: raise ValueError('Source lacks portable curriculum state')
        runner.env.validate_stage_transfer(deepcopy(state))
    elif state['transferable_progress'] is not None and hasattr(runner.env, 'validate_training_progress'):
        runner.env.validate_training_progress(state['transferable_progress'])
    return state


def clear_gate_evidence(state):
    """Clear source-domain evidence, never attained progress or absolute schedule origins."""
    state = deepcopy(state)
    defaults = dict(reward_ema=None, best_reward_ema=-float('inf'), required_reward=0.,
        reward_history=[], ema=None, history=[], pre_qp_reference=None, reference_frozen=False,
        recovery_counter=0, streak=0, command_streak=0, threshold=None, advanced=False,
        baseline_frozen=False)
    if isinstance(state, dict):
        return {k: deepcopy(defaults[k]) if k in defaults else clear_gate_evidence(v) for k, v in state.items()}
    return state


def restore_learning(runner, state, *, learning_rate=None):
    for name, model in modules(runner).items(): model.load_state_dict(state['models'][name], strict=True)
    for name, opt in optimizers(runner).items(): opt.load_state_dict(state['optimizers'][name])
    for name, value in state['adaptive'].items(): setattr(runner.alg, name, deepcopy(value))
    for name, value in state['extra'].items(): getattr(runner.alg, name).load_state_dict(value)
    if learning_rate is not None:
        # Apply a uniform factor, retaining per-owner/per-group LR ratios and moments.
        factor = learning_rate / runner.alg.learning_rate
        for opt in optimizers(runner).values():
            for group in opt.param_groups:
                group['lr'] *= factor
                if 'initial_lr' in group: group['initial_lr'] *= factor
        for name in state['extra']:
            scheduler = getattr(runner.alg, name)
            for attr in ('base_lrs', '_last_lr'):
                if hasattr(scheduler, attr): setattr(scheduler, attr, [v*factor for v in getattr(scheduler, attr)])
        runner.alg.learning_rate = learning_rate
    runner.training_stage = deepcopy(state['stage'])
    if adapter(runner) == 'beyond_robustness':
        runner.alg.set_training_phase(state['phase']); runner.alg.iteration = state['next_iteration']
        runner.interactions = state['interactions']; runner.total_time = state['total_time']
        runner.control_steps = runner.interactions // runner.env.num_envs
    else:
        runner.current_learning_iteration = state['next_iteration']
        runner.tot_timesteps = state['interactions']; runner.tot_time = state['total_time']
        runner.alg._last_completed_iteration = state['next_iteration']-1


def archive_report(runner, path, state, operation, learning_rate=None, *, unavailable=(), exact=False):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    runner._checkpoint_parent = dict(path=str(Path(path).resolve()), sha256=digest.hexdigest(), operation=operation)
    report = dict(version=VERSION, parent=runner._checkpoint_parent, operation=operation,
        restored=(['networks'] if operation == 'weights_only' else
                  ['networks', 'named optimizers and moments', 'adaptive schedules', 'RNG', 'global progress', 'compatible curricula']),
        reset=([] if exact else ['physical episodes', 'histories/delay/contact/reward buffers', 'rollout/solver caches']) +
              (['source performance gates', 'payload sampling', 'stage-local origin'] if operation == 'stage_transfer' else []),
        overrides=dict(learning_rate=learning_rate, target_stage=getattr(runner, 'training_stage', None),
                       runtime_policy='Only budget/save/log/provenance keys may differ; source and target snapshots below'),
        unavailable=list(unavailable), exact_environment_restored=exact, exact_physx_continuation=False,
        source_configuration=None if state is None else state['resolved_config'],
        source_training=None if state is None else state['resolved_training'],
        target_configuration=plain(getattr(runner.env, 'resolved_config', getattr(runner.env, 'cfg', None))),
        target_training=plain(getattr(runner, 'train_cfg', runner.cfg)))
    runner.last_transfer_report = report
    if runner.log_dir is not None:
        destination = Path(runner.log_dir)
        if destination.resolve() == Path(path).resolve().parent: raise ValueError('Source log directory cannot be overwritten')
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'training_transfer_report.json').write_text(json.dumps(plain(report), indent=2))
    return report


def warn_legacy():
    missing = ['named optimizer ownership', 'complete RNG', 'adaptive scheduler state',
               'portable curricula/terrain', 'semantic training envelope']
    warnings.warn('Legacy partial initialization; NOT full continuation. Unavailable: '+', '.join(missing), RuntimeWarning)
    return missing
