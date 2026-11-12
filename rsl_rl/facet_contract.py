"""FACET checkpoint contracts; metadata is explicit, not inferred from a task name."""
import json
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy


def apply_config_snapshot(cfg, saved):
    """Restore owned nested sections before construction; never mutate classes."""
    for name, value in saved.items():
        if not hasattr(cfg, name) or callable(getattr(cfg, name)):
            continue
        current = getattr(cfg, name)
        if isinstance(value, dict) and not isinstance(current, dict):
            apply_config_snapshot(current, value)
        else:
            setattr(cfg, name, deepcopy(value))


def contract_from_env(env):
    from legged_gym.utils.helpers import class_to_dict, _json_config_value
    cfg = class_to_dict(env.cfg)
    result = contract_from_snapshot(dict(environment=cfg, observations=env.get_facet_observation_schema()))
    result['reference'] = asdict(env.cfg.facet_reference)
    result['tracking'] = asdict(env.cfg.facet_tracking)
    result['environment'] = cfg
    result['control_dt'] = float(env.dt)
    result['kinematics'] = {key: getattr(env, 'z1_'+key).detach().cpu().tolist()
        for key in ('joint_offsets', 'joint_axes', 'link00_offset', 'ee_offset')}
    return _json_config_value(result)


def contract_from_snapshot(snapshot):
    cfg = snapshot['environment']
    return dict(schema=snapshot['observations'], control=cfg['control'],
        clip_actions=cfg['normalization']['clip_actions'], joints=cfg['asset']['dof_names'],
        default_joint_angles=cfg['init_state']['default_joint_angles'],
        delay={k: v for k, v in cfg['domain_rand'].items() if 'delay' in k},
        reference=cfg['facet_reference'], tracking={k: v for k, v in cfg['facet_tracking'].items() if k != 'validate'},
        environment=cfg,
        control_dt=cfg['control']['dt'], kinematics=None,
        action_convention='raw Gaussian in PPO; clipped issued positions before delay; PD uses existing Pos scaling and held joints',
        observations='raw SI; heading-relative command XY/world Z; trunk FK; newest-first histories')


def checkpoint_contract(state, path=None, legacy_contract=None):
    if 'contract' in state:
        return state['contract']
    if legacy_contract is not None:
        return legacy_contract
    sidecar = Path(path).parent/'facet_resolved_config.json' if path else None
    if sidecar and sidecar.exists():
        return contract_from_snapshot(json.loads(sidecar.read_text()))
    raise ValueError('Stage 5 checkpoint lacks deployment/observation metadata: supply its original facet_resolved_config.json or explicit legacy_contract; do not guess')


def validate_contract(saved, current):
    # JSON normalizes tuple/list differences, not semantic differences.
    for key in ('schema', 'control', 'clip_actions', 'joints', 'default_joint_angles', 'delay', 'control_dt'):
        if json.dumps(saved[key], sort_keys=True) != json.dumps(current[key], sort_keys=True):
            raise ValueError('Incompatible FACET contract: '+key)
    if saved.get('kinematics') is not None and saved['kinematics'] != current['kinematics']:
        raise ValueError('Incompatible FACET FK geometry')
    # These settings determine physical/task semantics and cannot safely be
    # restored after simulator construction. The FACET launcher restores them first.
    for key in ('commands', 'terrain', 'domain_rand', 'rewards', 'asset', 'noise', 'termination', 'init_state'):
        old, new = saved.get('environment', {}), current.get('environment', {})
        if key in old and json.dumps(old[key], sort_keys=True) != json.dumps(new.get(key), sort_keys=True):
            raise ValueError('Incompatible FACET environment section: '+key+'; restore saved configuration before construction')
