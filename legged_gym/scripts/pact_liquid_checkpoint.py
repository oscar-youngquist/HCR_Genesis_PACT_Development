"""Resolve legacy PACT inference without executing saved Python configs."""
import ast
import operator
from pathlib import Path


def _constant(node):
    if isinstance(node, ast.BinOp):
        operation = {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
                     ast.Div:operator.truediv}.get(type(node.op))
        if operation is not None:
            return operation(_constant(node.left),_constant(node.right))
    return ast.literal_eval(node)


def _section(node, *names):
    for name in names:
        node = next((n for n in node.body if isinstance(n,ast.ClassDef) and n.name==name),None)
        if node is None:
            return None
    return node


def _values(node, keys):
    result = {}
    if node is not None:
        for item in node.body:
            if isinstance(item,ast.Assign):
                for target in item.targets:
                    if isinstance(target,ast.Name) and target.id in keys:
                        try:
                            result[target.id] = _constant(item.value)
                        except (ValueError,TypeError) as exc:
                            raise ValueError(f'Cannot safely read saved config field {target.id}') from exc
    return result


def restore_inference_config(cfg, train, checkpoint, robot, config=None):
    """Restore controller/observation semantics, keeping liquid evaluation flags.

    Parse literal assignments and simple arithmetic only. Never import/execute
    the Python files stored beside a checkpoint. Training, terrain and domain
    randomization schedules are deliberately not restored for this player.
    """
    candidates = [checkpoint.parent/f'{robot}_pact_config.py',
                  checkpoint.parent/f'{robot}_pact_water_config.py']
    source = Path(config).resolve() if config else next((p for p in candidates if p.is_file()),None)
    if source is None:
        return None
    tree = ast.parse(source.read_text())
    env = next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name.endswith('Cfg')),None)
    ppo = next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name.endswith('CfgPPO')),None)
    if env is None:
        raise ValueError(f'{source}: no legacy PACT environment config found')
    asset = _values(_section(env,'asset'),{'name','dof_names','terminate_after_contacts_on'})
    if asset.get('name',robot) != robot:
        raise ValueError(f'{source} is for {asset["name"]}, not {robot}')
    if 'dof_names' in asset:
        if set(asset['dof_names']) != set(cfg.asset.dof_names):
            raise ValueError('Saved checkpoint joint names do not match the selected robot')
        cfg.asset.dof_names = asset['dof_names']
    if 'terminate_after_contacts_on' in asset:
        cfg.asset.terminate_after_contacts_on = asset['terminate_after_contacts_on']
    for section, keys in {
        'control': {'dt','decimation','action_scale','torque_scale','stiffness','damping',
                    'tradeoff_init_weights','tradeoff_final_weights'},
        'init_state': {'default_joint_angles','default_joint_torques','pos','rot'},
        'normalization': {'clip_observations','clip_actions'},
        'env': {'episode_length_s','fail_to_terminal_time_s'},
        'termination': {'termination_terms','roll_threshold','pitch_threshold','height_min','height_max'},
    }.items():
        for key,value in _values(_section(env,section),keys).items():
            setattr(getattr(cfg,section),key,value)
    dynamics_keys = {f'joint_{name}_range{suffix}' for name in
                    ('armature','friction','stiffness','damping') for suffix in ('','_start','_end')}
    dynamics_keys.update(f'randomize_joint_{name}' for name in ('armature','friction','stiffness','damping'))
    dynamics = _values(_section(env,'domain_rand'),dynamics_keys)
    for key,value in dynamics.items():
        setattr(cfg.domain_rand,key,value)
    # A saved fixed (non-curriculum) range takes precedence over inherited
    # curriculum ranges from today's task configuration.
    for name in ('friction','stiffness','damping'):
        key = f'joint_{name}_range'
        if key in dynamics and key+'_end' not in dynamics:
            setattr(cfg.domain_rand,key+'_start',dynamics[key])
            setattr(cfg.domain_rand,key+'_end',dynamics[key])
    scales = cfg.normalization.obs_scales
    for key,value in _values(_section(env,'normalization','obs_scales'),
                             set(vars(scales)) | set(dir(scales))).items():
        setattr(scales,key,value)
    for key,value in _values(_section(ppo,'policy') if ppo else None,{'activation'}).items():
        setattr(train.policy,key,value)
    cfg.sim.dt = cfg.control.dt/cfg.control.decimation
    return str(source)


def set_joint_dynamics_midpoints(cfg):
    """Match the legacy player's midpoint convention, using saved final ranges.

    The randomization APIs remain enabled for features enabled in training;
    [midpoint, midpoint] applies fixed values at creation and every reset.
    Global domain-randomization curriculum progression remains disabled.
    """
    import math
    result = {}
    dr = cfg.domain_rand
    for name in ('armature','friction','stiffness','damping'):
        key = f'joint_{name}_range'
        bounds = list(getattr(dr,key+'_end',getattr(dr,key,None)))
        if len(bounds)!=2 or not all(math.isfinite(v) for v in bounds) or bounds[0]>bounds[1]:
            raise ValueError(f'Invalid saved training range for joint {name}: {bounds}')
        midpoint = (bounds[0]+bounds[1])/2
        enabled = bool(getattr(dr,f'randomize_joint_{name}'))
        setattr(dr,key,[midpoint,midpoint])
        if name!='armature':
            setattr(dr,key+'_start',[midpoint,midpoint])
            setattr(dr,key+'_end',[midpoint,midpoint])
        result[name] = dict(training_range=bounds,midpoint=midpoint,enabled=enabled)
    return result


def infer_policy_contract(state):
    """Weights authoritatively determine supported coupled-PACT dimensions."""
    def layers(prefix):
        return [value for key,value in sorted(state.items(),
            key=lambda item:int(item[0].split('.')[1]) if item[0].startswith(prefix+'.') else -1)
            if key.startswith(prefix+'.') and key.endswith('.weight')]
    try:
        latent = state['context_encoder.ce_out_mean.weight'].shape[0]
        explicit = state['context_encoder.ce_velo_mean.weight'].shape[0]
        actor, critic = layers('act_trunk'),layers('critic')
        observations = actor[0].shape[1]-latent-explicit
        history_input = state['context_encoder.ce_in.weight'].shape[1]
        actions = state['act_pos_out.weight'].shape[0]
        if observations != 57 or history_input % observations or actions != 12:
            raise ValueError('Liquid player requires a 57-observation, 12+12-action legacy PACT policy')
        if state['act_tau_out.weight'].shape[0] != actions or state['std'].numel() != 2*actions:
            raise ValueError('Checkpoint is not a coupled position/feedforward PACT policy')
        return dict(observations=observations,history=history_input//observations,
            latent=latent,explicit=explicit,actions=actions,
            actor_layers=[w.shape[0] for w in actor],
            critic_layers=[w.shape[0] for w in critic[:-1]],critic_inputs=critic[0].shape[1],
            encoder_layers=[state['context_encoder.ce_in.weight'].shape[0],
                            state['context_encoder.ce_h1.weight'].shape[0]])
    except (KeyError,IndexError) as exc:
        raise ValueError('Unsupported checkpoint: expected legacy ActorCritic_PACT state keys') from exc
