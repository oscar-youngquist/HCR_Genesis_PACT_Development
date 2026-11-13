"""Portable HardPACT progress versus fresh episode state. No PhysX snapshots."""
from copy import deepcopy
import torch
from rsl_rl.utils.training_checkpoint import plain, clear_gate_evidence


def environment_contract(env):
    return dict(version=1, adapter='go2_hard_pact', simulator=type(env.simulator).__module__,
                settings=plain(env.cfg), payload_model='persistent_wrench')


def export_progress(env):
    s = env.simulator
    terrain = {n: getattr(s, n).detach().clone() for n in
               ('terrain_levels', 'terrain_types') if torch.is_tensor(getattr(s, n, None))}
    return deepcopy(dict(version=1, domain=env.domain_rand_curriculum_state_dict(),
        commands=env.command_curriculum_state_dict(), terrain=terrain,
        reward_scales=env.reward_scales, common_step_counter=env.common_step_counter,
        iteration=getattr(env, '_terrain_curriculum_iteration', 0),
        tradeoff={name:getattr(owner,name).clone() for owner,names in
            ((env,('tradeoff_step_ctr',)),(s,('feedforward_tau_weight','feedback_tau_weight',
                'feedforward_tau_weight_clean','feedback_tau_weight_clean')))
            for name in names if torch.is_tensor(getattr(owner,name,None))}))


def validate_progress(env, state):
    if state['version'] != 1: raise ValueError('Unsupported HardPACT curriculum state')
    if set(state['domain']['progress']) != set(env.domain_rand_curriculum.progress):
        raise ValueError('Domain curriculum families differ')
    if set(state['reward_scales']) != set(env.reward_scales): raise ValueError('Reward terms differ')
    for name,value in state.get('tradeoff',{}).items():
        owner=env if name=='tradeoff_step_ctr' else env.simulator
        if getattr(owner,name,None) is None or getattr(owner,name).shape!=value.shape:
            raise ValueError('Controller curriculum layout differs: '+name)
    for name, value in state['terrain'].items():
        live = getattr(env.simulator, name, None)
        if live is None or live.shape != value.shape: raise ValueError('Terrain layout mismatch')
        if (value < 0).any(): raise ValueError('Negative terrain index')
    origins = getattr(env.simulator, '_terrain_origins', None)
    if origins is not None:
        for name, maximum in (('terrain_levels', origins.shape[0]), ('terrain_types', origins.shape[1])):
            if name in state['terrain'] and (state['terrain'][name] >= maximum).any():
                raise ValueError('Terrain index outside destination geometry')


def import_progress(env, state, *, transfer=False):
    validate_progress(env, state)
    state = clear_gate_evidence(state) if transfer else deepcopy(state)
    env.load_domain_rand_curriculum_state_dict(state['domain'])
    env.load_command_curriculum_state_dict(state['commands'])
    env.reward_scales = deepcopy(state['reward_scales'])
    env.common_step_counter = state['common_step_counter']
    env._terrain_curriculum_iteration = state['iteration']
    s = env.simulator
    for name,value in state.get('tradeoff',{}).items():
        owner=env if name=='tradeoff_step_ctr' else s
        getattr(owner,name).copy_(value)
    for name, value in state['terrain'].items(): getattr(s, name).copy_(value)
    if hasattr(s, '_terrain_origins'):
        s._env_origins.copy_(s._terrain_origins[s.terrain_levels, s.terrain_types])


@torch.inference_mode()
def fresh_episodes(env):
    """Reset using existing simulator APIs; never promote terrain on a resume reset."""
    ids = torch.arange(env.num_envs, device=env.device)
    initialized = env.init_done
    command_enabled = env.cfg.commands.curriculum
    tradeoff = getattr(env, 'use_tradeoff', False)
    env.init_done = False
    env.cfg.commands.curriculum = False
    env.use_tradeoff = False
    recorder = getattr(env, '_episode_diagnostics', None)
    env._episode_diagnostics = None
    # Fresh physics must resample properties, not retain a source body's cadence.
    if hasattr(env.simulator, '_reset_cadence'): del env.simulator._reset_cadence
    try:
        env.reset_idx(ids)
    finally:
        env.init_done = initialized
        env.cfg.commands.curriculum = command_enabled
        env.use_tradeoff = tradeoff
        env._episode_diagnostics = recorder
    if recorder is not None:
        recorder.clear(); recorder.complete_episode.zero_(); recorder.last_iteration = None
    for name in ('obs_buf', 'obs_history', 'actions', 'last_actions', 'llast_actions', 'last_dof_vel',
                 'last_obs_hist', 'llast_obs_hist', 'last_contacts', 'contact_filt', 'feet_air_time', 'episode_length_buf',
                 'fail_buf', 'action_history', 'action_queue', 'action_delay_queue', 'actions_queue',
                 '_action_replay_valid_queue', '_action_replay_obs_queue', '_action_replay_hist_queue'):
        value = getattr(env, name, None)
        if torch.is_tensor(value): value.zero_()
    for values in getattr(env, 'episode_sums', {}).values(): values.zero_()
    env.extras = {}
    env._qp_sampled_transition = {}
    env.last_transition = {}
    env._hard_pact_policy_context_ready = False
    if hasattr(env, '_end_disturbance_interval'):
        env._capture_realized_inertial_randomization()
        env._pending_disturbance_transition = env._end_disturbance_interval()
    for name in ('_hard_pact_rollout_qp',):
        qp = getattr(env, name, None)
        if qp is not None and hasattr(qp, 'clear_warm_start'): qp.clear_warm_start(ids)
    if hasattr(env, 'begin_command_curriculum_iteration'): env.begin_command_curriculum_iteration()
    # Refresh measured states after writes without advancing physics (also
    # required by Genesis, whose reset_idx does not refresh all cached tensors).
    env.simulator.post_physics_step()
    # Do not simulate a zero-action step here (legacy reset() does).
    env.compute_observations()


def reject_unimplemented_transfer(env, state):
    raise ValueError('This HardPACT environment has no cross-payload adapter. '
                     'Liquid transfer requires explicit semantic validation and payload initialization hooks.')
