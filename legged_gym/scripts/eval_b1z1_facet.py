"""FACET portable-student play/evaluation with shared, recorded scenario specifications.

Baseline tasks reuse Exp1's frozen model loader and Original->Reject interface.
Only this process's environment callbacks are changed; training code is untouched.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_b1z1_exp1 import (EVAL_CONFIG, EvaluationAdapter, configure, build_actor,
    inference, simulator_family, isolated_rng, util)
from rsl_rl.facet_contract import apply_config_snapshot


def scenarios(seed, trial_ids, config):
    result = []
    for trial in trial_ids:
        condition = ('ee-only', 'base-only', 'combined')[trial % 3]
        motion = ('standing', 'walking')[(trial//3) % 2]
        direction = ('lateral', 'opposing')[(trial//6) % 2]
        stiffness = (4., 16., 28.)[(trial//12) % 3]
        if config.get('impedance_fixed'):
            stiffness = config['impedance_fixed']['ee_stiffness']
        result.append(Scenario(dict(trial_id=trial, evaluation_seed=seed, condition=condition,
            motion=motion, direction=direction, stiffness=stiffness, initial_heading=0.,
            events=config['event_times'], horizon=config['horizon_seconds'], impedance_fixed=config.get('impedance_fixed'))))
    return result


class Scenario:
    def __init__(self, metadata):
        self.metadata = metadata

    def at(self, t):
        m = self.metadata
        command = np.array([.3 if m['motion'] == 'walking' else 0., 0., 0.])
        a, b, c = m['events']
        magnitude = 20. if a <= t < b else (35. if b <= t < c else 0.)
        force = np.array([0., magnitude, 0.]) if m['direction'] == 'lateral' else np.array([-magnitude, 0., 0.])
        yaw = m['initial_heading']
        force[:2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]] @ force[:2]
        load = np.zeros(9)
        if m['condition'] != 'base-only':
            load[:3] = force
        if m['condition'] != 'ee-only':
            load[3:6] = force*.7
            load[8] = magnitude*.1  # Independent physical yaw couple, Nm.
        return command, np.array([.5, .1, 0.]), load, int(a <= t)+int(b <= t)+int(c <= t)


class Adapter(EvaluationAdapter):
    def __init__(self, env, config, seed):
        import torch
        self.env, self.sim, self.config, self.seed = env, env.simulator, config, seed
        self.saved_reset, self.saved_observations = env.reset_idx, env.compute_observations
        self.noise = bool(env.add_noise)
        self.pending, self.substeps, self.scenarios, self.step_index = None, [], [], -1
        self.torso_indices = [self.sim._body_names.index(k) for k in config['torso_body_names'] if k in self.sim._body_names]
        if not hasattr(env, 'base_torque_ext_world'):
            env.base_torque_ext_world = torch.zeros_like(env.base_force_ext_world)
        env.add_noise = False
        # Preserve actual termination/reward/terminal capture, but retain the
        # physical successor for evaluation rather than teleporting on failure.
        env.reset_idx = lambda ids: None
        env._resample_commands = lambda ids: None
        env._push_gripper = lambda ids: None
        env._push_robot_base = lambda ids: None
        env._post_physics_step_callback = self.post_control
        env.compute_observations = self.observations
        if hasattr(self.sim, '_check_base_pos_out_of_bound'):
            self.sim._check_base_pos_out_of_bound = lambda: None
        self.sim.collect_b1z1_bard_interval = False
        env.reward_scales.pop('tracking_arm_feedforward', None)

    def reset(self, batch):
        import torch
        super().reset(batch)  # Explicit feature-keyed initial state/mechanics, not seed-only matching.
        for slot, scenario in enumerate(batch):
            scenario.metadata['initial_heading'] = float(self.sim.base_euler[slot, 2])
        if hasattr(self.env, 'facet_reference'):
            self.env._reset_facet_reference(self.env.all_env_ids)
            p = self.env.facet_reference._parameters
            for slot, scenario in enumerate(batch):
                if scenario.metadata.get('impedance_fixed'):
                    for name, value in scenario.metadata['impedance_fixed'].items():
                        p[name][slot] = value
                    continue
                k = scenario.metadata['stiffness']
                for name in ('base_stiffness', 'ee_stiffness', 'yaw_stiffness'):
                    p[name][slot] = k
                for name in ('base_damping', 'ee_damping', 'yaw_damping'):
                    p[name][slot] = 1.8*math.sqrt(k)
                p['base_mass'][slot], p['ee_mass'][slot], p['yaw_inertia'][slot] = 4., 2., 1.5
            # History parameters share the same buffers through its broadcast views.
        self.publish_targets(0.)
        self.env.ee_start_sphere.copy_(self.env.curr_ee_goal_sphere)
        self.env.ee_goal_sphere.copy_(self.env.curr_ee_goal_sphere)


def sensor_inputs(env, impedance, seed, batch, step):
    """Sensor-only adapter: never reads reference states, labels, contacts or velocities of bodies."""
    import torch
    from legged_gym.utils.math_utils import quat_rotate_inverse
    from legged_gym.envs.b1z1.b1z1_facet.observations import nominal_command
    s = env.simulator
    data = s._robot.data
    q = data.joint_pos[:, s._dof_indices].clone()
    qd = data.joint_vel[:, s._dof_indices].clone()
    gravity = quat_rotate_inverse(data.root_link_quat_w[:, [1, 2, 3, 0]], q.new_tensor([0., 0., -1.]).expand(len(q), 3))
    if env.cfg.noise.add_noise:
        noise = env.cfg.noise
        for slot, scenario in enumerate(batch):
            rng = np.random.default_rng(util.seed_for(seed, scenario.metadata['trial_id'], 'facet_sensors', step))
            for value, name in ((gravity, 'gravity'), (q, 'dof_pos'), (qd, 'dof_vel')):
                value[slot] += torch.as_tensor(rng.uniform(-1, 1, value.shape[1]), device=value.device)*getattr(noise.noise_scales, name)*noise.noise_level
    return nominal_command(env), impedance, gravity, q, qd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='b1z1_facet')
    parser.add_argument('--artifact', help='TorchScript sensor-student artifact for FACET')
    parser.add_argument('--checkpoint', help='Existing baseline checkpoint, not a FACET artifact')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-envs', type=int, default=6)
    parser.add_argument('--episodes', type=int, default=36)
    parser.add_argument('--horizon-seconds', type=float, default=14.)
    parser.add_argument('--event-times', nargs=3, type=float, default=[3., 7., 11.])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', default='cuda:0')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.task == 'b1z1_facet' and not args.artifact:
        parser.error('FACET requires --artifact; explicitly export the initialized student or a copied teacher first')
    if args.task != 'b1z1_facet' and not args.checkpoint:
        parser.error('Baseline evaluation requires --checkpoint')
    if not 0 < args.event_times[0] < args.event_times[1] < args.event_times[2] < args.horizon_seconds or min(args.num_envs, args.episodes) < 1:
        parser.error('Require positive counts and onset < change < removal < horizon')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use an empty evaluation directory')
    os.environ['SIMULATOR'] = simulator_family(args.task)
    import torch
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.facet_deployment import load_student
    facet = args.task == 'b1z1_facet'
    model, metadata = load_student(args.artifact, args.gpu) if facet else (None, {})
    training = util.materialize(task_registry.train_cfgs[args.task])
    cfg = util.materialize(task_registry.env_cfgs[args.task])
    if facet:
        saved = metadata['contract']
        if 'environment' not in saved:
            raise ValueError('Simulator evaluation requires saved environment configuration')
        apply_config_snapshot(cfg, saved['environment'])
        cfg.facet_observations.enabled = False  # No privileged packet constructed for policy use.
    config = deepcopy(EVAL_CONFIG)
    config.update(event_times=args.event_times, horizon_seconds=args.horizon_seconds)
    if facet and saved['reference'].get('sampling_mode', 'fixed') == 'fixed':
        from legged_gym.envs.b1z1.b1z1_facet.reference import PARAMETERS
        config['impedance_fixed'] = {key: saved['reference'][key] for key in PARAMETERS}
        if saved['reference'].get('sample_ranges'):
            raise ValueError('Legacy custom impedance ranges require an explicit represented evaluation profile')
    config['velocity_ranges'][0] = [0., .3]
    cfg = configure(cfg, args, config)
    if facet:
        from legged_gym.envs.b1z1.b1z1_facet.tracking import FACETTrackingCfg
        from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReferenceCfg
        cfg.facet_tracking = FACETTrackingCfg(**saved['tracking'])
        cfg.facet_reference = FACETReferenceCfg(**saved['reference'])
    env_class = task_registry.get_task_class(args.task)
    reject = args.task in ('b1z1_unifp_original', 'b1z1_unifp_reject')
    if reject:
        env_class = task_registry.get_task_class('b1z1_unifp_reject')
        for key, value in vars(util.materialize(task_registry.env_cfgs['b1z1_unifp_reject'].commands)).items():
            if not hasattr(cfg.commands, key):
                setattr(cfg.commands, key, value)
        cfg.commands.use_external_impedance_compensation = True
        cfg.commands.compensate_ee_external_force = cfg.commands.compensate_base_external_force = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    writer = util.NumericWriter(args.output_dir, 2048)
    trials, initial = [], []
    start = time.perf_counter()
    inference_seconds, rollout_seconds, evaluation_interactions = 0., 0., 0
    try:
        with isolated_rng(args.seed, args.gpu):
            env = env_class(cfg, class_to_dict(cfg.sim), args.gpu, args.headless)
            if not facet:
                state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                model = build_actor(env, training, state)
        if reject:
            env._staged_force_curriculum.beta = 1.
        adapter = Adapter(env, config, args.seed)
        s = env.simulator
        steps = math.ceil(args.horizon_seconds/env.dt)
        for trial_ids in util.batches(args.episodes, args.num_envs):
            batch = scenarios(args.seed, trial_ids, config)
            # Partial final batches do not create extra trial outcomes.
            padded = batch+[batch[-1]]*(env.num_envs-len(batch))
            adapter.reset(padded)
            for slot, scenario in enumerate(batch):
                row = dict(scenario.metadata)
                row['joint_position'] = s.dof_pos[slot].cpu().tolist()
                row['base_position_relative_origin'] = (s.base_pos[slot]-s._env_origins[slot]).cpu().tolist()
                row['quaternion_xyzw'] = s.base_quat[slot].cpu().tolist()
                row['mass'] = adapter.body_masses[slot].cpu().tolist()
                for key in ('_kp_scale', '_kd_scale', '_motor_strength', '_joint_armature', '_joint_friction', '_joint_damping'):
                    row[key] = getattr(s, key)[slot].cpu().tolist()
                row['scenario_hash'] = util.digest(scenario.metadata)
                initial.append(row)
            alive = np.ones(len(batch), dtype=bool)
            records = [[] for _ in batch]
            hidden = torch.zeros(env.num_envs, 128, device=env.device)
            sensors = torch.zeros(env.num_envs, 4, 3+2*len(metadata['contract']['schema']['observed_joint_names']), device=env.device) if facet else None
            actions = torch.zeros(env.num_envs, 3, 17, device=env.device)
            issued = torch.zeros(env.num_envs, 17, device=env.device)
            if facet:
                from legged_gym.envs.b1z1.b1z1_facet.reference import PARAMETERS
                impedance = torch.cat([env.facet_reference._parameters[k] for k in PARAMETERS], -1).clone()
            for step in range(steps):
                adapter.step_index = step
                tick = time.perf_counter()
                with torch.no_grad():
                    if facet:
                        init = torch.full((env.num_envs, 1), step == 0, dtype=torch.bool, device=env.device)
                        action, hidden, sensors, actions = model(*sensor_inputs(env, impedance, args.seed, padded, step), issued, init, sensors, actions, hidden)
                        issued = action
                    else:
                        obs, history, _, _ = env.get_observations()
                        action = inference(model, obs, history)
                        if reject:
                            env.set_impedance_force_estimates(model.last_prediction.detach())
                if args.gpu.startswith('cuda'):
                    torch.cuda.synchronize(env.device)
                inference_seconds += time.perf_counter()-tick
                adapter.loads(step*env.dt)
                rollout_start = time.perf_counter()
                env.step(action)
                if args.gpu.startswith('cuda'):
                    torch.cuda.synchronize(env.device)
                rollout_seconds += time.perf_counter()-rollout_start
                evaluation_interactions += env.num_envs  # Includes simulated inactive/padded rows, never training credit.
                reasons = adapter.failures()
                physical = env.fail_buf.detach().cpu().numpy() if hasattr(env, 'fail_buf') else np.zeros(env.num_envs, bool)
                ee = (s.ee_pos-env.curr_ee_goal_cart_world).detach().cpu().numpy()
                velocity = (s.base_lin_vel[:, :2]-env.commands[:, :2]).detach().cpu().numpy()
                yaw = (s.base_ang_vel[:, 2]-env.commands[:, 2]).detach().cpu().numpy()
                force, couple, point = s.applied_torso_wrench_source()
                moment = couple+torch.linalg.cross(point-s.base_pos, force)  # One shift to base-link origin, world axes.
                applied = torch.cat((s._ee_force_world, force, moment), -1).detach().cpu().numpy()
                positions = torch.cat((s.ee_pos, s.base_pos), -1).detach().cpu().numpy()
                velocities = torch.cat((s.ee_vel, s._base_world_lin_vel), -1).detach().cpu().numpy()
                bad_contacts = (s.link_contact_forces[:, s.penalized_contact_indices].norm(dim=-1) > .1).sum(-1).cpu().numpy()
                tracking = env.get_facet_tracking() if facet else {}
                for slot, scenario in enumerate(batch):
                    if not alive[slot]:
                        continue
                    failure = reasons[slot] or ('native_physical_failure' if physical[slot] else '')
                    finite = np.isfinite(np.r_[ee[slot], velocity[slot], yaw[slot], applied[slot],
                                                positions[slot], velocities[slot]]).all()
                    failure = failure or ('' if finite else 'nonfinite')
                    row = dict(trial_id=int(trial_ids[slot]), time=(step+1)*env.dt, interval_start=step*env.dt,
                        ee_error=ee[slot], velocity_error=velocity[slot], yaw_error=yaw[slot], applied_wrench=applied[slot],
                        valid=not bool(failure), failed=bool(failure), phase=scenario.at(step*env.dt)[3])
                    row.update(ee_base_position_world=positions[slot], ee_base_velocity_world=velocities[slot],
                        nominal_commands=scenario.at((step+1)*env.dt)[0], effective_commands=env.commands[slot, :3].cpu().numpy(),
                        ee_reference_world=env.curr_ee_goal_cart_world[slot].cpu().numpy(), undesirable_contacts=int(bad_contacts[slot]))
                    if facet:
                        row['facet_errors_squared'] = tracking['errors_squared'][slot].cpu().numpy()
                        row['facet_valid'] = tracking['valid'][slot].cpu().numpy()
                    records[slot].append(row)
                    writer.append({k: np.asarray(v) for k, v in row.items()})
                    if failure or step == steps-1:
                        trials.append(summarize_trial(records[slot], scenario.metadata, steps, env.dt, config, failure))
                        alive[slot] = False
                if not alive.any():
                    break
        writer.close()
        result = dict(task=args.task, evaluation_seed=args.seed, trials=len(trials), failed_trials=sum(t['failed'] for t in trials),
            metrics=aggregate_trials(trials),
            evaluation_interactions=evaluation_interactions, training_interactions=metadata.get('phase_interactions'),
            wall_seconds=time.perf_counter()-start, inference_seconds=inference_seconds,
            rollout_seconds=rollout_seconds,
            inference_robot_steps_per_second=evaluation_interactions/max(inference_seconds, 1e-12),
            collection_robot_steps_per_second=evaluation_interactions/max(rollout_seconds+inference_seconds, 1e-12),
            artifacts=metadata, protocol=config, source_checkpoint=args.checkpoint,
            conventions=dict(timing='Loads held over [interval_start,time]; states/errors at successor time; no reset transitions',
                nominal_velocity='Body-frame planar velocity and body-z angular velocity; original nominal commands',
                positions='World meters; response displacements rotated into initial-heading frame',
                contacts='Native penalized-body contact count above 0.1 N, includes failed endpoint',
                applied_wrench='[EE force N, base force N, base moment Nm] world axes; base moment about base-link origin',
                response='Observed motion, not causal displacement; EE error and base velocity error reported separately',
                overshoot='Signed EE displacement peak beyond observed final 0.25-second plateau mean; not an analytical equilibrium'),
            interpretation='Integration/measurement only; smoke checkpoints are not trained task-performance evidence')
        write_reports(args.output_dir, trials, initial, result)
        print(json.dumps(result, allow_nan=False), flush=True)
    except BaseException:
        # Kit shutdown can mask Python exceptions/exit codes. Persist failure
        # before closing it; callers must also require COMPLETE and valid JSON.
        import traceback
        failure = traceback.format_exc()
        (args.output_dir/'ERROR.txt').write_text(failure)
        print(failure, file=sys.stderr, flush=True)
        raise
    finally:
        writer.close()
        if env is not None:
            env.simulator._app_launcher.app.close()


def write_reports(output_dir, trials, scenarios, summary):
    """Strict summary JSON only; raw transition arrays retain nonfinite values/masks."""
    for name, value in (('trials.json', trials), ('scenarios.json', scenarios), ('summary.json', summary)):
        (Path(output_dir)/name).write_text(json.dumps(value, indent=2, allow_nan=False))
    (Path(output_dir)/'COMPLETE').write_text('FACET evaluation complete\n')


def summarize_trial(rows, scenario, steps, dt, config, reason):
    valid = [r for r in rows if r['valid']]
    result = dict(scenario, failed=bool(reason), termination_reason=reason,
                  survival_seconds=len(rows)*dt, valid_samples=len(valid), prescribed_steps=steps)
    def mean(key, norm=False):
        values = np.asarray([r[key] for r in valid])
        return float(np.linalg.norm(values, axis=-1).mean() if norm else np.abs(values).mean()) if len(values) else None
    result.update(ee_error_m=mean('ee_error', True), velocity_error_m_s=mean('velocity_error', True), yaw_error_rad_s=mean('yaw_error'))
    passing = lambda r: np.linalg.norm(r['ee_error']) <= config['ee_tolerance_m'] and np.linalg.norm(r['velocity_error']) <= config['velocity_tolerance_m_s'] and abs(r['yaw_error']) <= config['yaw_tolerance_rad_s']
    result['S_track'] = sum(passing(r) for r in valid)/steps
    # Physical exposure includes finite failed endpoints, not the unobserved tail.
    # A missing EE measurement does not invalidate a finite torso measurement.
    loads = np.asarray([r['applied_wrench'] for r in rows]).reshape(-1, 3, 3)
    magnitudes = np.linalg.norm(loads, axis=-1)
    known = np.isfinite(magnitudes)
    result['wrench_valid_samples'] = known.sum(0).tolist()  # EE force, base force, base moment.
    result['force_valid_samples'] = known[:, :2].sum(0).tolist()
    result['force_observed_seconds'] = (known[:, :2].sum(0)*dt).tolist()
    result['force_measurements_complete'] = [bool(len(rows) and known[:, i].all()) for i in range(2)]
    observed = [magnitudes[known[:, i], i] for i in range(2)]
    peaks = [float(x.max()) if len(x) else None for x in observed]
    exposure = [float(x.sum()*dt) if len(x) else None for x in observed]
    result['observed_force_peak_N'] = peaks
    result['observed_force_exposure_N_s'] = exposure
    for i, key in enumerate(('force_peak_N', 'base_force_peak_N')):
        result[key] = peaks[i] if result['force_measurements_complete'][i] else None
    result['force_exposure_N_s'] = [x if ok else None for x, ok in zip(exposure, result['force_measurements_complete'])]
    result['undesirable_contact_steps'] = sum(r.get('undesirable_contacts', 0) > 0 for r in rows)
    result['contact_observed_steps'] = len(rows)
    result['release_recovery_seconds'] = None
    result['release_window_complete'] = not reason and len(rows)*dt >= scenario['events'][2]+config['recovery_window_seconds']
    dwell = 0
    for row in valid:
        if row['interval_start'] < scenario['events'][2]:
            continue
        if row['time'] > scenario['events'][2]+config['recovery_window_seconds']+1e-8:
            break
        dwell = dwell+1 if passing(row) else 0
        if dwell*dt >= config['recovery_dwell_seconds']:
            result['release_recovery_seconds'] = row['time']-scenario['events'][2]
            break
    if reason:
        result['release_recovery_seconds'] = None
    for phase in range(4):
        stage = [r for r in valid if r['phase'] == phase]
        result[f'stage{phase}_count'] = len(stage)
        result[f'stage{phase}_ee_bias_xyz_m'] = np.mean([r['ee_error'] for r in stage], axis=0).tolist() if stage else None
        result[f'stage{phase}_base_velocity_bias_xy_m_s'] = np.mean([r['velocity_error'] for r in stage], axis=0).tolist() if stage else None
    # Compare against the last valid unloaded endpoint, never a reset state.
    baseline = [r for r in valid if r['time'] <= scenario['events'][0]+1e-8 and 'ee_base_position_world' in r]
    result['responses'] = []
    if baseline:
        before = baseline[-1]
        angle = scenario['initial_heading']
        rotation = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        for phase in (1, 2, 3):
            segment = [r for r in valid if r['phase'] == phase and 'ee_base_position_world' in r]
            end = scenario['events'][phase] if phase < 3 else scenario['horizon']
            complete = bool(segment and segment[-1]['time'] >= end-1e-8)
            response = dict(phase=phase, valid_samples=len(segment), complete=complete,
                            ee_displacement_xyz_m=None, base_displacement_xyz_m=None, signed_ee_overshoot_m=None)
            if segment:
                displacement = (np.asarray([r['ee_base_position_world'] for r in segment])-before['ee_base_position_world']).reshape(-1, 2, 3) @ rotation.T
                velocity_change = (np.asarray([r['ee_base_velocity_world'] for r in segment])-before['ee_base_velocity_world']).reshape(-1, 2, 3) @ rotation.T
                response.update(ee_displacement_xyz_m=displacement[-1, 0].tolist(), base_displacement_xyz_m=displacement[-1, 1].tolist(),
                    ee_velocity_change_xyz_m_s=velocity_change.mean(0)[0].tolist(), base_velocity_change_xyz_m_s=velocity_change.mean(0)[1].tolist())
                if complete and phase < 3:
                    axis, sign = (1, 1) if scenario['direction'] == 'lateral' else (0, -1)
                    signed = sign*displacement[:, 0, axis]
                    tail = [i for i, r in enumerate(segment) if r['time'] > end-.25]
                    if tail:
                        response['signed_ee_overshoot_m'] = float(max(0., signed.max()-signed[tail].mean()))
            result['responses'].append(response)
    facet = [r for r in valid if 'facet_valid' in r]
    if facet:
        # Invalid virtual-reference diagnostics must not discard finite physical
        # tracking or leak NaN into JSON; retain per-horizon contributing counts.
        masks = [r['facet_valid'] & np.isfinite(r['facet_errors_squared']).all(-1) for r in facet]
        count = np.sum(masks, axis=0)
        sums = np.sum([np.where(mask[:, None], r['facet_errors_squared'], 0) for r, mask in zip(facet, masks)], axis=0)
        result['facet_horizon_counts'] = count.tolist()
        result['facet_horizon_mse'] = [[float(x) for x in row] if n else [None]*6 for row, n in zip(sums/np.maximum(count[:, None], 1), count)]
    return result


def aggregate_trials(trials):
    """One evaluation seed: equal-trial means, not a pooled-successful-step mean."""
    result = {}
    for key in ('ee_error_m', 'velocity_error_m_s', 'yaw_error_rad_s', 'S_track', 'survival_seconds', 'release_recovery_seconds'):
        values = [r[key] for r in trials if r.get(key) is not None and np.isfinite(r[key])]
        result[key] = dict(mean=float(np.mean(values)) if values else None, contributing_trials=len(values))
    total = len(trials)
    result['physical_failure_rate'] = dict(mean=sum(r['failed'] for r in trials)/total if total else None, attempted_trials=total)
    recovered = sum(r['release_recovery_seconds'] is not None for r in trials)
    censored = sum(not r['failed'] and not r['release_window_complete'] and r['release_recovery_seconds'] is None for r in trials)
    result['release_recovered_fraction'] = dict(
        mean=recovered/total if total and not censored else None,
        observed_lower_bound=recovered/total if total else None, censored_unrecovered=censored,
        planned_release_opportunities=total,
        complete_windows=sum(r['release_window_complete'] for r in trials))
    return result


if __name__ == '__main__':
    main()
