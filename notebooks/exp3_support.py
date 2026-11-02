"""CPU-only recorded action/support analysis; no simulator or policy imports."""
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import time
import inspect

import numpy as np
import pandas as pd

import exp1_analysis as shared
import exp2_analysis as exp2
from b1z1_paper_names import DISPLAY_NAMES
from analysis_runtime import NumericCache
import exp3_contact as contact

TASKS = ("b1z1_pact", "b1z1_unifp_original", "b1z1_ppo_pos",
         "b1z1_pact_ab6_position_film", "b1z1_pact_ab8_coupled_film", "b1z1_pact_ab9_representation")


def passive_bias(row):
    """Joint bias [Nm] at t; this is NOT the already executed explicit PD torque."""
    qd = np.asarray(row["joint_velocity_t"], dtype=float)
    damping = np.broadcast_to(np.asarray(row.get("mechanics_joint_damping", 0)), qd.shape)
    if not np.isfinite(damping).all() or np.any(damping < 0):
        raise ValueError("Invalid recorded passive damping")
    assumptions = ["Frozen-state Euler with interval-average executed torque/loads; passive drive target velocity assumed zero."]
    if np.any(np.asarray(row.get("mechanics_joint_friction", 0)) != 0):
        assumptions.append("Recorded joint friction coefficient retained in provenance; unknown constraint-dependent friction torque omitted, NOT coefficient*sign(velocity).")
    if np.any(np.asarray(row.get("mechanics_joint_stiffness", 0)) != 0):
        assumptions.append("Passive drive stiffness torque omitted: simulator drive target not recorded.")
    return damping * qd, assumptions


def hierarchy(frame, keys):
    """Equal trials -> evaluation seeds -> checkpoints; never timestep SD."""
    if frame.empty:
        return pd.DataFrame()
    frame = frame[np.isfinite(frame.value)].copy()
    trial_keys = keys + ["checkpoint", "seed", "trial"]
    trials = frame.groupby(trial_keys, dropna=False).value.mean().reset_index()
    seeds = trials.groupby(keys + ["checkpoint", "seed"], dropna=False).value.mean().reset_index()
    # Metadata is invariant across metric/time groups. Do not rescan the entire
    # frame once per group (formerly quadratic in the number of time bins).
    known = frame.groupby('checkpoint').training_seed.first() if 'training_seed' in frame else pd.Series(dtype=float)
    checkpoints = seeds.groupby(keys+['checkpoint'], dropna=False).value.mean().reset_index()
    checkpoints['training_seed'] = checkpoints.checkpoint.map(known)
    grouped = checkpoints.groupby(keys, dropna=False)
    out = grouped.agg(mean=('value','mean'), sd=('value','std'), checkpoints=('value','size'),
                      known=('training_seed','count'), independent=('training_seed','nunique'))
    seed_stats = seeds.groupby(keys, dropna=False).value.agg(['std','size'])
    out['sd'] = out.sd.where(out.checkpoints > 1, seed_stats['std'])
    out['contributing_seeds'] = out['evaluation_repetitions'] = seed_stats['size']
    out['variability_replicates'] = np.where(out.checkpoints > 1, out.checkpoints, seed_stats['size'])
    out['variability'] = np.where(out.checkpoints == 1, 'evaluation_seed_sd',
        np.where((out.known == out.checkpoints) & (out.independent == out.checkpoints),
                 'training_seed_sd', 'checkpoint_sd_independence_unverified'))
    return out.drop(columns=['known','independent']).reset_index()


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


class RecordedFK:
    """URDF tree FK in metres, using recorded joint names, not assumed arm indices."""
    def __init__(self, filename, base, ee):
        self.filename, self.base, self.ee, self._tree = filename, base, ee, None
        root = ET.parse(filename).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        links = {link.get("name") for link in root.findall("link")}
        def chain(link):
            result = []
            if links and link not in links:
                raise ValueError(f"Unknown URDF link: {link}")
            while link in joints:
                if len(result) > len(joints):
                    raise ValueError("Cyclic URDF chain")
                joint = joints[link]
                result.append(joint)
                link = joint.find("parent").get("link")
            return link, result[::-1]
        ee_root, self.chain = chain(ee)
        base_root, self.base_chain = chain(base)
        if ee_root != base_root:
            raise ValueError("Disconnected base/EE URDF trees")

    def __call__(self, q, names):
        rotation, position = self.transform(q, names, self.chain)
        base_rotation, base_position = self.transform(q, names, self.base_chain)
        # B1's trunk and arm mounting link can be fixed siblings of the URDF root.
        return base_rotation.T @ (position-base_position)

    def batch(self, q, names):
        if self._tree is None or self._tree.names != tuple(names):
            self._tree = contact.KinematicTree(self.filename, names)
        p, r, _, _ = self._tree.states(q)
        b, e = self._tree.links.index(self.base), self._tree.links.index(self.ee)
        return np.einsum('nji,nj->ni', r[:, b], p[:, e]-p[:, b])

    def transform(self, q, names, chain):
        positions = dict(zip(names, q))
        rotation, position = np.eye(3), np.zeros(3)
        for joint in chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            position += rotation @ xyz
            rotation = rotation @ axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1]) @ axis_rotation([1, 0, 0], rpy[0])
            if joint.get("type") != "fixed":
                name = joint.get("name")
                if name not in positions:
                    raise ValueError(f"Missing commanded joint {name}")
                axis = joint.find("axis")
                axis = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
                if joint.get("type") == "prismatic":
                    position += rotation @ (axis * positions[name])
                else:
                    rotation = rotation @ axis_rotation(axis, positions[name])
        return rotation, position


def channel_values(row, names):
    """Return joint-mean squared usage; square root only AFTER trial averaging."""
    values = {}
    for region in ("arm", "leg"):
        ids = [i for i, name in enumerate(names) if name.startswith("z1_") == (region == "arm")]
        if not ids:
            continue
        for channel in ("pd_weighted_motor", "ff_weighted_motor", "executed_torque"):
            key = channel + "_mean_square"
            if key in row:
                values[region + "_" + channel + "_ms"] = float(np.mean(np.asarray(row[key])[ids]))
            mean_key = channel + "_interval_mean"
            if mean_key in row:
                values[region + "_" + channel + "_mean_nm"] = float(np.mean(np.asarray(row[mean_key])[ids]))
        if "opposing_channel_cancellation_mean" in row:
            values[region + "_cancellation_nm"] = float(np.mean(np.asarray(row["opposing_channel_cancellation_mean"])[ids]))
    return values


def support_values(row, feet):
    """Use recorded mean cross-products about simultaneous whole-body CoM."""
    values = {}
    if "grf_raw_world_interval_mean" in row:
        grf = np.asarray(row["grf_raw_world_interval_mean"])
        for i, foot in enumerate(feet):
            values[f"foot_{foot}_fz_n"] = grf[i, 2]
        values.update({f"foot_support_f{a}_n": v for a, v in zip("xyz", grf.sum(0))})
    if "body_contact_forces_raw_world_interval_mean" in row:
        values.update({f"all_support_f{a}_n": v for a, v in zip("xyz", np.asarray(row["body_contact_forces_raw_world_interval_mean"]).sum(0))})
    for field, label in (("support_moment_world", "foot_support"),
                         ("all_contact_moment_world_interval_mean", "all_support")):
        if field in row:
            values.update({f"{label}_m{a}_nm": v for a, v in zip("xyz", row[field])})
    return values


def _reduce(acc, values, indices):
    """Wide grouped sums/counts: no Python transition x metric dictionary loop."""
    if not values: return acc
    frame = pd.DataFrame(values, index=pd.MultiIndex.from_arrays(indices),dtype=np.float64)
    frame = frame.where(np.isfinite(frame))
    group = list(range(len(indices)))
    pieces = (frame.groupby(level=group).sum(), frame.groupby(level=group).count(),
              frame.pow(2).groupby(level=group).sum())
    return pieces if acc is None else tuple(a.add(b, fill_value=0) for a,b in zip(acc,pieces))


def _long(acc, keys):
    if acc is None: return pd.DataFrame()
    total, count, squares = acc
    # Stack only compact reductions, never full transition records.
    frame = pd.concat([total.stack().rename('sum'), count.stack().rename('count'),
                       squares.stack().rename('sum_squares')], axis=1)
    frame = frame[frame['count'] > 0]
    frame['value'] = frame['sum']/frame['count']
    frame.index.names = keys+['metric']
    return frame.reset_index()


def _values_batch(data, ids, feet):
    values = {}
    for region, joints in ids.items():
        if not len(joints): continue
        for channel in ('pd_weighted_motor','ff_weighted_motor','executed_torque'):
            for suffix, label in (('_mean_square','_ms'),('_interval_mean','_mean_nm')):
                if channel+suffix in data:
                    # Contiguous joint axis retains the legacy per-row float32
                    # summation order before grouped float64 accumulation.
                    values[region+'_'+channel+label] = np.ascontiguousarray(data[channel+suffix][:,joints]).mean(1)
        if 'opposing_channel_cancellation_mean' in data:
            values[region+'_cancellation_nm'] = np.ascontiguousarray(data['opposing_channel_cancellation_mean'][:,joints]).mean(1)
    if 'grf_raw_world_interval_mean' in data:
        force = data['grf_raw_world_interval_mean']
        values.update({f'foot_{foot}_fz_n':force[:,i,2] for i,foot in enumerate(feet)})
        values.update({f'foot_support_f{axis}_n':force.sum(1)[:,i] for i,axis in enumerate('xyz')})
        for group, prefixes in (('front',('FR','FL')),('rear',('RR','RL'))):
            selected = [i for i,f in enumerate(feet) if f.startswith(prefixes)]
            if selected: values[group+'_normal_support_n'] = force[:,selected,2].sum(1)
    if 'body_contact_forces_raw_world_interval_mean' in data:
        force = data['body_contact_forces_raw_world_interval_mean'].sum(1)
        values.update({f'all_support_f{axis}_n':force[:,i] for i,axis in enumerate('xyz')})
    for field,label in (('support_moment_world','foot_support'),('all_contact_moment_world_interval_mean','all_support')):
        if field in data: values.update({f'{label}_m{axis}_nm':data[field][:,i] for i,axis in enumerate('xyz')})
    values['ee_error_m'] = np.linalg.norm(data['ee_position_world']-data['ee_reference_world'],axis=1)
    values['velocity_error_m_s'] = np.linalg.norm(data['base_velocity_body'][:,:2]-data['command_nominal'][:,:2],axis=1)
    return values


def _recording_statistics(run, config):
    from exp3_representation import chunks, file_hash, rotate
    meta, path = run['meta'], run['path']; env = meta['resolved_environment']; started = time.perf_counter()
    fields = set(contact.FIELDS) | {
        'task_stage_successor','ee_position_world','ee_reference_world','ee_reference_t_world',
        'base_position_t','base_quaternion_t','position_target_interval_mean','base_velocity_body','command_nominal',
        'support_moment_world','all_contact_moment_world_interval_mean','opposing_channel_cancellation_mean',
        'pd_weighted_motor_interval_mean','ff_weighted_motor_interval_mean','executed_torque_interval_mean',
        'torque_request_motor_scaled_interval_mean','pd_weighted_motor_mean_square',
        'ff_weighted_motor_mean_square','executed_torque_mean_square'}
    fields = sorted(fields & set(run['records']['fields']))
    diagnostics = [f'{path}: unavailable {k}' for k in
        ('position_target_interval_mean','pd_weighted_motor_mean_square','ff_weighted_motor_mean_square',
         'executed_torque_mean_square','opposing_channel_cancellation_mean') if k not in fields]
    fk = None
    try:
        model = config.get('urdf') or meta['urdf_source']
        if file_hash(model) != meta['urdf_sha256']: raise ValueError('URDF hash mismatch')
        fk = RecordedFK(model, env['asset']['base_name'], env['asset']['gripper_name'])
    except (KeyError,OSError,ValueError) as exc: diagnostics.append(f'{path} commanded FK unavailable: {exc}')
    names = meta['dof_names']; feet = env['asset']['foot_name']
    ids = {r:np.array([i for i,n in enumerate(names) if n.startswith('z1_') == (r=='arm')],int) for r in ('arm','leg')}
    momentum = contact.MomentumContact(meta, contact.initial_states(path), config)
    if config.get('contact_moment_reconstruction',False):
        diagnostics.append(f'{path}: total contact moment unavailable: complete interval external angular impulse about the moving whole-body CoM is not verified; normal-only sensor moments retained')
    trial_acc = curve_acc = None
    last = {}; files = []; missing = Counter()
    timers = dict(kinematics_seconds=0., reduction_seconds=0.,reference_write_seconds=0.)
    for _, data in chunks(run, fields):
        t0 = time.perf_counter()
        # Validate ordering once per trial/chunk; interleaved vector environments are legal.
        for tid in np.unique(data['trial_id']):
            steps = data['step'][data['trial_id']==tid]
            if tid not in run['trials'] or steps[0] <= last.get(tid,-1) or np.any(np.diff(steps)<=0):
                raise ValueError('Unknown/duplicate/out-of-order transition')
            last[tid] = steps[-1]
        physical = momentum.process(data)
        write_start = time.perf_counter()
        files.append(contact.save_reference_chunk(run['_contact_dir'],len(files),{**physical, **{k:data[k] for k in
            ('trial_id','step','time_seconds','successor_time_seconds','task_stage_interval')}}))
        timers['reference_write_seconds'] += time.perf_counter()-write_start
        good = contact.valid_rows(data)
        d = {k:v[good] for k,v in data.items()}
        phys = {k:v[good] for k,v in physical.items()}
        if not good.any(): continue
        accounting = ('pd_weighted_motor_interval_mean','ff_weighted_motor_interval_mean','torque_request_motor_scaled_interval_mean')
        if all(k in d for k in accounting) and not np.allclose(d[accounting[0]]+d[accounting[1]],d[accounting[2]],atol=1e-4,rtol=1e-5):
            raise ValueError(f'Recorded requested PD+FF accounting mismatch: {path}')
        values = _values_batch(d, ids, feet)
        heading = np.array([run['trials'][int(t)].get('initial_task_heading_rad',np.nan) for t in d['trial_id']])
        direction = (np.tile([0.,0.,1.],(len(heading),1)) if meta['arguments']['profile']=='carrying'
                     else np.column_stack((np.cos(heading),np.sin(heading),np.zeros_like(heading))))
        if 'ee_force_applied_world_interval_mean' in d:
            values['load_n'] = (direction*d['ee_force_applied_world_interval_mean']).sum(1)
        force = phys['momentum_contact_world']
        values.update({f'momentum_contact_f{a}_n':force[:,i] for i,a in enumerate('xyz')})
        values['momentum_contact_interaction_n'] = (direction*force).sum(1)
        if contact.flat_ground(meta) and 'foot_support_fz_n' in values:
            values['momentum_minus_normal_fz_n'] = np.where(phys['momentum_foot_eligible'],
                force[:,2]-values['foot_support_fz_n'],np.nan)
        if fk and all(k in d for k in ('position_target_interval_mean','base_position_t','base_quaternion_t','ee_reference_t_world')):
            command = fk.batch(d['position_target_interval_mean'],names)
            world = d['base_position_t']+rotate(command,d['base_quaternion_t'])
            values['command_geometry_error_m'] = np.linalg.norm(world-d['ee_reference_t_world'],axis=1)
        timers['kinematics_seconds'] += time.perf_counter()-t0
        t0 = time.perf_counter()
        for name,value in values.items(): missing[name] += int((~np.isfinite(value)).sum())
        for successor in (False,True):
            subset = {k:v for k,v in values.items() if (k in ('ee_error_m','velocity_error_m_s'))==successor}
            stage = d['task_stage_successor' if successor else 'task_stage_interval']
            timestamp = d['successor_time_seconds' if successor else 'time_seconds']
            trial_acc = _reduce(trial_acc,subset,[d['trial_id'],stage])
            curve_acc = _reduce(curve_acc,subset,[d['step'],timestamp])
        timers['reduction_seconds'] += time.perf_counter()-t0
    scalar, curves = _long(trial_acc,['trial','stage']), _long(curve_acc,['step','time_seconds'])
    if not scalar.empty:
        ms = scalar.metric.str.endswith('_ms')
        scalar.loc[ms,'value'] = np.sqrt(scalar.loc[ms,'value'])
        scalar.loc[ms,'metric'] = scalar.loc[ms,'metric'].str[:-3]+'_rms_nm'
    timers.update(read_seconds=run.get('_chunk_read_seconds',0.),wall_seconds=time.perf_counter()-started)
    diagnostics += [f'{path}: {n}' for n in sorted(momentum.notes)]
    diagnostics += [f'{path}: {k} unavailable/nonfinite on {v} valid rows' for k,v in missing.items() if v]
    return dict(scalar=scalar.to_dict('list'),curves=curves.to_dict('list'),contact_files=files,
                timings=timers,diagnostics=diagnostics,eligibility=dict(momentum.counts),missing_counts=dict(missing))



def analyze(config):
    """Stream numeric chunks, retain compact per-seed sufficient statistics only."""
    from exp3_representation import recording_signature, file_hash
    started = time.perf_counter()
    signature = shared.fingerprint([inspect.getsource(f) for f in
        (_recording_statistics,_values_batch,_reduce,_long,RecordedFK)]+[file_hash(Path(contact.__file__)),np.__version__,pd.__version__])
    cache = NumericCache(config, signature=signature)  # Plot/layout edits do not reread trajectories.
    timings = dict(runs=[], aggregation_seconds=0., export_seconds=0., plotting_seconds=0.)
    output = Path(config["output_dir"]) / "action_support"
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE").unlink(missing_ok=True)
    diagnostics, provenance, scalar_frames, curve_frames = [], [], [], []
    identities, protocols, gains = set(), set(), {}
    method_specs = config.get('support_methods')
    selected = method_specs if method_specs is not None else config.get("exp2_tasks", TASKS)
    for path in shared.discover(config.get("exp2_roots", []), "metadata.json"):
        if not (path.parent / "COMPLETE").is_file():
            continue
        meta = shared.read_json(path)
        if meta.get("experiment") != 2:
            continue
        # Numerical caches remain recording-based; aliases only label reductions.
        matched = ([name for name,spec in method_specs.items() if spec['task']==meta['task']
                    and spec['checkpoint_sha256']==meta['checkpoint_sha256']] if method_specs is not None
                   else [meta['task']] if meta['task'] in selected else [])
        if not matched: continue
        if len(matched)!=1: raise ValueError('Recording matches multiple support method IDs')
        seed, args = meta["evaluation_seed"], meta["arguments"]
        if seed not in config["evaluation_seeds"] or args["loading"] not in config.get("exp2_loading", ["loaded"]):
            continue
        task, checkpoint, profile, loading = matched[0], meta["checkpoint_sha256"], args["profile"], args["loading"]
        identity = (task, checkpoint, seed, profile, loading)
        if identity in identities:
            raise ValueError(f"Duplicate Exp2 cohort: {identity}")
        identities.add(identity)
        protocols.add(shared.fingerprint(exp2.protocol(meta)))
        if len(protocols) > 1:
            raise ValueError("Incompatible Exp2 protocols; narrow exp2_roots")
        env = meta["resolved_environment"]
        gains[task] = {k: env["control"].get(k) for k in ("stiffness", "damping")}
        trials = {r["trial_id"]: r for r in shared.read_json(path.parent / "trials.json")}
        records = shared.read_json(path.parent / "records.json")
        run = dict(path=path.parent, records=records, meta=meta, trials=trials,
                   _num_workers=config.get('num_workers',1))
        if config.get('progress',False): print(f'Exp3 support: {path.parent}',flush=True)
        run_started = time.perf_counter()
        options = {k:config.get(k) for k in ('contact_window_seconds','contact_nonfoot_threshold_n','contact_moment_reconstruction')}
        cache_key, numerical = cache.lookup('support_recordings',(
            recording_signature(run),contact.model_signature(meta,config.get('urdf')),options))
        run['_contact_dir'] = (cache_key.with_suffix('') if cache_key is not None else
                               output/'contact_intervals'/shared.fingerprint(str(path.parent))[:16])
        if numerical is not None and not all(Path(f).is_file() for f in numerical['contact_files']):
            numerical = None; cache.hits -= 1; cache.misses += 1
        hit = numerical is not None
        if numerical is None:
            numerical = _recording_statistics(run,config)
            cache.save(cache_key,numerical)
        diagnostics.extend(numerical['diagnostics'])
        timings['runs'].append(dict(path=str(path.parent),cached=hit,
                                   source_numerical_timings=numerical['timings'],
                                   wall_seconds=time.perf_counter()-run_started,
                                   eligibility=numerical['eligibility']))
        identity_values = dict(task=task,checkpoint=checkpoint,seed=seed,profile=profile,loading=loading,
                               training_seed=meta.get('checkpoint_training_seed'))
        scalar_frames.append(pd.DataFrame(numerical['scalar']).assign(**identity_values))
        curve_frames.append(pd.DataFrame(numerical['curves']).assign(**identity_values,trial=0))
        # Compact numeric references, no duplicated observations/features/trajectories.
        t0 = time.perf_counter()
        contact.export_references(output,path.parent,numerical['contact_files'],dict(**identity_values,
            eligibility=numerical['eligibility'],missing_counts=numerical['missing_counts'],notes=numerical['diagnostics'],window_seconds=config.get('contact_window_seconds',.1),
            nonfoot_threshold_n=config.get('contact_nonfoot_threshold_n',1e-6),urdf_sha256=meta.get('urdf_sha256')))
        timings['export_seconds'] += time.perf_counter()-t0
        provenance.append(dict(path=str(path.parent), task=task, checkpoint=checkpoint,
            training_seed=meta.get("checkpoint_training_seed"), evaluation_seed=seed,
            profile=profile, loading=loading, evaluation=meta["evaluation"], dt=meta["dt"],
            control=env["control"], domain_randomization=env.get("domain_rand"), urdf=meta.get("urdf_sha256")))
    seen = {r["task"] for r in provenance}
    for task in seen:
        hashes = {r["checkpoint"] for r in provenance if r["task"] == task}
        if len(hashes) > 1:
            diagnostics.append(f"{task}: multiple checkpoint hashes; independence of training runs is not established by hashes alone")
    diagnostics.extend(f"Missing Exp2 method: {task}" for task in selected if task not in seen)
    for task in seen:
        for profile in exp2.PROFILES:
            for loading in config.get("exp2_loading", ["loaded"]):
                seeds = {r["evaluation_seed"] for r in provenance if r["task"] == task and r["profile"] == profile and r["loading"] == loading}
                if seeds != set(config["evaluation_seeds"]):
                    diagnostics.append(f"{task}/{profile}/{loading}: evaluation seeds {sorted(seeds)}, expected {config['evaluation_seeds']}")
    pairs = config.get('support_pairs',[(TASKS[3], TASKS[4]), (TASKS[4], TASKS[0]), (TASKS[5], TASKS[0])])
    comparisons = [dict(left=a, right=b, available=a in seen and b in seen,
        gains_match=gains.get(a) == gains.get(b) if a in gains and b in gains else None) for a, b in pairs]
    if 'support_pairs' not in config and comparisons[0]["available"] and not comparisons[0]["gains_match"]:
        diagnostics.append("A1 versus A3 gains differ: NOT a matched-gain attribution comparison")
    t0 = time.perf_counter()
    scalar = pd.concat(scalar_frames,ignore_index=True) if scalar_frames else pd.DataFrame()
    curves = pd.concat(curve_frames,ignore_index=True) if curve_frames else pd.DataFrame()
    summary = hierarchy(scalar, ["task", "profile", "loading", "stage", "metric"])
    curve_summary = hierarchy(curves, ["task", "profile", "loading", "metric", "step", "time_seconds"])
    if not curves.empty:
        counts = curves.groupby(["task", "profile", "loading", "metric", "step", "time_seconds"])["count"].sum().reset_index(name="surviving_trials")
        curve_summary = curve_summary.merge(counts)
    timings['aggregation_seconds'] = time.perf_counter()-t0
    t0 = time.perf_counter()
    scalar.to_csv(output/"trial_stages.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    curves.to_csv(output/"per_seed_curves.csv", index=False)
    curve_summary.to_csv(output/"curves.csv", index=False)
    if 'metric' in summary:
        consistency = summary[summary.metric=='momentum_minus_normal_fz_n']
        consistency.to_csv(output/'vertical_support_consistency.csv',index=False)
        maximum = consistency['mean'].abs().max()
        if maximum > config.get('contact_vertical_consistency_warning_n',5.):
            diagnostics.append(f'Vertical normal-support consistency: largest absolute stage-mean residual {maximum:.6g} N. '
                'Momentum reference assumes complete, correctly framed recorded loads; discrepancy retained, not calibrated away.')
    if not scalar.empty:
        seed_stages = scalar.groupby(['task','checkpoint','seed','profile','loading','stage','metric'],dropna=False).agg(
            value=('value','mean'),trials=('trial','size'),samples=('count','sum')).reset_index()
        seed_stages.to_csv(output/'per_seed_stages.csv',index=False)
    result = dict(schema_version=2, experiment=3, section="action_support", status="complete", provenance=provenance,
        diagnostics=sorted(set(diagnostics)), comparisons=comparisons,
        configuration=config, timings=timings, units=dict(load="N", support_force="N", support_moment="Nm", torque_mean="Nm",
            torque_ms="Nm^2", torque_rms="Nm", cancellation="Nm", geometry="m", tracking="m", velocity="m/s", time="s"),
        summary=summary.to_dict("records"), conventions={
            "timing": "Loads/channels/support: interval start. Tracking: successor. No failure interpolation.",
            "geometry": "URDF FK(mean executed position target) at measured base_t versus nominal reference_t; approximate under substep delay/base motion.",
            "support": "Recorded sensor forces/moments are NORMAL-ONLY, not full GRFs. Momentum-Derived Net Contact Force uses whole-CoM endpoint momentum minus gravity and integrated known loads, world N. Foot attribution requires excluded non-foot contacts.",
            "contact_windows": "Non-overlapping complete windows, no gaps/failures/load/stage crossings; timestamps are interval start/end. Partial final windows unavailable.",
            "moments": "Recorded normal-force cross products retained unchanged; total contact moment reconstruction is unavailable (complete external angular impulse not recorded).",
            "reference_assumptions": "Recorded applied world loads must be complete and frame-correct; vertical normal-support consistency is exported, not enforced or calibrated away.",
            "channels": "Weighted motor-scaled requested PD/FF; executed clipped torque separately. RMS from substep squares; cancellation from recorded absolute substep values.",
            "bands": "Sample SD across checkpoint means if multiple checkpoints (training independence NOT inferred from hashes); otherwise evaluation-seed SD. Survivor counts exported/plotted.",
            "settling": "Not reported without an explicitly justified steady support reference; locomotion support is time-varying."})
    timings['export_seconds'] += time.perf_counter()-t0
    t0 = time.perf_counter()
    if config.get('support_plots',True):
        if config.get('support_legacy_figures',True): plot_responses(curve_summary, provenance, output, config)
        plot_compact(curve_summary,provenance,output,config)
    timings.update(plotting_seconds=time.perf_counter()-t0,total_seconds=time.perf_counter()-started,
                   cache_hits=cache.hits,cache_misses=cache.misses)
    shared.write_json(output/'timings.json',timings)
    shared.write_json(output/"summary.json", result)
    (output/"COMPLETE").write_text("Offline action/support analysis complete\n")
    return shared.clean(result)


def plot_responses(frame, provenance, output, config):
    if frame.empty:
        return
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    main = config.get('support_plot_methods',TASKS[:3])
    names = {**DISPLAY_NAMES,**config.get('display_names',{})}
    colors = {**dict(zip(TASKS, ("#2277AA", "#CC6677", "#228833", "#AA3377", "#EE7733", "#666666"))),
              **config.get('support_colors',{})}
    channel = ("load_n", "command_geometry_error_m", "ee_error_m", "arm_pd_weighted_motor_mean_nm",
               "arm_ff_weighted_motor_mean_nm", "arm_executed_torque_mean_nm", "leg_pd_weighted_motor_mean_nm",
               "leg_ff_weighted_motor_mean_nm", "leg_executed_torque_mean_nm", "arm_pd_weighted_motor_ms",
               "arm_ff_weighted_motor_ms", "leg_pd_weighted_motor_ms", "leg_ff_weighted_motor_ms")
    support = ("load_n", "all_support_fx_n", "all_support_fy_n", "all_support_fz_n",
               "all_support_mx_nm", "all_support_my_nm", "all_support_mz_nm", "ee_error_m", "velocity_error_m_s")
    feet = sorted(set(frame.metric[frame.metric.str.startswith("foot_") & frame.metric.str.endswith("_fz_n")]) - {"foot_support_fz_n"})
    support += tuple(feet)
    groups = [("channels", main, channel), ("support", main, support)]
    if 'support_pairs' not in config:
        groups.append(("support_A6_PILOT", (TASKS[5], TASKS[0]), support))
    groups += [(f"channels_{names[a]}_{names[b]}", (a, b), channel)
               for a, b in config.get('support_pairs',((TASKS[3], TASKS[4]), (TASKS[4], TASKS[0]), (TASKS[5], TASKS[0])))]
    for loading in sorted(frame.loading.unique()):
        for name, methods, metrics in groups:
            metrics = [m for m in metrics if m in set(frame.metric)]
            if not metrics or not set(methods) & set(frame.task):
                continue
            fig, axes = plt.subplots(len(metrics)+1, 3, squeeze=False,
                figsize=(config.get("figure_width", 12), 1.65*(len(metrics)+1)), sharex="col", sharey="row")
            for col, profile in enumerate(exp2.PROFILES):
                for j, task in enumerate(methods):
                    color = colors[task]
                    for row, metric in enumerate(metrics):
                        d = frame[(frame.task == task) & (frame.profile == profile) & (frame.loading == loading) & (frame.metric == metric)].sort_values("time_seconds")
                        if d.empty:
                            continue
                        # Missing/invalid time points break lines; never bridge a gap.
                        dt = next(p["dt"] for p in provenance if p["task"] == task and p["profile"] == profile and p["loading"] == loading)
                        offset = float(d.time_seconds.iloc[0] - d.step.iloc[0]*dt)
                        d = d.set_index("step").reindex(range(int(d.step.min()), int(d.step.max())+1))
                        d["time_seconds"] = d.index*dt + offset
                        # MS curves remain (Nm)^2; do not average per-step RMS as trial RMS.
                        ax = axes[row, col]
                        x, y, sd = d.time_seconds.to_numpy(float), d["mean"].to_numpy(float), d.sd.to_numpy(dtype=float, na_value=np.nan)
                        ax.plot(x, y, color=color, label=names[task])
                        ax.fill_between(x, y-sd, y+sd, color=color, alpha=.15)
                        if metric == "ee_error_m":
                            axes[-1, col].plot(x, d.surviving_trials, color=color)
                schedules = [p["evaluation"] for p in provenance if p["profile"] == profile and p["loading"] == loading]
                if schedules:
                    c = schedules[0]
                    for ax in axes[:, col]:
                        for t in [c["move_start_s"], *c["load_times_s"], c["stop_s"]]:
                            ax.axvline(t, color="0.6", lw=.5, ls="--")
                axes[0, col].set_title(profile.title())
                axes[-1, col].set_xlabel("Sequence time [s]; dashed event boundaries")
            for row, metric in enumerate(metrics):
                label = metric.replace('_', ' ')
                if 'support' in metric or metric.startswith('foot_'): label = 'Normal-only sensor: '+label
                if metric.endswith('_ms'): label += ' [(Nm)^2]'
                axes[row, 0].set_ylabel(label, fontsize=8)
            axes[-1, 0].set_ylabel("Surviving trials")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=len(labels))
            shared.style_figure(fig, config)
            fig.tight_layout(rect=(0, 0, 1, .97))
            for suffix in ("pdf", "png"):
                fig.savefig(output/f"{name}_{loading}.{suffix}", dpi=160)
            plt.close(fig)


def plot_compact(frame, provenance, output, config):
    """Reference provenance stays visible; no force-balance superiority claims."""
    if frame.empty: return
    import matplotlib.pyplot as plt
    colors = {**dict(zip(TASKS, ('#2277AA','#CC6677','#228833','#AA3377','#EE7733','#666666'))),
              **config.get('support_colors',{})}
    names = {**DISPLAY_NAMES,**config.get('display_names',{})}
    rows = [(['load_n','momentum_contact_interaction_n'], 'Interaction force [N]'),
            (['front_normal_support_n','rear_normal_support_n'], 'Normal support [N]'),
            (['arm_pd_weighted_motor_ms','arm_ff_weighted_motor_ms'], 'Arm usage [(Nm)^2]'),
            (['leg_pd_weighted_motor_ms','leg_ff_weighted_motor_ms'], 'Leg usage [(Nm)^2]'),
            (['ee_error_m','command_geometry_error_m'], 'EE error [m]'), (['velocity_error_m_s'], 'Planar velocity error [m/s]')]
    groups = [('contact_response',config.get('support_plot_methods',TASKS[:3]))]
    if 'support_pairs' not in config: groups.append(('contact_response_A6_PILOT',(TASKS[5],TASKS[0])))
    for name, methods in groups:
        if not set(methods)&set(frame.task): continue
        for loading in sorted(frame.loading.unique()):
            fig,axes = plt.subplots(len(rows)+1,3,figsize=(config.get('figure_width',12),11),sharex='col',squeeze=False)
            # Carrying includes weight support; sharing its force range would hide
            # horizontal reactions. Other row scales and pushing/pulling match.
            axes[0,1].sharey(axes[0,0])
            axes[0,1].tick_params(labelleft=False)
            for row in range(1,len(rows)+1):
                for col in (1,2):
                    axes[row,col].sharey(axes[row,0]); axes[row,col].tick_params(labelleft=False)
            for col,profile in enumerate(exp2.PROFILES):
                subset=frame[(frame.profile==profile)&(frame.loading==loading)]
                for row,(metrics,label) in enumerate(rows):
                    for task in methods:
                        for j,metric in enumerate(metrics):
                            d=subset[(subset.task==task)&(subset.metric==metric)].sort_values('step')
                            if d.empty: continue
                            # Explicit NaN breaks retain actual sample times and gaps.
                            x=d.time_seconds.to_numpy(float); y=d['mean'].to_numpy(float)
                            sd=d.sd.to_numpy(float); gaps=np.flatnonzero(np.diff(d.step)>1)+1
                            x=np.insert(x,gaps,np.nan); y=np.insert(y,gaps,np.nan); sd=np.insert(sd,gaps,np.nan)
                            ax=axes[row,col]
                            ax.plot(x,y,color=colors[task],ls='-' if j==0 else '--',label=names[task])
                            ax.fill_between(x,y-sd,y+sd,color=colors[task],alpha=.12)
                            if metric=='ee_error_m': axes[-1,col].plot(d.time_seconds,d.surviving_trials,color=colors[task])
                    if col==0: axes[row,col].set_ylabel(label)
                schedules=[p['evaluation'] for p in provenance if p['profile']==profile and p['loading']==loading]
                if schedules:
                    c=schedules[0]
                    for ax in axes[:,col]:
                        for t in [c['move_start_s'],*c['load_times_s'],c['stop_s']]: ax.axvline(t,c='0.65',lw=.5,ls=':')
                axes[0,col].set_title(profile.title()+(' (world z)' if profile=='carrying' else ' (initial-heading x)'))
                axes[-1,col].set_xlabel('Sequence time [s]')
            axes[-1,0].set_ylabel('Surviving trials')
            handles=[plt.Line2D([],[],color=colors[t],label=names[t]) for t in methods if t in set(frame.task)]
            fig.legend(handles=handles,loc='upper center',ncol=len(handles))
            fig.text(.5,.005,'Solid / dashed: applied EE / momentum contact; front / rear normal support; PD / FF usage; measured / commanded-geometry EE error.',ha='center',fontsize=config.get('annotation_font_size',8))
            shared.style_figure(fig,config); fig.tight_layout(rect=(0,.03,1,.96))
            for suffix in ('pdf','png'): fig.savefig(output/f'{name}_{loading}.{suffix}',dpi=160)
            plt.close(fig)
