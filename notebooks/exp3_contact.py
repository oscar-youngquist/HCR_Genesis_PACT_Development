"""Recorded momentum balance, CPU-only kinematics, and explicit eligibility.

No decoder/torso velocity substitutes for whole-body CoM velocity. Endpoint t+1
minus the preceding endpoint t uses THIS row's applied interval loads, not the
preceding row's loads. This reference is dynamics consistency, not contact sensing.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import os
import shutil
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

FIELDS = {
    'trial_id', 'step', 'time_seconds', 'successor_time_seconds', 'valid', 'alive_for_metrics',
    'physical_failure', 'artificial_reset', 'phase', 'task_stage_interval',
    'robot_com_velocity_world', 'body_com_velocities_world', 'body_contacts_interval_mean',
    'body_contact_forces_raw_world_interval_mean', 'nonfoot_contact',
    'ee_force_applied_world_interval_mean', 'base_force_applied_world_interval_mean',
    'ee_force_world', 'base_force_world', 'base_moment_world',
    'base_quaternion_xyzw', 'base_velocity_world', 'base_angular_velocity_world',
    'joint_position', 'joint_velocity', 'grf_raw_world_interval_mean', 'surface_normals_world',
}
OUTPUT_FIELDS = ('momentum_contact_world', 'momentum_foot_eligible', 'momentum_window_reference',
                 'momentum_window_prediction', 'momentum_window_duration')


def save_reference_chunk(directory, index, arrays):
    """Stream compact reference arrays to disk; keep only file paths in run caches."""
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'intervals-{index:06d}.npz'
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.close(); os.replace(temporary, path)
        finally: temporary.unlink(missing_ok=True)
    return str(path)


def export_references(output, recording, files, metadata):
    import exp1_analysis as shared
    destination = Path(output)/'contact_intervals'/shared.fingerprint(str(recording))[:16]
    destination.mkdir(parents=True, exist_ok=True)
    for i,file in enumerate(files):
        target = destination/f'intervals-{i:06d}.npz'
        if Path(file).resolve() != target.resolve(): shutil.copyfile(file,target)
    signatures = {name:hashlib.sha256((Path(recording)/name).read_bytes()).hexdigest()
                  for name in ('metadata.json','initial_states.jsonl') if (Path(recording)/name).is_file()}
    shared.write_json(destination/'metadata.json',dict(schema_version=1,recording=str(recording),**metadata,source_signatures=signatures,
        chunks=[f'intervals-{i:06d}.npz' for i in range(len(files))],
        units=dict(momentum_contact_world='N',momentum_external_world='N',momentum_com_velocity_world='m/s',
                   momentum_mass_kg='kg',momentum_window_reference='N',momentum_window_prediction='N',time='s'),
        frame='world z-up',reference='Momentum-Derived Net Contact Force, observed momentum not independent sensing',
        timing='row end t+dt minus previous consecutive valid endpoint t; THIS row applied load. Window start/end explicitly recorded.',
        unavailable='NaN arrays; false foot eligibility also includes unknown non-foot contact status'))


def valid_rows(data):
    return (data['valid'].astype(bool) & data['alive_for_metrics'].astype(bool)
            & ~data['physical_failure'].astype(bool) & ~data['artificial_reset'].astype(bool))


def initial_states(path):
    file = Path(path) / 'initial_states.jsonl'
    if not file.exists(): return {}
    with file.open() as stream:
        return {r['trial_id']: r for line in stream if line.strip() for r in [json.loads(line)]}


def model_signature(meta, override=None):
    path = Path(override or meta.get('urdf_source', ''))
    return (str(path), hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)


class KinematicTree:
    """Preparsed URDF transforms; batched link-origin velocities in the root frame.

    Recorded Isaac Lab base states describe the articulation ROOT LINK, not the
    trunk CoM. Body-local CoM positions/masses come from PhysX initial_states, so
    randomized mass/CoM and fixed-link bodies are retained without guessing them.
    """
    def __init__(self, path, names):
        root = ET.parse(path).getroot()
        self.names = tuple(names)
        joints = list(root.findall('joint'))
        children = {j.find('child').get('link') for j in joints}
        parents = {j.find('parent').get('link') for j in joints}
        roots = parents - children
        if len(roots) != 1: raise ValueError('URDF must have one connected root')
        self.root = roots.pop(); self.links = [self.root]; self.joints = []
        while joints:
            ready = [j for j in joints if j.find('parent').get('link') in self.links]
            if not ready: raise ValueError('Disconnected/cyclic URDF')
            for j in ready:
                parent = self.links.index(j.find('parent').get('link'))
                self.links.append(j.find('child').get('link'))
                origin, axis = j.find('origin'), j.find('axis')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                axis = np.fromstring(axis.get('xyz', '1 0 0') if axis is not None else '1 0 0', sep=' ')
                kind = j.get('type')
                if j.find('mimic') is not None: raise ValueError('Mimic joint requires recorded mapping')
                index = None if kind == 'fixed' else self.names.index(j.get('name'))
                if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'): raise ValueError('Unsupported joint')
                axis = axis / np.linalg.norm(axis)
                self.joints.append((parent, xyz, Rotation.from_euler('xyz', rpy).as_matrix(), axis, kind, index))
                joints.remove(j)

    def states(self, q, qd=None):
        q = np.atleast_2d(np.asarray(q, float)); n = len(q)
        qd = np.zeros_like(q) if qd is None else np.asarray(qd, float)
        p = np.zeros((n, len(self.links), 3)); v = np.zeros_like(p); w = np.zeros_like(p)
        r = np.broadcast_to(np.eye(3), (n, len(self.links), 3, 3)).copy()
        for child, (parent, offset, origin, axis, kind, index) in enumerate(self.joints, 1):
            shift = r[:, parent] @ offset
            p[:, child] = p[:, parent] + shift
            v[:, child] = v[:, parent] + np.cross(w[:, parent], shift)
            w[:, child] = w[:, parent]
            rotation = r[:, parent] @ origin
            world_axis = rotation @ axis
            if index is not None:
                if kind == 'prismatic':
                    shift = world_axis*q[:, index, None]
                    p[:, child] += shift
                    v[:, child] += np.cross(w[:, parent], shift) + world_axis*qd[:, index, None]
                else:
                    rotation = rotation @ Rotation.from_rotvec(q[:, index, None]*axis).as_matrix()
                    w[:, child] += world_axis*qd[:, index, None]
            r[:, child] = rotation
        return p, r, v, w

    def com_velocity(self, data, masses, local_com, bodies):
        p, r, v, w = self.states(data['joint_position'], data['joint_velocity'])
        ids = [self.links.index(name) for name in bodies]
        offset = np.einsum('nbij,nbj->nbi', r[:, ids], local_com[..., :3])
        local_p = p[:, ids] + offset
        local_v = v[:, ids] + np.cross(w[:, ids], offset)
        rotation = Rotation.from_quat(data['base_quaternion_xyzw']).as_matrix()
        pos = np.einsum('nij,nbj->nbi', rotation, local_p)
        vel = (np.einsum('nij,nbj->nbi', rotation, local_v)
               + np.cross(data['base_angular_velocity_world'][:, None], pos)
               + data['base_velocity_world'][:, None])
        return np.sum(vel*masses[..., None], axis=1)/masses.sum(1)[:, None]


def flat_ground(meta):
    return meta.get('resolved_environment', {}).get('terrain', {}).get('mesh_type') == 'plane'


def normal_errors(prediction, data, meta):
    measured = data.get('grf_raw_world_interval_mean')
    if measured is None: return None
    if flat_ground(meta): return prediction[..., 2] - measured[..., 2]
    normals = data.get('surface_normals_world')
    if normals is None: return None
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, length, out=np.full_like(normals, np.nan), where=length > 1e-12)
    return np.sum((prediction-measured)*normals, axis=-1)


class MomentumContact:
    """One previous endpoint/window per trial; vectorized kinematics per chunk.

    Windows are non-overlapping, duration-weighted, and reset at stage/load changes
    or gaps. Partial windows are not exported as full-duration comparisons.
    Unknown non-foot contact status permits TOTAL contact reconstruction only.
    """
    def __init__(self, meta, initial, config=None):
        self.meta, self.initial, self.config = meta, initial, config or {}
        self.duration = float(self.config.get('contact_window_seconds', .1))
        if not np.isfinite(self.duration) or self.duration <= 0: raise ValueError('contact_window_seconds must be positive')
        self.previous, self.windows = {}, {}
        self.counts = Counter(); self.notes = set(); self.tree = None; self.tree_attempted = False
        self.gravity = np.asarray(meta.get('resolved_environment', {}).get('sim', {}).get('gravity', [np.nan]*3), float)
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            self.gravity = np.full(3,np.nan); self.notes.add('Recorded gravity unavailable; force reference unavailable')
        threshold = float(self.config.get('contact_nonfoot_threshold_n',1e-6))
        if not np.isfinite(threshold) or threshold < 0: raise ValueError('contact_nonfoot_threshold_n must be finite and nonnegative')

    def endpoints(self, data):
        n = len(data['trial_id']); velocity = np.full((n, 3), np.nan); mass = np.full(n, np.nan)
        try:
            properties = [self.initial[int(t)] for t in data['trial_id']]
            masses = np.array([p['body_masses_kg'] for p in properties], float)
            if (masses < 0).any() or not np.isfinite(masses).all() or (masses.sum(1) <= 0).any():
                raise ValueError('Invalid recorded body masses')
            mass = masses.sum(1)
            if 'robot_com_velocity_world' in data:
                velocity = np.asarray(data['robot_com_velocity_world'], float)
                self.counts['recorded_com_endpoints'] += n
            elif 'body_com_velocities_world' in data:
                velocity = np.sum(data['body_com_velocities_world'][..., :3]*masses[..., None], 1)/mass[:, None]
                self.counts['mass_weighted_body_endpoints'] += n
            else:
                if not self.tree_attempted:
                    self.tree_attempted = True
                    path, digest = model_signature(self.meta, self.config.get('urdf'))
                    if digest is None or digest != self.meta.get('urdf_sha256'): raise ValueError('Verified URDF unavailable/mismatch')
                    self.tree = KinematicTree(path, self.meta['dof_names'])
                if self.tree is None: return velocity, mass
                local = np.array([p['body_local_com_pose'] for p in properties], float)
                velocity = self.tree.com_velocity(data, masses, local, self.meta['body_names'])
                self.counts['kinematic_com_endpoints'] += n
        except (KeyError, ValueError, OSError) as exc:
            self.notes.add('CoM reconstruction unavailable: '+str(exc))
        return velocity, mass

    def external(self, data):
        forces = []
        for name in ('ee', 'base'):
            field = name+'_force_applied_world_interval_mean'
            if field in data:
                forces.append(data[field]); self.notes.add(field+': recorded interval average')
            elif name+'_force_world' in data and 'forces' in self.meta.get('conventions', {}):
                # Exp1 Adapter.loads(t) sets world loads ONCE before step; only
                # target commands change at t+1. These are held actual applied loads.
                forces.append(data[name+'_force_world'])
                self.notes.add(name+'_force_world: collector-held actual load over [t,t+dt)')
            else:
                self.notes.add(name+' applied interval force unavailable; never assumed zero')
                return np.full((len(data['trial_id']), 3), np.nan)
        rand = self.meta.get('resolved_environment', {}).get('domain_rand', {})
        if rand.get('push_robots', False):
            self.notes.add('Unrecorded pushes enabled; complete external impulse unavailable')
            return np.full_like(forces[0], np.nan)
        return np.asarray(forces[0], float)+forces[1]

    def foot_only(self, data):
        n = len(data['trial_id']); threshold = self.config.get('contact_nonfoot_threshold_n', 1e-6)
        if 'nonfoot_contact' in data: return ~np.asarray(data['nonfoot_contact'], bool)
        ids = self.meta.get('nonfoot_body_indices')
        if ids is None and 'body_names' in self.meta and 'foot_indices' in self.meta:
            ids = sorted(set(range(len(self.meta['body_names'])))-set(self.meta['foot_indices']))
        if ids is not None:
            if 'body_contact_forces_raw_world_interval_mean' in data:
                forces = data['body_contact_forces_raw_world_interval_mean'][:, ids]
                clear = np.isfinite(forces).all((1, 2)) & (np.linalg.norm(forces, axis=-1) <= threshold).all(1)
                if 'body_contacts_interval_mean' in data:
                    # A substep contact flag catches impulses masked by signed averaging.
                    clear &= (data['body_contacts_interval_mean'][:, ids] == 0).all(1)
                return clear
            if 'body_contacts_interval_mean' in data:
                contact = data['body_contacts_interval_mean'][:, ids]
                return np.isfinite(contact).all(1) & (contact == 0).all(1)
        self.notes.add('Non-foot contacts unrecorded: net foot-GRF metric unavailable; total contact reference retained')
        return np.zeros(n, bool)

    def process(self, data, predicted_net=None):
        n = len(data['trial_id']); velocity, masses = self.endpoints(data)
        external = self.external(data); good = valid_rows(data)
        eligible = self.foot_only(data)
        force = np.full((n, 3), np.nan); start = np.full(n, np.nan)
        reference = np.full_like(force, np.nan); prediction = np.full_like(force, np.nan)
        window_foot = np.zeros(n, bool)
        window_duration = np.full(n, np.nan); window_start = np.full(n, np.nan)
        known = good & np.isfinite(velocity).all(1) & np.isfinite(masses) & np.isfinite(external).all(1)
        stage = data.get('task_stage_interval', data.get('phase', np.zeros(n)))
        # Equal resultants can hide redistribution between EE/torso, or a pure
        # moment change. These still delimit distinct loading stages.
        load_fields = [k for k in ('ee_force_applied_world_interval_mean','ee_force_world',
            'base_force_applied_world_interval_mean','base_force_world','base_moment_world') if k in data]
        load_signature = np.concatenate([data[k] for k in load_fields],axis=1) if load_fields else external
        for i, (tid, step) in enumerate(zip(data['trial_id'], data['step'])):
            tid, step = int(tid), int(step)
            t, end = float(data['time_seconds'][i]), float(data['successor_time_seconds'][i])
            old = self.previous.get(tid)
            self.previous[tid] = (step, end, velocity[i].copy(), known[i], masses[i])
            self.counts['rows'] += 1
            contiguous = (old is not None and old[0]+1 == step and old[3] and known[i]
                          and end > t and np.isclose(old[1], t, atol=1e-7, rtol=0)
                          and np.isclose(old[4], masses[i]))
            if not contiguous:
                self.counts['unpaired_or_invalid'] += 1; self.windows.pop(tid, None); eligible[i] = False
                continue
            force[i] = masses[i]*(velocity[i]-old[2])/(end-t)-masses[i]*self.gravity-external[i]
            if not np.isfinite(force[i]).all():
                self.windows.pop(tid, None); eligible[i] = False; continue
            start[i] = t; self.counts['contact_intervals'] += 1
            self.counts['foot_only_intervals'] += int(eligible[i])
            window = self.windows.get(tid)
            key = (stage[i].item() if isinstance(stage[i], np.generic) else stage[i], tuple(load_signature[i]))
            if window is None or window['key'] != key:
                window = dict(key=key, start=t, velocity=old[2].copy(), impulse=np.zeros(3),
                              predicted=np.zeros(3), foot=True, predicted_valid=True)
                self.windows[tid] = window
            window['impulse'] += external[i]*(end-t)
            window['foot'] &= bool(eligible[i])
            if predicted_net is None or not np.isfinite(predicted_net[i]).all(): window['predicted_valid'] = False
            else: window['predicted'] += predicted_net[i]*(end-t)
            duration = end-window['start']
            if duration+1e-8 >= self.duration:
                reference[i] = masses[i]*(velocity[i]-window['velocity'])/duration-masses[i]*self.gravity-window['impulse']/duration
                if window['foot'] and window['predicted_valid']: prediction[i] = window['predicted']/duration
                window_foot[i] = window['foot']
                window_duration[i], window_start[i] = duration, window['start']
                self.windows.pop(tid); self.counts['complete_windows'] += 1
        return dict(momentum_contact_world=force, momentum_foot_eligible=eligible,
                    momentum_window_reference=reference, momentum_window_prediction=prediction,
                    momentum_window_duration=window_duration, momentum_window_start=window_start,
                    momentum_window_foot_eligible=window_foot,
                    momentum_interval_start=start, momentum_external_world=external,
                    momentum_com_velocity_world=velocity, momentum_mass_kg=masses)
