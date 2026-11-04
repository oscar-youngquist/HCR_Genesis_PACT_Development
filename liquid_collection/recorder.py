"""Version-audited Genesis native-substep hooks, installed only in a worker.

The installed legacy coupler applies -m*delta(v)/dt at the particle position.
Sampling its reordered state on both sides measures that impulse without
mistaking particle.acc for the total acceleration. No installation is edited.
"""
import math
import numpy as np
import torch
from .physics import aggregates, coupling_impulse, rotate, CausalClock
from .cadence import GridAccumulator, interval_fields
from .storage import ChunkWriter


class NativeRecorder:
    def __init__(self, env, writer, job):
        self.env, self.writer, self.job = env, writer, job
        self.sim = env.simulator
        self.native = self.sim._scene.sim
        self.sph = self.native.sph_solver
        self.coupler = self.native.coupler
        if type(self.coupler).__name__ != 'LegacyCoupler' or not self.coupler._rigid_sph:
            raise RuntimeError('Recorder requires inspected LegacyCoupler rigid-SPH path')
        if self.sph.n_particles != self.sim._liquid.n_particles or self.sph._ffs:
            raise RuntimeError('Only a single liquid and no external fluid force fields are supported')
        for flag in ('_mpm_sph', '_fem_sph'):
            if getattr(self.coupler, flag, False):
                raise RuntimeError('Additional fluid couplings are unsupported')
        self.device = env.device
        self.dt = float(self.sph.substep_dt)
        self.policy_ticks = round(env.dt/self.dt)
        feature_ticks = round(job['feature_dt']/self.dt)
        for interval, ticks in ((env.dt, self.policy_ticks), (job['feature_dt'], feature_ticks), (.005, round(.005/self.dt))):
            if ticks < 1 or not math.isclose(ticks*self.dt, interval, rel_tol=0, abs_tol=1e-9):
                raise ValueError('Policy, feature and 200 Hz intervals must be integral native steps')
        self.mass = self.sph.particles_info.mass.to_torch(device=self.device)[None].expand(env.num_envs, -1).clone()
        self.rest_rho = self.sph.particles_info.rho.to_torch(device=self.device)[None].expand_as(self.mass)
        if not bool(torch.isfinite(self.mass).all() & (self.mass > 0).all()):
            raise RuntimeError('Invalid actual solver particle masses')
        self.gravity = self.sph._gravity.to_torch(device=self.device).clone()
        self.tick = 0
        self.clock = CausalClock(self.base()['velocity_W'], feature_ticks)
        self.completed = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.episodes_per_env = job.get('episodes_per_env')
        self.episode_supported = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
        self.forced_reset = False
        self._bad_counts = {}
        self.save_mode = job.get('save_mode', '200hz')
        self.grid_ticks = round(.005/self.dt)
        self.sequence = 0
        self.feature_phase_by_env = np.zeros(env.num_envs, dtype=np.int64)
        self.feature_phases = {0}
        self.accumulator = GridAccumulator(self.dt)
        self.event_writer = self.boundary_writer = None
        if self.save_mode == '200hz':
            self.event_writer = ChunkWriter(writer.path/'events', writer.chunk_size, writer.max_bytes)
            self.boundary_writer = ChunkWriter(writer.path/'boundaries', writer.chunk_size, writer.max_bytes)
        self.last_commands = None
        self.original_substep, self.original_couple = self.native.substep, self.coupler.couple
        self.original_reset = env.reset_idx
        self.native.substep = self.substep
        self.coupler.couple = self.couple
        env.reset_idx = self.reset
        self.state = self.read_state()
        self.emit(self.state, reset=torch.ones(env.num_envs, dtype=torch.bool, device=self.device))

    def base(self):
        robot = self.sim._robot
        # Installed get_vel() defaults to link_origin, not link/root CoM.
        return dict(x_O_W=robot.get_pos().clone(), q_WB_wxyz=robot.get_quat().clone(),
                    velocity_W=torch.cat((robot.get_vel(), robot.get_ang()), -1))

    def read_state(self):
        b = self.base()
        x, u = self.sim._liquid.get_particles_pos(), self.sim._liquid.get_particles_vel()
        active = self.sim._liquid.get_particles_active()
        a = aggregates(x, u, self.mass, b['x_O_W'], b['q_WB_wxyz'], b['velocity_W'][:, :3], b['velocity_W'][:, 3:])
        local = rotate(b['q_WB_wxyz'][:, None], x-b['x_O_W'][:, None], True)
        geom = self.job['geometry']
        lower, upper = x.new_tensor(geom['lower_B']), x.new_tensor(geom['upper_B'])
        domain_low = x.new_tensor(self.sim.liquid_domain['lower_bound'])
        domain_high = x.new_tensor(self.sim.liquid_domain['upper_bound'])
        finite = torch.isfinite(torch.cat((x, u), -1)).all(dim=(-2, -1))
        finite &= torch.isfinite(torch.cat(tuple(b.values()), -1)).all(-1)
        # Keep nominal cavity penetration visible: soft SDF coupling can permit
        # millimetres of penetration without escape through the solid shell.
        inside_cavity = ((local >= lower-1e-5) & (local <= upper+1e-5)).all(dim=(-2, -1))
        penetration = torch.maximum(lower-local, local-upper).clamp_min(0).amax(dim=(-2, -1))
        outer_low, outer_high = x.new_tensor(geom['outer_lower_B']), x.new_tensor(geom['outer_upper_B'])
        contained = ((local >= outer_low) & (local <= outer_high)).all(dim=(-2, -1))
        pad = self.sim._cfg.liquid.particle_size/2
        domain_contact = ((x <= domain_low+pad) | (x >= domain_high-pad)).any(dim=(-2, -1))
        attached = torch.ones_like(finite)
        for entity, expected in self.sim.liquid_attachment_targets:
            offset = rotate(b['q_WB_wxyz'], entity.get_pos()-b['x_O_W'], True)
            attached &= (offset-offset.new_tensor(expected)).abs().amax(-1) < 1e-4
            attached &= (entity.get_quat()*b['q_WB_wxyz']).sum(-1).abs() > 1-1e-5
        result = dict(b, **a, active_count=active.sum(-1), finite=finite,
                      contained=contained, domain_contact=domain_contact, attached=attached,
                      inside_nominal_cavity=inside_cavity, max_wall_penetration_m=penetration,
                      all_active=active.all(-1), particle_min_B=local.amin(-2), particle_max_B=local.amax(-2),
                      particle_min_W=x.amin(-2), particle_max_W=x.amax(-2))
        if getattr(self.sim._cfg.liquid, 'resample_terrain_on_reset', False) and self.sim.custom_origins:
            result.update(terrain_level=self.sim._terrain_levels.clone(),
                          terrain_type=self.sim._terrain_types.clone(),
                          terrain_origin_W=self.sim.env_origins.clone())
        if self.job['snapshots'] and self.tick % self.job['snapshots'] == 0:
            # A separate chunk stream keeps the dense schema fixed.
            self.snapshot = dict(tick=np.array(self.tick), pos_W=x.cpu().numpy(), vel_W=u.cpu().numpy(),
                                 active=active.cpu().numpy(), episode_id=self.clock.episode.cpu().numpy(),
                                 episode_selected=(torch.ones_like(self.completed, dtype=torch.bool)
                                     if self.episodes_per_env is None else self.completed < self.episodes_per_env).cpu().numpy())
        else:
            self.snapshot = None
        return result

    def couple(self, f):
        # Reordering has happened; use its masses/active mask at this phase.
        x = self.sph.particles_reordered.pos.to_torch(device=self.device).transpose(0, 1)
        before = self.sph.particles_reordered.vel.to_torch(device=self.device).transpose(0, 1)
        mass = self.sph.particles_info_reordered.mass.to_torch(device=self.device).transpose(0, 1)
        active = self.sph.particles_ng_reordered.active.to_torch(device=self.device).transpose(0, 1).bool()
        mass = torch.where(active, mass, 0.)
        # Inactive reordered slots have unspecified state. Zero them before sums.
        x = torch.where(active[..., None], x, 0.)
        before = torch.where(active[..., None], before, 0.)
        pose = self.base()
        self.original_couple(f)
        after = self.sph.particles_reordered.vel.to_torch(device=self.device).transpose(0, 1)
        after = torch.where(active[..., None], after, 0.)
        impulse, moment, world_moment = coupling_impulse(x, before, after, mass, pose['x_O_W'])
        self.interval = dict(coupling_force_impulse_W=impulse, coupling_moment_impulse_O_W=moment,
            coupling_moment_impulse_world_origin_W=world_moment,
            coupling_origin_W=pose['x_O_W'], coupling_q_WB_wxyz=pose['q_WB_wxyz'],
            coupling_force_impulse_B=rotate(pose['q_WB_wxyz'], impulse, True),
            coupling_moment_impulse_O_B=rotate(pose['q_WB_wxyz'], moment, True))

    def substep(self, f):
        start = self.state
        self.original_substep(f)
        self.tick += 1
        end = self.read_state()
        event = self.clock.advance(end['velocity_W'], self.dt)
        impulse_balance = self.mass.sum(-1, keepdim=True)*self.gravity*self.dt-(end['momentum_W']-start['momentum_W'])
        # SPH is kick-then-drift: gravity acts at the interval-start particle
        # positions. Angular audit uses a FIXED world origin at both endpoints.
        gravity_moment = torch.linalg.cross(start['com_W'], self.mass.sum(-1, keepdim=True)*self.gravity)*self.dt
        angular_balance = gravity_moment-(end['angular_momentum_world_origin_W']-start['angular_momentum_world_origin_W'])
        self.interval.update(momentum_balance_force_impulse_W=impulse_balance,
            momentum_balance_moment_impulse_world_origin_W=angular_balance,
            force_closure_residual_impulse_W=impulse_balance-self.interval['coupling_force_impulse_W'],
            moment_closure_residual_impulse_W=angular_balance-self.interval['coupling_moment_impulse_world_origin_W'],
            base_acceleration_interval_average_W=(end['velocity_W']-start['velocity_W'])/self.dt)
        self.emit(end, interval=self.interval, feature=event)
        self.state = end

    def reset(self, ids):
        if len(ids) == 0:
            return self.original_reset(ids)
        mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        # Called before commands, rigid state, liquid or observation history reset.
        self.emit(self.state, terminal=mask, forced=mask & self.forced_reset)
        if not self.forced_reset:
            self.completed[ids] += 1
        self.original_reset(ids)
        self.state = self.read_state()
        self.clock.reset(ids, self.state['velocity_W'])
        if hasattr(self, 'feature_phase_by_env'):
            self.feature_phase_by_env[ids.cpu().numpy()] = self.tick % self.clock.feature_ticks
            self.feature_phases = set(self.feature_phase_by_env.tolist())
        self.episode_supported[ids] = True
        self.emit(self.state, reset=mask, forced=mask & self.forced_reset)

    def policy_event(self):
        self.emit(self.state, policy=True)

    def emit(self, state, interval=None, feature=None, reset=None, terminal=None, forced=None, policy=False):
        B = self.env.num_envs
        zero = torch.zeros(B, dtype=torch.bool, device=self.device)
        row = dict(state)
        q = state['q_WB_wxyz']
        held = self.clock.held
        row.update(held_acceleration_W=held.clone(), gravity_W=self.gravity,
                   z=torch.cat((rotate(q, self.gravity, True), rotate(q, state['velocity_W'][:, :3], True),
                                rotate(q, state['velocity_W'][:, 3:], True), rotate(q, held[:, :3], True),
                                rotate(q, held[:, 3:], True)), -1),
                   commands=self.env.commands.clone(), episode_id=self.clock.episode.clone(),
                   episode_native_step=self.clock.age.clone(), held_valid=self.clock.valid.clone(),
                   feature_update=zero if feature is None else feature,
                   policy_update=zero | policy, reset=zero if reset is None else reset,
                   terminal=zero if terminal is None else terminal, forced_reset=zero if forced is None else forced,
                   interval_valid=zero | (interval is not None), instantaneous_wrench_valid=zero,
                   derivative_reconstruction_valid=zero)
        target = getattr(self, 'episodes_per_env', None)
        row['episode_selected'] = ~zero if target is None else self.completed < target
        row['state_valid'] = state['finite'] & state['contained'] & state['all_active'] & ~state['domain_contact'] & state['attached']
        self.episode_supported &= row['state_valid']
        row['episode_supported'] = self.episode_supported.clone()
        row['command_update'] = (~zero if self.last_commands is None else
                                 (row['commands'] != self.last_commands).any(-1))
        self.last_commands = row['commands'].clone()
        row['interval_valid'] &= row['state_valid'] & self.state['finite'] & self.state['contained'] & self.state['all_active'] & ~self.state['domain_contact'] & self.state['attached']
        row['interval_valid'] &= self.episode_supported
        row['interval_valid'] &= row['episode_selected']
        # Uniform schema for endpoint, policy, terminal and reset records.
        fields = ('coupling_force_impulse_W', 'coupling_moment_impulse_O_W',
                  'coupling_moment_impulse_world_origin_W', 'coupling_origin_W',
                  'coupling_q_WB_wxyz', 'coupling_force_impulse_B', 'coupling_moment_impulse_O_B',
                  'momentum_balance_force_impulse_W', 'momentum_balance_moment_impulse_world_origin_W',
                  'force_closure_residual_impulse_W', 'moment_closure_residual_impulse_W',
                  'base_acceleration_interval_average_W')
        for key in fields:
            width = 4 if key == 'coupling_q_WB_wxyz' else 6 if key == 'base_acceleration_interval_average_W' else 3
            row[key] = interval[key] if interval is not None else torch.zeros(B, width, device=self.device)
        row['force_interval_average_B'] = row['coupling_force_impulse_B']/self.dt
        row['moment_interval_average_O_B'] = row['coupling_moment_impulse_O_B']/self.dt
        for name in ('finite', 'contained', 'all_active', 'attached'):
            self._bad_counts[name] = self._bad_counts.get(name, 0)+(~row[name]).sum()
        for name, value in (('domain_contact', row['domain_contact']),
                            ('nominal_cavity_penetration', ~row['inside_nominal_cavity'])):
            self._bad_counts[name] = self._bad_counts.get(name, 0)+value.sum()
        self.sequence += 1
        if self.save_mode == 'native':
            self.write_row(self.writer, interval_fields(row, self.dt if interval is not None else 0.,
                                                        int(interval is not None)), self.tick, self.sequence)
        else:
            if interval is not None:
                self.accumulator.push(row, self.state['velocity_W'], self.tick, self.sequence)
                if self.tick % self.grid_ticks == 0:
                    self.write_row(self.writer, *self.accumulator.pop())
            elif reset is not None or terminal is not None:
                # Flush an off-grid interval before any physical reset. This
                # splits the batch consistently and never mixes episode impulses.
                self.flush_pending()
                boundary = interval_fields(row, 0., 0)
                if self.tick == 0 and self.sequence == 1:
                    self.write_row(self.writer, boundary, self.tick, self.sequence)
                else:
                    self.write_row(self.boundary_writer, boundary, self.tick, self.sequence)
            # Sparse exact-time events retain command/held-input changes. The
            # feature schedule is known on CPU, avoiding a GPU sync per substep.
            event_due = interval is None or self.tick % self.clock.feature_ticks == 0
            # Per-environment resets can shift feature clocks off the global phase.
            if interval is not None and self.feature_phases:
                event_due |= any(self.tick % self.clock.feature_ticks == p for p in self.feature_phases)
            if event_due:
                keys = ('episode_id', 'episode_native_step', 'episode_selected', 'commands', 'held_acceleration_W',
                        'held_valid', 'feature_update', 'policy_update', 'command_update',
                        'reset', 'terminal', 'forced_reset')
                self.write_row(self.event_writer, {k: row[k] for k in keys}, self.tick, self.sequence)
        if getattr(self, 'snapshot_writer', None) is not None and self.snapshot is not None and interval is not None:
            self.snapshot_writer.append(self.snapshot)

    @property
    def bad_counts(self):
        return {k: int(v) for k, v in self._bad_counts.items()}

    def write_row(self, writer, row, tick, sequence):
        # Transfer only saved endpoints/events; native accumulation stays on GPU.
        B = self.env.num_envs
        keys = list(row)
        packed = torch.cat([row[k].reshape(B, -1).to(torch.float64) for k in keys], -1).cpu().numpy()
        result, offset = {}, 0
        for key in keys:
            value = row[key]
            size = value.numel()//B
            dtype = np.bool_ if value.dtype == torch.bool else np.int64 if value.dtype in (torch.int32, torch.int64) else np.float32
            result[key] = packed[:, offset:offset+size].reshape(tuple(value.shape)).astype(dtype)
            offset += size
        result.update(time_s=np.array(tick*self.dt), native_step=np.array(tick),
                      record_sequence=np.array(sequence), target_grid_event=np.array(tick % self.grid_ticks == 0))
        writer.append(result)

    def flush_pending(self):
        pending = self.accumulator.pop()
        if pending is not None:
            self.write_row(self.boundary_writer, *pending)

    def finish_streams(self, metadata):
        self.flush_pending()
        for writer in (self.event_writer, self.boundary_writer):
            if writer is not None:
                writer.finish(metadata)

    def flush_failure(self):
        if self.save_mode == '200hz' and not self.boundary_writer.failed:
            self.flush_pending()
        for writer in (self.event_writer, self.boundary_writer):
            if writer is not None and not writer.failed:
                writer.flush()

    def close(self):
        self.native.substep, self.coupler.couple = self.original_substep, self.original_couple
        self.env.reset_idx = self.original_reset
