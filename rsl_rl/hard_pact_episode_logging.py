"""HardPACT-only reset diagnostics; never changes termination or curriculum.

GPU sums cover complete episodes ending in the reporting interval, not a mean
of per-reset-batch means. Terrain IDs/levels are sampled BEFORE reset changes
them. The first reset of each env is excluded (runner may randomize initial
episode ages). Failure > timeout > other defines disjoint outcomes; failure
reasons describe simultaneous conditions on the terminating control step and
may overlap. They are not claims about the original cause of a fall.
"""
from bisect import bisect_right
from itertools import accumulate
import math

import torch


TERRAIN_NAMES = ('slope', 'rough', 'stairs_down', 'stairs_up', 'discrete_obstacles',
                 'wave', 'stepping_stones', 'gap', 'pit', 'platforms', 'platforms_and_gaps')
REASONS = ('contact', 'roll', 'pitch', 'height_min', 'height_max', 'unattributed')
OUTCOMES = ('all', 'failure', 'timeout', 'other')


class HardPACTEpisodeDiagnostics:
    def __init__(self, env):
        self.interval = int(env.cfg.env.episode_diagnostics_interval)
        if self.interval < 1:
            raise ValueError('episode_diagnostics_interval must be positive')
        terrain = env.cfg.terrain
        self.level_count = int(terrain.num_rows)
        self.column_kinds = None
        if (getattr(terrain, 'curriculum', False) and not getattr(terrain, 'selected', False)
                and terrain.mesh_type in ('heightfield', 'trimesh')):
            # Same family selection as Terrain.curiculum/make_terrain.
            thresholds = list(accumulate(terrain.terrain_proportions))[:8]
            self.column_kinds = torch.tensor([
                bisect_right(thresholds, j/terrain.num_cols+.001)
                for j in range(terrain.num_cols)], device=env.device)
        self.groups = (['all'] + ['terrain/'+name for name in (*TERRAIN_NAMES, 'unknown')]
                       + ['level/'+str(i) for i in range(self.level_count)] + ['level/unknown'])
        self.counts = torch.zeros(len(self.groups), len(OUTCOMES), device=env.device, dtype=torch.float64)
        self.seconds = torch.zeros_like(self.counts)
        self.reasons = torch.zeros(len(REASONS), device=env.device, dtype=torch.float64)
        self.excluded_initial = self.reasons.new_zeros(())
        self.failure_timeout_overlap = self.reasons.new_zeros(())
        self.out_of_bounds = self.reasons.new_zeros(())
        self.complete_episode = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.last_iteration = None
        self.reporting_iterations = 0

    def begin_iteration(self, iteration):
        if iteration != self.last_iteration:
            self.reporting_iterations += 1
            self.last_iteration = iteration

    @torch.no_grad()
    def record(self, env, ids):
        if ids.numel() == 0:
            return
        sim = env.simulator
        eligible = self.complete_episode[ids] & (env.episode_length_buf[ids] > 0)
        self.excluded_initial.add_((~self.complete_episode[ids]).sum())
        self.complete_episode[ids] = True
        duration = env.episode_length_buf[ids].double()*env.dt
        # Use the existing accumulated failure decision, not a new rule.
        failed = env.fail_buf[ids] > env.cfg.env.fail_to_terminal_time_s/env.dt
        timeout = env.time_out_buf[ids].bool()
        outcome = torch.where(failed, 1, torch.where(timeout, 2, 3))
        kinds = getattr(sim, 'terrain_kind_ids', None)
        if kinds is not None:
            family = kinds[ids].long()
        elif self.column_kinds is not None and hasattr(sim, 'terrain_types'):
            columns = sim.terrain_types[ids].long()
            safe_columns = columns.clamp(0, self.column_kinds.numel()-1)
            family = torch.where((columns >= 0) & (columns < self.column_kinds.numel()),
                                 self.column_kinds[safe_columns], len(TERRAIN_NAMES))
        else:
            family = torch.full_like(ids, len(TERRAIN_NAMES))
        family = torch.where((family >= 0) & (family < len(TERRAIN_NAMES)), family, len(TERRAIN_NAMES))
        levels = getattr(sim, 'terrain_levels', None)
        level = levels[ids].long() if levels is not None else torch.full_like(ids, self.level_count)
        level = torch.where((level >= 0) & (level < self.level_count), level, self.level_count)
        group = torch.stack((torch.zeros_like(ids), 1+family, 2+len(TERRAIN_NAMES)+level), -1)
        status = torch.stack((torch.zeros_like(outcome), outcome), -1)
        index = (group[:,:,None]*len(OUTCOMES)+status[:,None,:]).flatten()
        weight = eligible[:,None,None].expand(-1,3,2).double().flatten()
        self.counts.flatten().index_add_(0, index, weight)
        self.seconds.flatten().index_add_(0, index, (duration[:,None,None]*eligible[:,None,None]).expand(-1,3,2).flatten())

        contact = (sim.link_contact_forces[ids][:,sim.termination_contact_indices].norm(dim=-1)>10.).any(-1)
        flags = [contact]
        termination = getattr(env.cfg, 'termination', None)
        if termination is not None:
            # Clone-free, out-of-place equivalent of legacy wrap_to_pi.
            rpy = torch.remainder(sim._base_euler[ids,:2], 2*math.pi)
            rpy = torch.where(rpy > math.pi, rpy-2*math.pi, rpy)
            height = (sim.base_pos[ids,2,None]-sim.measured_heights[ids]).mean(-1)
            tests = (rpy[:,0].abs()>termination.roll_threshold,
                     rpy[:,1].abs()>termination.pitch_threshold,
                     height<termination.height_min, height>termination.height_max)
            flags += [test if name in termination.termination_terms else torch.zeros_like(contact)
                      for name,test in zip(REASONS[1:5], tests)]
        else:
            flags += [torch.zeros_like(contact)]*4
        active = torch.stack(flags,-1)
        active = torch.cat((active, ~active.any(-1,keepdim=True)), -1)
        self.reasons.add_((active & (eligible & failed)[:,None]).sum(0))
        self.failure_timeout_overlap.add_((eligible & failed & timeout).sum())
        bounds = getattr(env, 'non_failure_reset_buf', None)
        if getattr(env.cfg.terrain, 'reset_out_of_bounds', False) and bounds is not None:
            self.out_of_bounds.add_((eligible & ~failed & ~timeout & bounds[ids].bool()).sum())

    def metrics(self):
        nan = self.seconds.new_tensor(float('nan'))
        result = {}
        for g, name in enumerate(self.groups):
            for o, outcome in enumerate(OUTCOMES):
                prefix = f'EpisodeDiagnostics/{name}/{outcome}'
                result[prefix+'_count'] = self.counts[g,o]
                result[prefix+'_mean_length_s'] = torch.where(self.counts[g,o]>0,
                    self.seconds[g,o]/self.counts[g,o].clamp_min(1), nan)
        total, failed = self.counts[0,0], self.counts[0,1]
        for i, name in enumerate(REASONS):
            result[f'EpisodeDiagnostics/reason/{name}_count'] = self.reasons[i]
            result[f'EpisodeDiagnostics/reason/{name}_fraction_of_failures'] = torch.where(
                failed>0, self.reasons[i]/failed.clamp_min(1), nan)
        for i,name in enumerate(OUTCOMES[1:],1):
            result[f'EpisodeDiagnostics/all/{name}_fraction'] = torch.where(
                total>0, self.counts[0,i]/total.clamp_min(1), nan)
        result['EpisodeDiagnostics/excluded_initial_episodes_count'] = self.excluded_initial
        result['EpisodeDiagnostics/failure_timeout_overlap_count'] = self.failure_timeout_overlap
        result['EpisodeDiagnostics/out_of_bounds_count'] = self.out_of_bounds
        result['EpisodeDiagnostics/reporting_iterations'] = total.new_tensor(self.reporting_iterations)
        return result

    def clear(self):
        for tensor in (self.counts, self.seconds, self.reasons, self.excluded_initial,
                       self.failure_timeout_overlap, self.out_of_bounds):
            tensor.zero_()
        self.reporting_iterations = 0


def log_hard_pact_episodes(env, writer, iteration):
    """One compact on-device reduction/CPU transfer per scheduled report."""
    recorder = getattr(env, '_episode_diagnostics', None)
    if recorder is None or (iteration+1) % recorder.interval or not recorder.reporting_iterations:
        return
    metrics = recorder.metrics()
    values = torch.stack(tuple(metrics.values())).detach().cpu().tolist()
    for name, value in zip(metrics, values):
        writer.add_scalar(name, value, iteration)
    recorder.clear()
