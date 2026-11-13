"""Method-neutral BR/MULE logging. Detached reductions; no forwards, RNG or loss changes.

Episode/rew_* matches HardPACT's sum(weighted per-step reward)/maximum episode
seconds, count-weighted over COMPLETE episodes ending in this reporting window.
Missing measurements are omitted with explicit counts, not zeros.
Failure reasons are overlapping conditions at termination, not causal attribution.
"""
from copy import deepcopy
import math
import torch


class LearningEpisodeDiagnostics:
    def __init__(self, num_envs, device, dt=None, max_episode_seconds=None, streams=()):
        self.num_envs, self.device = num_envs, torch.device(device)
        self.streams=tuple(streams)
        self.dt = dt if dt is not None and math.isfinite(dt) and dt > 0 else None
        self.maximum = (max_episode_seconds if max_episode_seconds is not None
                        and math.isfinite(max_episode_seconds) and max_episode_seconds > 0 else None)
        self.term_sums, self.term_steps = {}, {}
        self.recent_returns=torch.zeros(0,len(self.streams) or 1,device=self.device)
        self.recent_lengths=torch.zeros(0,device=self.device)
        self.begin_iteration()

    def begin_iteration(self):
        self.sums = {}

    def _add(self, key, value):
        if key not in self.sums: self.sums[key] = torch.zeros((), dtype=torch.float64, device=self.device)
        self.sums[key].add_(value.detach().double())

    @torch.no_grad()
    def record(self, done, timeout, lengths, *, failure=None, reasons=None, reward_terms=None, raw_timeout=None, returns=None):
        if returns is not None:
            # Same last-100-completed-episodes convention as HardPACT's rew/len buffers.
            values=returns.detach().reshape(self.num_envs,-1)[done]
            self.recent_returns=torch.cat((self.recent_returns,values))[-100:]
            self.recent_lengths=torch.cat((self.recent_lengths,lengths[done].float()))[-100:]
        failure = done & ~timeout if failure is None else done & failure
        timeout = done & timeout & ~failure
        for name, mask in (('all', done), ('failure', failure), ('timeout', timeout),
                           ('other', done & ~failure & ~timeout)):
            self._add(f'outcome/{name}/count', mask.sum())
            self._add(f'outcome/{name}/steps', lengths[mask].sum())
        if raw_timeout is not None:
            self._add('failure_timeout_overlap', (failure & raw_timeout).sum())
        attributed = torch.zeros_like(done)
        if reasons is not None:
            for name, mask in reasons.items():
                if mask.shape != done.shape or mask.dtype != torch.bool or mask.device != done.device:
                    raise ValueError(f'Invalid diagnostic reason {name}')
                self._add('reason/'+name, (mask & failure).sum())
                attributed |= mask
            self._add('reason/unattributed', (failure & ~attributed).sum())
            self._add('reason_available', failure.sum())
        else:
            self._add('reason_unavailable', failure.sum())
        terms = reward_terms or {}
        for name, values in terms.items():
            if values.shape != done.shape or values.device != done.device:
                raise ValueError(f'Invalid diagnostic reward term {name}')
            if name not in self.term_sums:
                self.term_sums[name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
                self.term_steps[name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            finite = torch.isfinite(values)
            self.term_sums[name].add_(torch.where(finite, values.detach(), 0.).double())
            self.term_steps[name].add_(finite.long())
        # Missing terms mid-episode cannot silently become complete episode returns.
        for name, values in self.term_sums.items():
            valid = done & (lengths > 0) & (self.term_steps[name] == lengths)
            self._add('term_count/'+name, valid.sum())
            self._add('term_incomplete/'+name, (done & ~valid).sum())
            self._add('term_sum/'+name, values[valid].sum())
            values[done] = 0.; self.term_steps[name][done] = 0

    def metrics(self, *, active_streams=None):
        if not self.sums: return {}
        report=dict(self.sums)
        if self.recent_lengths.numel():
            report['rolling/length']=self.recent_lengths.mean()
            for i,name in enumerate(self.streams or ('',)):
                report['rolling/return/'+name]=self.recent_returns[:,i].mean()
        values = torch.stack(list(report.values())).cpu().tolist()
        sums = dict(zip(report, values)); result = {}
        if 'rolling/length' in sums:
            result['Train/mean_episode_length']=sums['rolling/length']
            result['train/episode_length']=sums['rolling/length']
            for name in self.streams or ('',):
                if active_streams is not None and name not in active_streams: continue
                value=sums['rolling/return/'+name]
                result['Train/'+(name+'/' if name else '')+'mean_reward']=value
                result['train/'+(name+'/' if name else '')+'return']=value
        total = sums.get('outcome/all/count', 0.)
        failures = sums.get('outcome/failure/count', 0.)
        for outcome in ('all', 'failure', 'timeout', 'other'):
            count = sums.get(f'outcome/{outcome}/count', 0.)
            prefix = 'EpisodeDiagnostics/all/'+outcome
            result[prefix+'_count'] = count
            if count:
                result[prefix+'_mean_length_steps'] = sums[f'outcome/{outcome}/steps']/count
                if self.dt is not None: result[prefix+'_mean_length_s'] = result[prefix+'_mean_length_steps']*self.dt
            if total and outcome != 'all': result[prefix+'_fraction'] = count/total
        for key, value in sums.items():
            if key.startswith('reason/'):
                prefix = 'EpisodeDiagnostics/'+key
                result[prefix+'_count'] = value
                if failures: result[prefix+'_fraction_of_failures'] = value/failures
        for key in ('reason_available', 'reason_unavailable', 'failure_timeout_overlap'):
            if key in sums: result['EpisodeDiagnostics/'+key+'_count'] = sums[key]
        if self.maximum is not None: result['EpisodeDiagnostics/reward_normalization_seconds'] = self.maximum
        for name in self.term_sums:
            count = sums.get('term_count/'+name, 0.)
            result['EpisodeDiagnostics/reward_valid_count/'+name] = count
            result['EpisodeDiagnostics/reward_incomplete_count/'+name] = sums.get('term_incomplete/'+name, 0.)
            if count:
                mean = sums['term_sum/'+name]/count
                if self.maximum is not None:
                    path, _, term = name.rpartition('/')
                    result['Episode/'+(path+'/' if path else '')+'rew_'+term] = mean/self.maximum
        return result

    def clear_episodes(self):
        self.term_sums.clear(); self.term_steps.clear(); self.begin_iteration()
        self.recent_returns=self.recent_returns[:0]; self.recent_lengths=self.recent_lengths[:0]

    def state_dict(self):
        return deepcopy(dict(version=1, term_sums=self.term_sums, term_steps=self.term_steps,
            recent_returns=self.recent_returns,recent_lengths=self.recent_lengths))

    def validate_state_dict(self, state):
        if state is None: return  # Old checkpoint: do not fabricate partial episode totals.
        if state.get('version') != 1 or set(state['term_sums']) != set(state['term_steps']):
            raise ValueError('Incompatible episode diagnostics state')
        for group, dtype in (('term_sums', torch.float64), ('term_steps', torch.long)):
            for value in state[group].values():
                if value.shape != (self.num_envs,) or value.dtype != dtype or not torch.isfinite(value).all():
                    raise ValueError('Invalid episode diagnostic buffers')
        if 'recent_lengths' in state:
            n=state['recent_lengths'].numel()
            if n>100 or state['recent_lengths'].shape!=(n,) or state['recent_returns'].shape!=(n,len(self.streams) or 1):
                raise ValueError('Invalid rolling episode buffers')

    def load_state_dict(self, state):
        self.validate_state_dict(state); self.clear_episodes()
        if state is not None:
            self.term_sums = {k: v.to(self.device).clone() for k, v in state['term_sums'].items()}
            self.term_steps = {k: v.to(self.device).clone() for k, v in state['term_steps'].items()}
            if 'recent_lengths' in state:
                self.recent_returns=state['recent_returns'].to(self.device).clone()
                self.recent_lengths=state['recent_lengths'].to(self.device).clone()


@torch.no_grad()
def policy_diagnostics(std, joint_names, *, prefix='Policy'):
    """Same position-channel std tags/units as HardPACT; no policy recomputation."""
    std = std.detach()
    if std.shape != (len(joint_names),): raise ValueError('Policy std/joint ordering differs')
    tensors = {prefix+'/mean_noise_std': std.mean(), prefix+'/mean_noise_std_position': std.mean()}
    return dict(zip(tensors, torch.stack(list(tensors.values())).cpu().tolist()))


def compatible_training_metrics(metrics, collection_s, update_s, interactions, streams=()):
    """Canonical published tags; lower-case internal runner metrics are NOT logged."""
    duration = collection_s+update_s
    result = {'Perf/collection time': collection_s, 'Perf/learning_time': update_s,
              'Perf/iteration_time': duration, 'Perf/total_fps': interactions/max(duration, 1e-9)}
    if not streams:
        for src, dst in (('loss/value', 'Loss/value_function'), ('loss/surrogate', 'Loss/surrogate'),
                         ('train/learning_rate', 'Loss/learning_rate'),
                         ('train/episode_return', 'Train/mean_reward'),
                         ('train/episode_length_steps', 'Train/mean_episode_length'),
                         ('train/episode_return', 'train/return'), ('train/episode_length_steps', 'train/episode_length')):
            if src in metrics: result[dst] = metrics[src]
    else:
        for owner in streams:
            for src, dst in ((f'loss/{owner}/value', f'Loss/{owner}/value_function'),
                             (f'loss/{owner}/surrogate', f'Loss/{owner}/surrogate'),
                             (f'train/{owner}/learning_rate', f'Loss/{owner}/learning_rate'),
                             (f'train/{owner}_episode_return', f'Train/{owner}/mean_reward')):
                if src in metrics: result[dst] = metrics[src]
        if 'train/episode_length_steps' in metrics:
            result['Train/mean_episode_length'] = metrics['train/episode_length_steps']
            result['train/episode_length'] = metrics['train/episode_length_steps']
    if not streams:
        for src, dst in (('latent_matching','latent_matching'), ('load_estimation','load_estimation'),
                         ('kl','policy_kl'), ('ratio','policy_ratio')):
            if 'loss/'+src in metrics: result['Loss/'+dst] = metrics['loss/'+src]
        if 'train/entropy_coefficient' in metrics: result['Values/entropy'] = metrics['train/entropy_coefficient']
    else:
        for owner in streams:
            for key in ('kl', 'clip_fraction', 'ratio_mean'):
                if f'loss/{owner}/{key}' in metrics: result[f'Loss/{owner}/{key}'] = metrics[f'loss/{owner}/{key}']
            if f'train/{owner}/entropy_coef' in metrics:
                result[f'Values/{owner}/entropy'] = metrics[f'train/{owner}/entropy_coef']
        for src, dst in (('total','autoenc_function'), ('velocity_mse','velo_pred'),
                         ('reconstruction_mse','recon'), ('latent_kl','kl_div')):
            if 'loss/auxiliary/'+src in metrics: result['Loss/'+dst] = metrics['loss/auxiliary/'+src]
    for key, value in metrics.items():
        if key.startswith('load/load_mae/'):
            result['LoadEstimation/mae/'+key.rsplit('/',1)[-1]] = value
        elif key.startswith('load/load_rmse/'):
            result['LoadEstimation/rmse/'+key.rsplit('/',1)[-1]] = value
    return result


def format_training_summary(method, iteration, phase, metrics, budget=None):
    def fmt(value): return f'{value:.5g}' if math.isfinite(value) else 'unavailable'
    stage = f" stage={metrics['Train/environment_stage']}" if 'Train/environment_stage' in metrics else ''
    progress = str(iteration) if budget is None else f'{iteration}/{budget}'
    lines = ['#'*80, f'Learning iteration {progress} | {method} | {phase}{stage}'.center(80),
        f"{'Computation:':>35} {metrics['Perf/total_fps']:.0f} steps/s "
        f"(collection: {metrics['Perf/collection time']:.3f}s, learning {metrics['Perf/learning_time']:.3f}s)"]
    names = {'Train/mean_reward':'Mean reward', 'Train/mean_episode_length':'Mean episode length',
             'Policy/mean_noise_std':'Mean action noise std', 'Loss/value_function':'Value function loss',
             'Loss/surrogate':'Surrogate loss', 'Perf/total_timesteps':'Total timesteps', 'Perf/total_time':'Total time [s]'}
    for key, value in metrics.items():
        show = (key.startswith(('Loss/', 'Episode/')) or key in names or
                (key.startswith('Policy/') and key.endswith('/mean_noise_std')) or
                (key.startswith('Train/') and key.endswith('/mean_reward')) or
                (key.startswith('EpisodeDiagnostics/') and key.endswith('_count')
                 and ('/all/' in key or '/reason/' in key) and value > 0))
        if show:
            label = names.get(key, 'Mean episode '+key[8:] if key.startswith('Episode/') else key)
            lines.append(f'{label+":":>35} {fmt(value)}')
    return '\n'.join(lines)
