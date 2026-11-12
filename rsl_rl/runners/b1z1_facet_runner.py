"""FACET teacher PPO -> student fine-tuning, with no PACT auxiliary construction."""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import time
import torch

from rsl_rl.facet_model import FACETModel
from rsl_rl.facet_ppo import FACETPPO
from rsl_rl.facet_storage import FACETStorage
from rsl_rl.utils.adaptive_entropy import terrain_entropy_progress
from rsl_rl.utils.simulator_diagnostics import domain_rand_state, load_domain_rand_state, log_tradeoff_curriculum
from legged_gym.envs.b1z1.force_task_utils import (
    staged_force_curriculum_state_dict, load_staged_force_curriculum_state_dict,
    update_force_curriculum_from_rollout)


class B1Z1FACETRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.env, self.cfg, self.device, self.log_dir = env, deepcopy(train_cfg), device, log_dir
        if not env.cfg.facet_observations.enabled or not env.cfg.facet_tracking.enabled:
            raise ValueError('FACET training requires observation packets and temporal tracking rewards')
        packet = env.get_facet_observations()
        self.model = FACETModel(packet['observation'].shape[-1], packet['privileged'].shape[-1]).to(device)
        self.alg = FACETPPO(self.model, self.cfg['algorithm'])
        self.steps = self.cfg['runner']['num_steps_per_env']
        self.save_interval = self.cfg['runner']['save_interval']
        self.current_learning_iteration = 0  # Number of COMPLETED updates; next rollout index.
        self.normalizers_initialized = False
        self.hidden = torch.zeros(env.num_envs, 128, device=device)
        self.ema_hidden = torch.zeros_like(self.hidden)
        self.lengths = deque(maxlen=100)
        self.running_lengths = torch.zeros(env.num_envs, device=device)
        self.writer = None
        self.last_metrics = {}
        if log_dir is not None:
            from legged_gym.utils.helpers import class_to_dict, _json_config_value
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            Path(log_dir, 'facet_resolved_config.json').write_text(json.dumps(_json_config_value(dict(
                training=self.cfg, environment=class_to_dict(env.cfg), observations=env.get_facet_observation_schema(),
                optimized_reward_scales={key: env.reward_scales[key] for key in env.reward_names},
                excluded_rewards=env.cfg.facet_tracking.excluded_rewards)), indent=2))

    def _normalize(self, packet):
        m = self.model
        return dict(obs=m.obs_norm(packet['observation'].to(self.device)),
            priv=m.priv_norm(packet['privileged'].to(self.device)),
            wrench=m.wrench_norm(packet['wrench'].to(self.device)))

    @torch.no_grad()
    def collect(self):
        packet = self.env.get_facet_observations()
        first = not self.normalizers_initialized
        if first:
            self.model.update_normalizers(*(packet[k].to(self.device) for k in ('observation', 'privileged', 'wrench')))
            self.normalizers_initialized = True
        storage = FACETStorage(self.steps, self.env.num_envs, packet['observation'].shape[-1],
                               packet['privileged'].shape[-1], self.device, self.alg.phase)
        storage.initial_hidden.copy_(self.hidden)
        self._norm_moments = {k: [torch.zeros_like(v[0], device=self.device), torch.zeros_like(v[0], device=self.device), 0]
                              for k, v in packet.items() if k != 'is_init'}
        ep_infos = []
        for t in range(self.steps):
            batch = self._normalize(packet)
            init = packet['is_init'].to(self.device)
            _, _, _, self.hidden = self.model.estimator(batch['obs'], self.hidden, init)
            priv, force, _, self.ema_hidden = self.alg.ema(batch['obs'], self.ema_hidden, init)
            batch['features'] = torch.cat((priv, force), -1)
            distribution = self.alg.distribution(batch)
            action = distribution.sample()  # RAW Gaussian sample/log-prob; env owns clipping/delay.
            value = self.alg.value(batch)
            if self.alg.phase == 'teacher' and not (first and t == 0):
                for key, (total, squared, _) in self._norm_moments.items():
                    raw = packet[key].to(self.device)
                    total.add_(raw.sum(0)); squared.add_(raw.square().sum(0))
                    self._norm_moments[key][2] += len(raw)
            result = self.env.step(action.to(self.env.device))
            reward, done, info = result[4].to(self.device), result[5].to(self.device).bool(), result[6]
            next_packet = self.env.get_facet_observations()
            successor_value = self.alg.value(self._normalize(next_packet))
            successor_value[done] = 0  # True failures never bootstrap, even when timeout coincides.
            terminal = self.env.get_facet_terminal_observations()
            if done.any():
                if terminal is None or not torch.equal(terminal['ids'].to(self.device), done.nonzero().flatten()):
                    raise RuntimeError('Missing/misaligned FACET terminal successor packet')
                ids = terminal['ids'].to(self.device)
                terminal_value = self.alg.value(self._normalize(terminal))
                successor_value[ids] = torch.where(terminal['timeouts'].to(self.device), terminal_value, 0.)
            storage.add(**batch, actions=action, means=distribution.mean, stds=distribution.stddev,
                values=value, log_probs=distribution.log_prob(action), rewards=reward,
                successor_values=successor_value, done=done, is_init=init,
                valid=torch.ones_like(done))
            self.running_lengths += 1
            self.lengths.extend(self.running_lengths[done].cpu().tolist())
            self.running_lengths[done] = 0
            # Reset recurrent state on both sides of collection, never carry a
            # terminal hidden state into the freshly reset observation.
            self.hidden[done] = 0
            self.ema_hidden[done] = 0
            if 'episode' in info and done.any():
                ep_infos.append(deepcopy(info['episode']))
            packet = next_packet
        storage.finish(self.alg.cfg['gamma'], self.alg.cfg['lam'])
        return storage, ep_infos

    @torch.no_grad()
    def _finish_normalization(self):
        # Freeze through collection AND all PPO/estimator epochs. Incorporate
        # each teacher observation once at the boundary for the NEXT rollout.
        # Only sufficient statistics are retained, not another full raw rollout.
        if self.alg.phase != 'teacher':
            return
        for key, norm in (('observation', self.model.obs_norm), ('privileged', self.model.priv_norm), ('wrench', self.model.wrench_norm)):
            total, squared, count = self._norm_moments[key]
            if count:
                norm.count.add_(count)
                weight = count/norm.count
                norm.sum.lerp_(total/count, weight)
                norm.ssq.lerp_(squared/count, weight)

    @staticmethod
    def _metric(infos, *names):
        values = [float(torch.as_tensor(info[name]).float().mean()) for info in infos
                  for name in [next((n for n in names if n in info), None)] if name is not None]
        return max(values, default=0.)

    def _schedules(self, iteration, infos):
        cfg = self.alg.cfg
        if infos:
            self.alg.update_entropy(self._metric(infos, 'rew_tracking_lin_vel_force_world'),
                self._metric(infos, 'rew_tracking_ang_vel'), terrain_entropy_progress(self.env.cfg.terrain,
                    self._metric(infos, 'terrain_level', 'terrain_level_mean'), cfg['adaptive_ent_ter_threshold']))
        if getattr(self.env, 'use_reward_curriculum', False):
            self.env.step_reward_curriculum(iteration)
        sim = self.env.simulator
        if getattr(sim, 'use_domainrand_curriculum', False) and any('rew_tracking_lin_vel_force_world' in x for x in infos):
            sim._step_domian_rand(iteration, self._metric(infos, 'rew_tracking_lin_vel_force_world'))
        return update_force_curriculum_from_rollout(self.env, iteration, infos,
            sum(self.lengths)/len(self.lengths) if self.lengths else None)

    @torch.no_grad()
    def maybe_transition(self):
        if self.alg.transitioned or self.current_learning_iteration < self.alg.cfg['student_start_iteration']:
            return False
        if not self.normalizers_initialized:
            # Explicit start-at-zero runs still initialize statistics BEFORE
            # freezing targets, not on their first student rollout.
            packet = self.env.get_facet_observations()
            self.model.update_normalizers(*(packet[k].to(self.device) for k in ('observation', 'privileged', 'wrench')))
            self.normalizers_initialized = True
        self.alg.switch_to_student()
        # EMA state has been advanced on EVERY teacher observation, not invented
        # or reset here. This query uses a copy and does not consume the next obs.
        packet = self.env.get_facet_observations()
        batch = self._normalize(packet)
        p, f, _, _ = self.alg.ema(batch['obs'], self.ema_hidden, packet['is_init'].to(self.device))
        target_p, target_f = self.model.priv_encoder(batch['priv']), self.model.force_encoder(batch['wrench'])
        teacher = self.model.teacher(batch['obs'], target_p, target_f).mean
        student = self.model.student(batch['obs'], p, f).mean
        self.last_metrics['FACET/handoff_action_rms_difference'] = float((teacher-student).square().mean().sqrt())
        self.last_metrics['FACET/handoff_ema_priv_mse'] = float((p-target_p).square().mean())
        self.last_metrics['FACET/handoff_ema_force_mse'] = float((f-target_f).square().mean())
        self.last_metrics['FACET/phase_student'] = 1.
        curriculum = getattr(self.env, '_staged_force_curriculum', None)
        if curriculum is not None:
            self.last_metrics.update(curriculum.metrics(self.current_learning_iteration))
        if self.log_dir:
            # Shared latest-checkpoint discovery matches filenames containing
            # "model"; keep the archival handoff name OUT of that candidate set.
            self.save(str(Path(self.log_dir, f'transition_{self.current_learning_iteration}.pt')))
            self.save(str(Path(self.log_dir, f'model_{self.current_learning_iteration}.pt')))
        return True

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Argument is the TOTAL global budget (including resumed completed updates)."""
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)
        if init_at_random_ep_len and self.current_learning_iteration == 0:
            self.env.episode_length_buf.random_(self.env.max_episode_length)
        self.model.train()
        self.maybe_transition()
        while self.current_learning_iteration < num_learning_iterations:
            iteration = self.current_learning_iteration
            self.env.set_training_iteration(iteration)
            start = time.perf_counter()
            storage, infos = self.collect()
            collected = time.perf_counter()
            metrics = self.alg.update(storage, iteration)
            self._finish_normalization()
            updated = time.perf_counter()
            metrics = {'Loss/'+k: v for k, v in metrics.items()}
            metrics.update(self._schedules(iteration, infos))
            for key in {k for info in infos for k in info}:
                values = [torch.as_tensor(info[key], device=self.device).float().mean() for info in infos if key in info]
                metrics['Episode/'+key] = float(torch.stack(values).mean())
            metrics.update({'FACET/phase_student': float(self.alg.phase == 'student'),
                'Perf/collection_seconds': collected-start, 'Perf/update_seconds': updated-collected,
                'Loss/learning_rate': self.alg.learning_rate, 'Values/entropy': self.alg.current_entropy_coef})
            self.current_learning_iteration += 1
            self.env.set_training_iteration(self.current_learning_iteration)
            self.last_metrics = metrics
            self.maybe_transition()  # Completed teacher update first; next rollout is wholly student.
            if self.writer:
                for key, value in self.last_metrics.items():
                    self.writer.add_scalar(key, value, self.current_learning_iteration)
                log_tradeoff_curriculum(self.writer, self.env, self.current_learning_iteration)
            print(f"FACET {self.current_learning_iteration}/{num_learning_iterations} {self.alg.phase}: "
                  f"rollout {collected-start:.3f}s, update {updated-collected:.3f}s", flush=True)
            if self.log_dir and self.current_learning_iteration % self.save_interval == 0:
                self.save(str(Path(self.log_dir, f'model_{self.current_learning_iteration}.pt')))
        if self.log_dir:
            self.save(str(Path(self.log_dir, f'model_{self.current_learning_iteration}.pt')))
        if self.writer:
            self.writer.flush()

    def save(self, path):
        env = self.env
        state = dict(version=1, algorithm=self.alg.state_dict(), iteration=self.current_learning_iteration,
            normalizers_initialized=self.normalizers_initialized, config=self.cfg, last_metrics=self.last_metrics,
            force_curriculum=staged_force_curriculum_state_dict(env), domain_curriculum=domain_rand_state(env.simulator),
            reward_scales=deepcopy(env.reward_scales), command_ranges=deepcopy(env.command_ranges),
            common_step_counter=env.common_step_counter, lengths=list(self.lengths))
        # No simulator state is saved; recurrent/history/episode state is NOT
        # restored into fresh episodes. Per-env tradeoff progress is a curriculum.
        state['tradeoff_step_ctr'] = getattr(env, 'tradeoff_step_ctr', None)
        torch.save(state, path)

    def load(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get('version') != 1:
            raise ValueError('Not a FACET Stage 5 checkpoint')
        self.alg.load_state_dict(state['algorithm'], load_optimizer)
        self.current_learning_iteration = int(state['iteration'])
        self.normalizers_initialized = state['normalizers_initialized']
        self.last_metrics = state['last_metrics']
        load_domain_rand_state(self.env.simulator, state['domain_curriculum'])
        load_staged_force_curriculum_state_dict(self.env, state['force_curriculum'])
        self.env.reward_scales.update(state['reward_scales'])
        self.env.command_ranges.update(state['command_ranges'])
        self.env.set_training_iteration(self.current_learning_iteration)
        self.env.reset_idx(self.env.all_env_ids)
        self.env.common_step_counter = state['common_step_counter']
        if state['tradeoff_step_ctr'] is not None:
            if state['tradeoff_step_ctr'].shape != self.env.tradeoff_step_ctr.shape:
                raise ValueError('Changing env count on resume cannot preserve per-env tradeoff curriculum')
            self.env.tradeoff_step_ctr.copy_(state['tradeoff_step_ctr'])
        self.hidden.zero_(); self.ema_hidden.zero_(); self.running_lengths.zero_()
        self.lengths = deque(state['lengths'], maxlen=100)
        return self.current_learning_iteration

    def get_inference_policy(self, device=None):
        """Student packet API: callable(observation, hidden, is_init) -> action, hidden."""
        if device is not None and torch.device(device) != torch.device(self.device):
            raise ValueError('Construct the FACET runner on the requested device')
        @torch.no_grad()
        def infer(observation, hidden, is_init):
            obs = self.model.obs_norm(observation)
            p, f, _, h = self.alg.ema(obs, hidden, is_init)
            return self.model.student(obs, p, f).mean, h
        return infer
