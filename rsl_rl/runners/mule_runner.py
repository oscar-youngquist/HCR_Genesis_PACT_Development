"""Simulator-neutral MULE learning; physical stage scheduling belongs to the adapter."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import random
import time
import numpy as np
import torch
from legged_gym.envs.go2.go2_mule.go2_mule import Go2MULE, MULEObservation
from legged_gym.envs.go2.go2_mule.mule_schema import MULEHistory
from rsl_rl.modules.actor_critic_mule import ActorCriticMULE
from rsl_rl.algorithms.ppo_mule import PPOMULE
from rsl_rl.utils.learning_diagnostics import (LearningEpisodeDiagnostics, policy_diagnostics,
    compatible_training_metrics, format_training_summary)


class MULERunner:
    def __init__(self, env, config, device='cpu', log_dir=None, writer=None, phase_transition_hook=None, console=True):
        if not isinstance(env, Go2MULE): raise TypeError('Expected the Go2MULE adapter interface')
        if env.input_schema != config.schema or env.num_envs != config.num_envs:
            raise ValueError('Environment/learning schema or environment count differs')
        for key in ('control_dt', 'action_scale', 'clip_actions'):
            if getattr(env.cfg, key) != getattr(config, key): raise ValueError(f'Environment {key} differs')
        self.env, self.config, self.device = env, config, torch.device(device)
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        self.alg = PPOMULE(ActorCriticMULE(config), config, self.device)
        self.history = MULEHistory(config.schema, config.num_envs, self.device)
        self.standing_position = torch.tensor(env.cfg.standing_position, device=self.device)
        self.current = None
        self.interactions, self.control_steps = 0, 0
        self.episode_returns = torch.zeros(config.num_envs, 2, device=self.device)
        self.episode_lengths = torch.zeros(config.num_envs, dtype=torch.long, device=self.device)
        self.episode_diagnostics=LearningEpisodeDiagnostics(config.num_envs,self.device,config.control_dt,
            getattr(env,'max_episode_length_s',None),streams=('nominal','adaptive'))
        self.console,self.total_time,self.last_logged_metrics=console,0.,{}
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.writer = writer
        self._owns_writer=False
        # Optional future curriculum callback returns an optimization phase, not a payload schedule.
        self.phase_transition_hook = phase_transition_hook
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if writer is None:
                from torch.utils.tensorboard import SummaryWriter
                self.writer=SummaryWriter(str(self.log_dir)); self._owns_writer=True
            manifest = dict(environment=asdict(env.cfg), training=config.resolved(), model_contract=config.contract())
            (self.log_dir/'mule_configuration.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))

    def _validate_observation(self, observation, *, terminal=False):
        if not isinstance(observation, MULEObservation): raise TypeError('Expected MULEObservation')
        n, s = self.config.num_envs, self.config.schema
        for key, width in (('actor', 45), ('critic', s.critic_dim), ('velocity_b', 3)):
            x = getattr(observation, key)
            if x.shape != (n, width) or x.dtype != torch.float32 or x.device != self.device:
                raise ValueError(f'Invalid {key} observation')
            if not terminal and key != 'velocity_b' and not torch.isfinite(x).all():
                raise ValueError(f'Nonfinite live {key}')
        for key, dtype in (('episode_id', torch.long), ('supervision_valid', torch.bool)):
            x = getattr(observation, key)
            if x.shape != (n,) or x.dtype != dtype or x.device != self.device: raise ValueError(f'Invalid {key}')
        f = observation.foot_force_estimate
        if (f.force_b.shape != (n, 4, 3) or f.valid.shape != (n, 4) or f.valid.dtype != torch.bool
                or f.force_b.device != self.device or f.valid.device != self.device):
            raise ValueError('Invalid deployable force estimate')
        if not terminal and not torch.isfinite(torch.where(f.valid[..., None], f.force_b, 0.)).all():
            raise ValueError('Nonfinite valid force estimate')
        if not isinstance(observation.timestamp, int): raise ValueError('Expected integer control timestamp')

    def _fresh_environment(self):
        self.current = self.env.reset().snapshot()
        self._validate_observation(self.current)
        self.history = MULEHistory(self.config.schema, self.config.num_envs, self.device)
        self.episode_returns.zero_(); self.episode_lengths.zero_()
        self.episode_diagnostics.clear_episodes()

    def transition_to_joint_adaptation(self):
        """Explicit clean-boundary hook; does not change physical environment stage."""
        self.alg.transition_to_joint_adaptation()

    @torch.no_grad()
    def collect_step(self):
        """Collect one timestamp, also exposed for bounded/debug collectors."""
        if self.alg.pending is not None or self.alg.storage.full: raise RuntimeError('Consume pending rollout first')
        if self.current is None: self._fresh_environment()
        o = self.current.snapshot()
        self.history.append(o.actor, o.foot_force_estimate, o.episode_id, o.timestamp)
        h, ah = self.history.snapshot()
        command = self.alg.act(o, h, ah, self.standing_position)
        step = self.env.step(command)
        next_o = step.observation.snapshot()
        self._validate_observation(next_o)
        n = self.config.num_envs
        for key in ('terminated', 'timed_out', 'terminal_valid'):
            value = getattr(step, key)
            if value.shape != (n,) or value.dtype != torch.bool or value.device != self.device:
                raise ValueError(f'Invalid {key}')
        done = step.terminated | step.timed_out
        timeout = step.timed_out & ~step.terminated
        if next_o.timestamp != o.timestamp+1: raise ValueError('Expected exactly one control timestamp per action')
        if not torch.equal(next_o.episode_id != o.episode_id, done):
            raise ValueError('Auto-reset episode IDs must change exactly on termination/timeout')
        terminal = None if step.terminal is None else step.terminal.snapshot()
        terminal_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        if terminal is not None:
            self._validate_observation(terminal, terminal=True)
            if terminal.timestamp != next_o.timestamp: raise ValueError('Misaligned terminal timestamp')
            terminal_valid = done & step.terminal_valid & (terminal.episode_id == o.episode_id)
        if torch.any(timeout & ~terminal_valid): raise ValueError('Timeout requires same-episode terminal observation')
        terminal_values = torch.zeros(n, 2, device=self.device)
        if torch.any(timeout):
            critic = terminal.critic[timeout]
            if not torch.isfinite(critic).all(): raise ValueError('Nonfinite timeout critic inputs')
            terminal_values[timeout] = torch.stack(self.alg.model.values(critic), -1)
        # Auxiliary pair: history/velocity at t and same-episode observation at t+1.
        target = next_o.actor.clone()
        target_episode = next_o.episode_id.clone()
        target_valid = next_o.supervision_valid & ~done
        if terminal is not None:
            target[terminal_valid] = terminal.actor[terminal_valid]
            target_episode[terminal_valid] = terminal.episode_id[terminal_valid]
            target_valid[terminal_valid] = terminal.supervision_valid[terminal_valid]
        auxiliary_valid = o.supervision_valid & target_valid & (target_episode == o.episode_id)
        rewards = torch.stack((step.nominal_reward, step.adaptive_reward), -1)
        if rewards.shape != (n, 2) or not torch.isfinite(rewards).all(): raise ValueError('Invalid reward streams')
        ex = step.execution
        self.alg.record_outcome(rewards=rewards, terminal_values=terminal_values,
            terminated=step.terminated, timed_out=step.timed_out, terminal_valid=terminal_valid,
            auxiliary_valid=auxiliary_valid, next_observation=target, next_episode_id=target_episode,
            next_timestamp=torch.full_like(o.episode_id, next_o.timestamp),
            executed_target=ex.position_target, applied_torque=ex.applied_torque_nm,
            executed_source_id=ex.source_transition_id)
        self.current = next_o
        self.interactions += n; self.control_steps += 1
        self.episode_returns += rewards; self.episode_lengths += 1
        terms=step.reward_terms or {}
        if set(terms)-{'nominal','adaptive'}: raise ValueError('Diagnostic terms require named MULE reward streams')
        terms={owner+'/'+name:value for owner in self.alg.active_owners for name,value in terms.get(owner,{}).items()}
        self.episode_diagnostics.record(done,timeout,self.episode_lengths,failure=step.terminated,
            reasons=step.failure_reasons,reward_terms=terms,raw_timeout=step.timed_out,returns=self.episode_returns)
        completed_returns = self.episode_returns[done].clone()
        completed_lengths = self.episode_lengths[done].clone()
        self.episode_returns[done] = 0.; self.episode_lengths[done] = 0
        self.history.observations[done] = 0.; self.history.adaptive[done] = 0.
        self.history.episode_ids[done] = -1
        return rewards, completed_returns, completed_lengths, step.performance_metrics

    def collect_rollout(self):
        self.alg.require_boundary()
        self.episode_diagnostics.begin_iteration()
        if self.phase_transition_hook is not None:
            phase = self.phase_transition_hook(self.alg.iteration, self.env)
            if phase == 'joint_adaptation': self.transition_to_joint_adaptation()
            elif phase not in (None, self.alg.model.phase): raise ValueError('Phase hook cannot reverse optimization')
        if hasattr(self.env, 'begin_iteration'): self.env.begin_iteration(self.alg.iteration)
        if self.current is not None:
            refreshed = self.env.observe().snapshot()
            self._validate_observation(refreshed)
            if refreshed.timestamp != self.current.timestamp or not torch.equal(refreshed.episode_id, self.current.episode_id):
                raise ValueError('observe() advanced physics/reset/history')
            self.current = refreshed
        rewards, returns, lengths, performance = [], [], [], []
        for _ in range(self.config.num_steps_per_env):
            r, er, el, evidence = self.collect_step()
            rewards.append(r); returns.append(er); lengths.append(el)
            keys = ('lin_vel_tracking', 'ang_vel_tracking', 'terrain_level')
            if evidence is not None and all(k in evidence and math.isfinite(float(evidence[k])) for k in keys):
                performance.append({k: float(evidence[k]) for k in keys})
        self.alg.compute_returns(self.current.critic)  # Critic only: no history append or action/context sampling.
        reward = torch.stack(rewards).mean((0, 1))
        completed = torch.cat(lengths)
        episode = torch.cat(returns)
        metrics = dict(nominal_reward=float(reward[0]), adaptive_reward=float(reward[1]), episodes=len(completed))
        if len(completed):
            metrics.update(episode_length_steps=float(completed.float().mean()),
                nominal_episode_return=float(episode[:, 0].mean()), adaptive_episode_return=float(episode[:, 1].mean()))
        # Shared method-neutral performance-gap entropy; full-rollout finite evidence only.
        evidence = ({k: sum(p[k] for p in performance)/len(performance) for k in keys}
                    if len(performance) == self.config.num_steps_per_env else None)
        return metrics, evidence

    def learn(self, iterations=None):
        """iterations is ADDITIONAL; None runs to the configured absolute budget."""
        if iterations is None: iterations = max(0, self.config.max_iterations-self.alg.iteration)
        if not isinstance(iterations, int) or iterations < 0: raise ValueError('Expected nonnegative iteration count')
        logs = []
        for _ in range(iterations):
            started = time.perf_counter()
            collection, performance = self.collect_rollout()
            collected = time.perf_counter()
            losses = self.alg.update()
            updated = time.perf_counter()
            self.alg.update_entropy(performance)
            if hasattr(self.env, 'end_iteration'): self.env.end_iteration(self.alg.iteration-1, performance)
            metrics = {f'loss/{k}': v for k, v in losses.items()}
            metrics.update({f'train/{k}': v for k, v in collection.items()})
            metrics.update({'train/iteration': self.alg.iteration, 'train/interactions': self.interactions,
                'train/joint_adaptation': int(self.alg.model.phase == 'joint_adaptation'),
                'train/collection_s': collected-started, 'train/update_s': updated-collected})
            for owner in self.alg.active_owners:
                for key, value in self.alg.schedulers[owner].items(): metrics[f'train/{owner}/{key}'] = value
            self.total_time+=updated-started
            published=compatible_training_metrics(metrics,collected-started,updated-collected,
                self.config.num_envs*self.config.num_steps_per_env,streams=self.alg.active_owners)
            published.update(self.episode_diagnostics.metrics(active_streams=self.alg.active_owners))
            stds=[]
            for owner in self.alg.active_owners:
                std=getattr(self.alg.model,owner+'_actor').std
                stds.append(std.detach())
                published.update(policy_diagnostics(std,self.config.schema.joint_order,prefix='Policy/'+owner))
            published.update(policy_diagnostics(torch.stack(stds).mean(0),self.config.schema.joint_order))
            published.update({'Perf/total_time':self.total_time,'Perf/total_timesteps':self.interactions,
                'Train/phase_joint_adaptation':int(self.alg.model.phase=='joint_adaptation')})
            if hasattr(self.env,'iteration_metrics'): published.update(self.env.iteration_metrics())
            self.last_logged_metrics=published
            if self.writer is not None:
                for key, value in published.items(): self.writer.add_scalar(key, value, self.alg.iteration-1)
                self.writer.flush()
            if self.console:
                print(format_training_summary('MULE',self.alg.iteration,self.alg.model.phase,published,self.config.max_iterations),flush=True)
            logs.append(metrics)
            if self.log_dir is not None and self.alg.iteration % self.config.save_interval == 0:
                self.save(self.log_dir/f'model_{self.alg.iteration}.pt')
        if iterations and self.log_dir is not None:
            self.save(self.log_dir/f'model_{self.alg.iteration}.pt')
        return logs

    def get_inference_policy(self): return self.alg.model.inference

    def close(self):
        if self._owns_writer: self.writer.close()

    def save(self, path):
        algorithm = self.alg.state_dict()
        exact = bool(getattr(self.env, 'supports_exact_resume', False) and self.current is not None)
        if exact and not all(hasattr(self.env, name) for name in ('state_dict', 'validate_state_dict', 'load_state_dict')):
            raise ValueError('Exact environments must implement state validation/restoration')
        curriculum = self.env.curriculum_state_dict() if hasattr(self.env, 'curriculum_state_dict') else None
        torch.save(dict(version=1, device=str(self.device), algorithm=algorithm, environment_contract=asdict(self.env.cfg),
            environment_state=self.env.state_dict() if exact else None, exact_environment=exact,
            curriculum=curriculum, interactions=self.interactions, control_steps=self.control_steps,
            episode_returns=self.episode_returns, episode_lengths=self.episode_lengths,
            episode_diagnostics=self.episode_diagnostics.state_dict(),total_time=self.total_time,
            current=None if self.current is None else self.current.snapshot(),
            history=dict(observations=self.history.observations, adaptive=self.history.adaptive,
                         episode_ids=self.history.episode_ids, timestamp=self.history.timestamp),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None), path)

    def load(self, path, *, resume_mode='fresh'):
        """Trusted local checkpoints only. Fresh physics resets history, not learning progress."""
        self.alg.require_boundary()
        if resume_mode not in ('fresh', 'exact'): raise ValueError('Expected fresh or exact resume')
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data.get('version') != 1 or data.get('environment_contract') != asdict(self.env.cfg):
            raise ValueError('Incompatible MULE runner/environment contract')
        self.alg.validate_state_dict(data['algorithm'])
        self.episode_diagnostics.validate_state_dict(data.get('episode_diagnostics'))
        exact = resume_mode == 'exact'
        if exact:
            if data['device'] != str(self.device): raise ValueError('Exact resume requires the same device')
            if not data['exact_environment'] or not getattr(self.env, 'supports_exact_resume', False):
                raise ValueError('Exact continuation requires a fully restorable environment')
            self.env.validate_state_dict(data['environment_state'])
            self._validate_observation(data['current'])
        for key in ('observations', 'adaptive', 'episode_ids'):
            x, expected = data['history'][key], getattr(self.history, key)
            if x.shape != expected.shape or x.dtype != expected.dtype: raise ValueError('Incompatible saved history')
        if not isinstance(data['history']['timestamp'], int): raise ValueError('Invalid history timestamp')
        if exact and data['history']['timestamp'] >= data['current'].timestamp:
            raise ValueError('Saved current observation was already consumed into history')
        for key in ('episode_returns', 'episode_lengths'):
            if data[key].shape != getattr(self, key).shape: raise ValueError('Invalid saved episode state')
        if any(not isinstance(data[k], int) or data[k] < 0 for k in ('interactions', 'control_steps')):
            raise ValueError('Invalid interaction counters')
        if data['curriculum'] is not None:
            if not all(hasattr(self.env, k) for k in ('validate_curriculum_state_dict', 'load_curriculum_state_dict')):
                raise ValueError('Adapter cannot validate/restore curriculum state')
            self.env.validate_curriculum_state_dict(data['curriculum'])
        # Validate RNG states on private generators, before any live mutation.
        random.Random().setstate(data['python_rng'])
        np.random.RandomState().set_state(data['numpy_rng'])
        torch.Generator().set_state(data['torch_rng'].cpu())
        if data['cuda_rng'] is not None:
            if len(data['cuda_rng']) != torch.cuda.device_count(): raise ValueError('CUDA RNG device count differs')
            for i, state in enumerate(data['cuda_rng']): torch.Generator(device=f'cuda:{i}').set_state(state.cpu())
        self.alg.load_state_dict(data['algorithm'])
        if data['curriculum'] is not None: self.env.load_curriculum_state_dict(data['curriculum'])
        self.interactions, self.control_steps = data['interactions'], data['control_steps']
        self.total_time=float(data.get('total_time',0.))
        if exact:
            self.env.load_state_dict(data['environment_state'])
            self.current = data['current'].snapshot()
            for key in ('observations', 'adaptive', 'episode_ids'): getattr(self.history, key).copy_(data['history'][key])
            self.history.timestamp = data['history']['timestamp']
            self.episode_returns.copy_(data['episode_returns']); self.episode_lengths.copy_(data['episode_lengths'])
            self.episode_diagnostics.load_state_dict(data.get('episode_diagnostics'))
        random.setstate(data['python_rng']); np.random.set_state(data['numpy_rng'])
        torch.set_rng_state(data['torch_rng'].cpu())
        if data['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in data['cuda_rng']])
        if not exact: self._fresh_environment()
