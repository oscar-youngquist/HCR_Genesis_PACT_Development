"""BR collection/update/phase/checkpoint orchestration; no simulator-specific code."""
from pathlib import Path
import json
import math
import random
import numpy as np
import torch

from rsl_rl.beyond_robustness_config import BeyondRobustnessConfig
from rsl_rl.beyond_robustness_env import BeyondRobustnessEnv, BRObservation
from rsl_rl.modules.actor_critic_beyond_robustness import ActorCriticBeyondRobustness
from rsl_rl.algorithms.ppo_beyond_robustness import PPOBeyondRobustness


class BeyondRobustnessRunner:
    checkpoint_version = 1

    def __init__(self, env: BeyondRobustnessEnv, train_cfg, log_dir=None, device="cpu", *, writer=None):
        self.env = env
        self.cfg = train_cfg if isinstance(train_cfg, BeyondRobustnessConfig) else BeyondRobustnessConfig(**train_cfg)
        self.device = torch.device(device)
        if env.num_envs != self.cfg.num_envs or torch.device(env.device) != self.device:
            raise ValueError("BR environment count/device differs from resolved runner config")
        if env.input_schema != self.cfg.input_schema:
            raise ValueError("BR environment input schema differs from model/storage schema")
        model = ActorCriticBeyondRobustness(self.cfg)
        self.alg = PPOBeyondRobustness(model, self.cfg, device=self.device)
        self.current = None
        self.interactions = 0
        self.control_steps = 0
        self.episode_returns = torch.zeros(env.num_envs, device=self.device)
        self.episode_lengths = torch.zeros_like(self.episode_returns)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = writer
        self._owns_writer = False
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if writer is None:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
                self._owns_writer = True
            (self.log_dir / "beyond_robustness_resolved_config.json").write_text(
                json.dumps(dict(config=self.cfg.resolved(), input_contract=model.input_contract(),
                    environment=(env.resolved_configuration() if hasattr(env,'resolved_configuration') else None)), indent=2))
            if hasattr(env,'configuration_manifest'):
                (self.log_dir/'beyond_robustness_inheritance.json').write_text(json.dumps(env.configuration_manifest,indent=2))
            if getattr(env,'rewards',None) is not None:
                (self.log_dir/'beyond_robustness_rewards.json').write_text(json.dumps(env.rewards.spec,indent=2))

    @property
    def current_learning_iteration(self):
        """Absolute index of the next rollout; algorithm increments after each update."""
        return self.alg.iteration

    def _validate(self, observation):
        if not isinstance(observation, BRObservation):
            raise TypeError("Environment must return BRObservation, not a legacy observation stack")
        observation.validate(self.cfg.input_schema, self.env.num_envs, self.device)

    def set_training_phase(self, phase):
        self.alg.set_training_phase(phase)

    def _fresh_environment(self):
        self.current = self.env.reset()
        self._validate(self.current)
        self.alg.clear_history()
        self.episode_returns.zero_()
        self.episode_lengths.zero_()

    @torch.no_grad()
    def collect_rollout(self):
        if self.alg.storage.step or self.alg.pending is not None:
            raise RuntimeError("Complete update before collecting another rollout")
        self.alg.apply_phase_schedule()
        if hasattr(self.env, 'begin_iteration'):
            self.env.begin_iteration(self.alg.iteration)
        if self.current is None:
            self._fresh_environment()
        else:
            # Pure refresh includes any reset-property/curriculum changes made at
            # the iteration boundary. Exact resume must see the same current labels.
            self.current = self.env.observe()
            self._validate(self.current)
        completed_returns = torch.zeros((), device=self.device)
        completed_lengths = torch.zeros_like(completed_returns)
        completed_count = torch.zeros_like(completed_returns)
        reward_sum = torch.zeros_like(completed_returns)
        performance = {}
        tracking_sum = torch.zeros((), device=self.device)
        tracking_count = torch.zeros((), device=self.device, dtype=torch.long)
        for _ in range(self.cfg.num_steps_per_env):
            # Both labels and validity describe t, before any in-place simulator reset.
            valid = self.current.supervision_valid.detach().clone()
            actions = self.alg.act(*self.current.inputs())
            step = self.env.step(actions)
            self._validate(step.observation)
            for name in ("rewards", "dones", "timeouts"):
                value = getattr(step, name)
                if value.shape != (self.env.num_envs,) or value.device != self.device:
                    raise ValueError(f"{name} must be device-local [N]")
            if step.dones.dtype != torch.bool or step.timeouts.dtype != torch.bool:
                raise ValueError("done/timeout masks must be boolean")
            if (step.timeouts & ~step.dones).any():
                raise ValueError("Timeouts must also be done")
            terminal_values = None
            if step.timeouts.any():
                if step.terminal is None or step.terminal_valid is None:
                    raise ValueError("Timeout requires explicit pre-reset terminal observations")
                self._validate(step.terminal)
                mask = step.terminal_valid
                if mask.shape != step.timeouts.shape or mask.dtype != torch.bool or mask.device != self.device:
                    raise ValueError("terminal_valid must be a device-local bool [N] mask")
                if (step.timeouts & ~mask).any():
                    raise ValueError("Missing pre-reset terminal rows for timeout bootstrap")
                ids = step.timeouts.nonzero(as_tuple=False).flatten()
                terminal = step.terminal.select(ids)
                # Only evaluate selected terminal rows; unrelated entries may be NaN.
                values = self.alg.bootstrap_values(*terminal.inputs(), env_ids=ids)
                terminal_values = torch.zeros_like(step.rewards)
                terminal_values[ids] = values.squeeze(-1)
            self.alg.process_env_step(step.rewards, step.dones, timeouts=step.timeouts,
                terminal_values=terminal_values, supervision_valid=valid)
            self.current = step.observation
            self.episode_returns += step.rewards
            self.episode_lengths += 1
            completed_returns += self.episode_returns[step.dones].sum()
            completed_lengths += self.episode_lengths[step.dones].sum()
            completed_count += step.dones.sum()
            self.episode_returns[step.dones] = 0
            self.episode_lengths[step.dones] = 0
            reward_sum += step.rewards.sum()
            self.interactions += self.env.num_envs
            self.control_steps += 1
            if step.curriculum_tracking is not None:
                tracking = step.curriculum_tracking.detach()
                if tracking.shape != step.rewards.shape or tracking.device != self.device:
                    raise ValueError('curriculum_tracking must be device-local [N]')
                finite = torch.isfinite(tracking)
                tracking_sum += tracking[finite].sum()
                tracking_count += finite.sum()
            if step.performance_metrics:
                keys = ("lin_vel_tracking", "ang_vel_tracking", "terrain_level")
                # One small set of episode summaries; same max-over-rollout convention
                # as HardPACT, no policy forwards or dependency on curriculum stage.
                if all(k in step.performance_metrics and math.isfinite(float(step.performance_metrics[k])) for k in keys):
                    for key in keys:
                        performance[key] = max(performance.get(key, 0.), float(step.performance_metrics[key]))
        self.alg.compute_returns(*self.current.inputs())
        count = completed_count.item()
        self._curriculum_evidence = (tracking_sum.item(), tracking_count.item())
        return dict(reward_mean=(reward_sum / (self.env.num_envs * self.cfg.num_steps_per_env)).item(),
                    episodes=count, episode_return=completed_returns.item() / count if count else float("nan"),
                    episode_length_steps=completed_lengths.item() / count if count else float("nan")), performance

    def learn(self, num_learning_iterations=None, *, save_interval=None):
        """Run a bounded number of additional iterations; no implicit curriculum steps."""
        if num_learning_iterations is None: num_learning_iterations=max(0,self.cfg.max_iterations-self.alg.iteration)
        if save_interval is None: save_interval=self.cfg.save_interval if self.log_dir is not None else 0
        if num_learning_iterations < 0 or save_interval < 0:
            raise ValueError("Iteration counts must be nonnegative")
        self.alg.train_mode()
        logs = []
        for _ in range(num_learning_iterations):
            iteration = self.alg.iteration
            collection, performance = self.collect_rollout()
            phase = self.alg.training_phase
            entropy_used = self.alg.entropy_coef
            losses = self.alg.update()
            if hasattr(self.env, 'end_iteration'):
                total, count = self._curriculum_evidence
                self.env.end_iteration(iteration, total/count if count else None, count)
            entropy_updated = self.alg.update_adaptive_entropy_coef(performance)
            error_keys = ("load_mae/", "load_rmse/")
            metrics = {f"loss/{key}": value for key, value in losses.items() if not key.startswith(error_keys)}
            metrics.update({f"load/{key}": value for key, value in losses.items() if key.startswith(error_keys)})
            metrics.update({f"train/{key}": value for key, value in collection.items()})
            if hasattr(self.env,'iteration_metrics'):
                metrics.update(self.env.iteration_metrics())
            metrics.update({"train/learning_rate": self.alg.learning_rate,
                            "train/entropy_coefficient": entropy_used,
                            "train/entropy_coefficient_next": self.alg.entropy_coef,
                            "train/entropy_schedule_updated": float(entropy_updated),
                            "train/phase_student_reinforcement": float(phase == "student_reinforcement"),
                            "train/interactions": self.interactions, "train/control_steps": self.control_steps})
            if self.writer is not None:
                for name, value in metrics.items(): self.writer.add_scalar(name, value, iteration)
                self.writer.add_text("train/training_phase", phase, iteration)
                self.writer.flush()
            logs.append(metrics)
            if save_interval and self.alg.iteration % save_interval == 0:
                if self.log_dir is None:
                    raise ValueError("Automatic checkpointing requires log_dir")
                self.save(self.log_dir / f"model_{self.alg.iteration}.pt")
        return logs

    def get_inference_policy(self):
        """Stateless student callable (observations, explicit history); no teacher inputs."""
        return self.alg.actor_critic.act_inference

    def save(self, path):
        algorithm = self.alg.checkpoint_state()  # also rejects partial iterations
        env_state = self.env.state_dict() if hasattr(self.env, "state_dict") and self.current is not None else None
        curriculum = self.env.curriculum_state_dict() if hasattr(self.env, "curriculum_state_dict") else None
        torch.save(dict(runner_schema_version=self.checkpoint_version, algorithm=algorithm,
                        environment_state=env_state, curriculum_state=curriculum,
                        exact_environment=bool(getattr(self.env, "supports_exact_resume", False) and env_state is not None),
                        interactions=self.interactions, control_steps=self.control_steps,
                        episode_returns=self.episode_returns, episode_lengths=self.episode_lengths,
                        python_rng=random.getstate(), numpy_rng=np.random.get_state()), path)

    def load(self, path, *, resume_mode="fresh"):
        """fresh keeps learned/optimizer/curriculum progress, but resets physical state/history.

        exact fails explicitly without a complete environment-state implementation.
        Configuration, schema, schedule and active phase are validated/restored by PPO.
        """
        if resume_mode not in ("fresh", "exact"):
            raise ValueError("resume_mode must be 'fresh' or 'exact'")
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data.get("runner_schema_version") != self.checkpoint_version:
            raise ValueError("Incompatible BR runner checkpoint; core-only checkpoints lack environment state")
        exact = resume_mode == "exact"
        if exact and (not data["exact_environment"] or not getattr(self.env, "supports_exact_resume", False)
                      or not hasattr(self.env, "load_state_dict")):
            raise ValueError("Exact continuation requires a complete saved/restorable environment state")
        if data["curriculum_state"] is not None and not hasattr(self.env, "load_curriculum_state_dict"):
            raise ValueError("Checkpoint curriculum state cannot be restored by this environment")
        self.alg.load_checkpoint_state(data["algorithm"], restore_rng=False, restore_history=exact)
        if data["curriculum_state"] is not None:
            self.env.load_curriculum_state_dict(data["curriculum_state"])
        if exact:
            self.env.load_state_dict(data["environment_state"])
            self.current = self.env.observe()
            self._validate(self.current)
            self.episode_returns.copy_(data["episode_returns"])
            self.episode_lengths.copy_(data["episode_lengths"])
        # Restore global RNG AFTER environment restore; fresh reset then consumes it normally.
        random.setstate(data["python_rng"])
        np.random.set_state(data["numpy_rng"])
        torch.set_rng_state(data["algorithm"]["torch_rng"].cpu())
        if data["algorithm"]["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(data["algorithm"]["cuda_rng"].cpu(), self.device)
        if not exact: self._fresh_environment()
        self.interactions, self.control_steps = data["interactions"], data["control_steps"]
        return self.alg.iteration

    def close(self):
        if self._owns_writer:
            self.writer.close()
