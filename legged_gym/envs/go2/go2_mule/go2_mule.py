"""Go2 MULE learning/environment boundary.

The isolated IsaacLabMULEEnv implements this class, not BR's sliding box or
scripted payload wrenches. observe() is read-only (no RNG/history/physics).
Histories advance once per action and clear per reset on the learning side.
Physical stage and optimization phase are independent; no scheduler here.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from .mule_schema import FootForceEstimate, MULECommand, MULEExecution


@dataclass(frozen=True)
class MULEObservation:
    actor: torch.Tensor  # packed deployable 45-D observation at t
    critic: torch.Tensor  # named privileged state, never fed to actors
    velocity_b: torch.Tensor  # clean physical auxiliary label at t
    foot_force_estimate: FootForceEstimate  # ONLY applied torque + leg kinematics
    episode_id: torch.Tensor
    supervision_valid: torch.Tensor
    timestamp: int

    def snapshot(self):
        copy=lambda x:x.detach().clone()
        return MULEObservation(copy(self.actor),copy(self.critic),copy(self.velocity_b),
            FootForceEstimate(copy(self.foot_force_estimate.force_b),copy(self.foot_force_estimate.valid)),
            copy(self.episode_id),copy(self.supervision_valid),self.timestamp)


@dataclass(frozen=True)
class MULEStep:
    observation: MULEObservation  # next state, potentially auto-reset
    execution: MULEExecution  # actual delay-selected target, not raw/issued action
    nominal_reward: torch.Tensor
    adaptive_reward: torch.Tensor
    terminated: torch.Tensor
    timed_out: torch.Tensor
    terminal: MULEObservation | None  # owned pre-reset inputs for timeout bootstrap
    terminal_valid: torch.Tensor
    # Optional shared, unscaled performance evidence for entropy scheduling.
    # All three keys are required; missing/nonfinite evidence holds the schedule.
    performance_metrics: dict | None = None
    # Logging-only pre-reset masks, plus ALREADY weighted/dt-scaled terms by
    # reward stream: {'nominal': {'tracking_lin_vel': tensor[N]}, 'adaptive': ...}.
    # Future physical adapters supply these; never manufacture unavailable data.
    failure_reasons: dict | None = None
    reward_terms: dict | None = None


class Go2MULE(ABC):
    """Abstract boundary; concrete rolling adapter deliberately not registered.

    Adapter owns the action-delay queue and reports applied source identity.
    Failures take precedence over timeouts. Bootstrap uses terminal snapshots,
    never reset observations or a history-mutating action evaluation.

    step() advances timestamp by one, auto-resets done rows, and changes only
    their episode IDs. terminal carries owned pre-reset rows at that timestamp.
    Optional begin_iteration(iteration)/end_iteration(iteration, performance)
    hooks connect future shared curricula; no payload scheduler lives here.
    Exact-resume adapters must implement state_dict, validate_state_dict and
    load_state_dict and declare supports_exact_resume. Fresh physical resume
    can separately restore validated curriculum_state_dict without physics.
    Physical adapters expose max_episode_length_s for HardPACT-compatible
    Episode/rew_* normalization, and optional canonical iteration_metrics().
    Retain the shared runner logging contract when adding rolling payloads.
    """
    def __init__(self, cfg):
        self.cfg=cfg
        self.num_envs=cfg.num_envs
        self.input_schema=cfg.schema

    @abstractmethod
    def observe(self) -> MULEObservation: ...

    @abstractmethod
    def reset(self, env_ids=None) -> MULEObservation: ...

    @abstractmethod
    def step(self, command: MULECommand) -> MULEStep: ...
