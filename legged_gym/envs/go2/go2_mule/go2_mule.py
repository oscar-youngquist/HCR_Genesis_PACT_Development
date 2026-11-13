"""Go2 MULE environment boundary. Stage 1 provides no physical implementation.

Future rolling-ball adapter implements this class, not BR's sliding box or
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


class Go2MULE(ABC):
    """Abstract until physical integration, deliberately not registered.

    Adapter owns the action-delay queue and reports applied source identity.
    Failures take precedence over timeouts. Bootstrap uses terminal snapshots,
    never reset observations or a history-mutating action evaluation.
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
