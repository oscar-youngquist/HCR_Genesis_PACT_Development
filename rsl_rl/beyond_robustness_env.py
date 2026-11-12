"""Simulator-neutral BR runner boundary. No payload/reward implementation.

reset()/observe() return current t inputs. step(raw_position_actions) auto-resets
done rows, returning t+1 inputs AND owned pre-reset terminal inputs for timeouts.
supervision_valid belongs to the CURRENT observation's labels, not the next step.
Observations may alias simulator buffers: the runner/algorithm snapshot before step.
All floating inputs are float32 on the runner device. Packed native state fields
are physical units (no HardPACT observation scaling). Relative load velocity means
the time derivative of body-relative CoM position, including body rotation; the
physical adapter must verify this convention and actual link/joint field mappings.

Optional exact continuation: supports_exact_resume=True, state_dict/load_state_dict
must cover physical state, command/action history, random generators, reset state
and curricula. Merely restoring curriculum state is NOT exact continuation.
Optional curriculum_state_dict/load_curriculum_state_dict allows fresh-state resume
at the saved curriculum progress. Learning phase never chooses that progress.
Stages 1/2 of the future box curriculum can therefore retain teacher_student;
there is no automatic mode change unless explicitly configured by PPO iteration.
"""
from dataclasses import dataclass, fields
from typing import Protocol
import torch
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema


@dataclass
class BRObservation:
    observations: torch.Tensor
    state: torch.Tensor
    parameters: torch.Tensor
    load_labels: torch.Tensor
    supervision_valid: torch.Tensor

    def inputs(self):
        return self.observations, self.state, self.parameters, self.load_labels

    def select(self, ids):
        return type(self)(**{f.name: getattr(self, f.name)[ids] for f in fields(self)})

    def validate(self, schema, num_envs, device):
        for name in ("observations", "state", "parameters", "load_labels"):
            schema.validate(name, getattr(self, name), num_envs, device)
        if (self.supervision_valid.shape != (num_envs,) or self.supervision_valid.dtype != torch.bool
                or self.supervision_valid.device != torch.device(device)):
            raise ValueError("supervision_valid must be a device-local bool [N] mask")


@dataclass
class BRStep:
    observation: BRObservation  # post-reset current observation on done rows
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor
    terminal: BRObservation | None = None  # full N rows; only terminal_valid rows meaningful
    terminal_valid: torch.Tensor | None = None
    # Optional scalar episode summaries, same convention as existing entropy scheduling.
    # Keys: lin_vel_tracking, ang_vel_tracking, terrain_level. Max across collection
    # matches the PACT runner. Missing/nonfinite complete evidence holds entropy.
    performance_metrics: dict | None = None


class BeyondRobustnessEnv(Protocol):
    num_envs: int
    device: torch.device
    input_schema: BeyondRobustnessInputSchema

    def reset(self) -> BRObservation: ...
    def observe(self) -> BRObservation: ...  # pure, no history/RNG/physics advancement
    def step(self, actions: torch.Tensor) -> BRStep: ...
