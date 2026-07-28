"""Backend-independent whole-body dynamics contract for B1/Z1 PACT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class WholeBodyTerms:
    mass_matrix: torch.Tensor
    bias: torch.Tensor
    generalized_contacts: torch.Tensor


class WholeBodyDynamicsBackend(ABC):
    """Abstract boundary for Pinocchio now and a future BARD implementation."""

    @abstractmethod
    def evaluate(
        self, base_pos: torch.Tensor, base_quat_xyzw: torch.Tensor,
        dof_pos: torch.Tensor, base_linear_velocity: torch.Tensor,
        base_angular_velocity: torch.Tensor, dof_velocity: torch.Tensor,
        grfs_world: torch.Tensor, ee_force_world: torch.Tensor,
        base_wrench_world: torch.Tensor, base_added_mass: torch.Tensor | None = None,
        base_com_shift: torch.Tensor | None = None,
        gripper_added_mass: torch.Tensor | None = None,
    ) -> WholeBodyTerms:
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources. Async backends override this method."""
