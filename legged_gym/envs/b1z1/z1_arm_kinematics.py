"""Differentiable Z1 kinematics in the existing trunk frame; geometry is caller-owned."""
import torch


def compute_z1_arm_jacobian(q_arm: torch.Tensor, joint_offsets, joint_axes, link00_offset, ee_offset) -> torch.Tensor:
    """
    Compute translational EE Jacobian for the Z1 arm.

    Args:
        q_arm: shape (N, 6), ordered as:
            [
                z1_waist,
                z1_shoulder,
                z1_elbow,
                z1_wrist_angle,
                z1_forearm_roll,
                z1_wrist_rotate,
            ]

    Returns:
        J: shape (N, 3, 6)
    """
    assert q_arm.ndim == 2 and q_arm.shape[1] == 6, (
        f"Expected q_arm shape (N, 6), got {q_arm.shape}"
    )

    q_arm = torch.nan_to_num(q_arm, nan=0.0, posinf=0.0, neginf=0.0)

    N = q_arm.shape[0]
    device = q_arm.device
    dtype = q_arm.dtype

    # Cast constants only if needed.
    joint_offsets = joint_offsets.to(device=device, dtype=dtype)
    joint_axes = joint_axes.to(device=device, dtype=dtype)
    link00_offset = link00_offset.to(device=device, dtype=dtype)
    ee_offset = ee_offset.to(device=device, dtype=dtype)

    # Current transform from base to active frame.
    R = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3).clone()
    p = link00_offset.view(1, 3).expand(N, 3).clone()

    joint_pos = torch.empty(N, 6, 3, device=device, dtype=dtype)
    joint_axis_world = torch.empty(N, 6, 3, device=device, dtype=dtype)

    for i in range(6):
        # p = p + R @ offset_i
        p = p + torch.einsum("nij,j->ni", R, joint_offsets[i])

        # store joint origin
        joint_pos[:, i, :] = p

        # axis_world = R @ axis_local
        joint_axis_world[:, i, :] = torch.einsum("nij,j->ni", R, joint_axes[i])

        qi = q_arm[:, i]
        c = torch.cos(qi)
        s = torch.sin(qi)

        R_next = R.clone()

        if i == 0 or i == 4:
            # local z rotation
            # R = R @ Rz(q)
            r0 = R[:, :, 0].clone()
            r1 = R[:, :, 1].clone()

            R_next[:, :, 0] = c[:, None] * r0 + s[:, None] * r1
            R_next[:, :, 1] = -s[:, None] * r0 + c[:, None] * r1
            R_next[:, :, 2] = R[:, :, 2]

        elif i == 1 or i == 2 or i == 3:
            # local y rotation
            # R = R @ Ry(q)
            r0 = R[:, :, 0].clone()
            r2 = R[:, :, 2].clone()

            R_next[:, :, 0] = c[:, None] * r0 - s[:, None] * r2
            R_next[:, :, 1] = R[:, :, 1]
            R_next[:, :, 2] = s[:, None] * r0 + c[:, None] * r2

        else:
            # i == 5, local x rotation
            # R = R @ Rx(q)
            r1 = R[:, :, 1].clone()
            r2 = R[:, :, 2].clone()

            R_next[:, :, 0] = R[:, :, 0]
            R_next[:, :, 1] = c[:, None] * r1 + s[:, None] * r2
            R_next[:, :, 2] = -s[:, None] * r1 + c[:, None] * r2

        R = R_next

    p_ee = p + torch.einsum("nij,j->ni", R, ee_offset)

    r = p_ee[:, None, :] - joint_pos  # [N, 6, 3]

    # cross(axis, r), then transpose to [N, 3, 6]
    J_cols = torch.cross(joint_axis_world, r, dim=2)

    J = J_cols.transpose(1, 2).contiguous()

    return torch.nan_to_num(J, nan=0.0, posinf=1e6, neginf=-1e6)


def compute_z1_arm_fk(q_arm: torch.Tensor, joint_offsets, joint_axes, link00_offset, ee_offset) -> torch.Tensor:
    """Return the EE Cartesian position in the Z1 trunk frame.

    This mirrors the retained analytic Jacobian chain so FiLM's EE tracking
    error is genuinely computed from arm forward kinematics rather than a
    hidden simulator link-position observation.
    """
    n = q_arm.shape[0]
    R = torch.eye(3, dtype=q_arm.dtype, device=q_arm.device).expand(n, 3, 3).clone()
    p = link00_offset.to(q_arm).view(1, 3).expand(n, 3).clone()
    offsets = joint_offsets.to(q_arm)
    for i in range(6):
        p = p + torch.einsum("nij,j->ni", R, offsets[i])
        c, s = torch.cos(q_arm[:, i]), torch.sin(q_arm[:, i])
        next_R = R.clone()
        if i in (0, 4):
            r0, r1 = R[:, :, 0].clone(), R[:, :, 1].clone()
            next_R[:, :, 0], next_R[:, :, 1] = c[:, None] * r0 + s[:, None] * r1, -s[:, None] * r0 + c[:, None] * r1
        elif i in (1, 2, 3):
            r0, r2 = R[:, :, 0].clone(), R[:, :, 2].clone()
            next_R[:, :, 0], next_R[:, :, 2] = c[:, None] * r0 - s[:, None] * r2, s[:, None] * r0 + c[:, None] * r2
        else:
            r1, r2 = R[:, :, 1].clone(), R[:, :, 2].clone()
            next_R[:, :, 1], next_R[:, :, 2] = c[:, None] * r1 + s[:, None] * r2, -s[:, None] * r1 + c[:, None] * r2
        R = next_R
    return p + torch.einsum("nij,j->ni", R, ee_offset.to(q_arm))
