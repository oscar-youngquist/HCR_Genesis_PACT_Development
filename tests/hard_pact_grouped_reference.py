"""Reviewed ce95d0c grouped builder, test-only forward/VJP reference."""
import torch
from rsl_rl.algorithms.hard_pact_qp import _ScaleClipRows, _row_scale, _QPBuild

def _build(self, data, stance_pattern=None):
    r"""Assemble x=[tau_12; f_FR,FL,RR,RL_world_12], and substitute x=D z.

    One call contains one detached stance pattern. Swing coordinates have
    exactly one identity equality each and NO friction rows. No dynamics
    equalities, accelerations, contact slacks, or temporal costs are added.
    """
    ref = data["tau_nom"]
    batch = ref.shape[0]
    eye, scale = self._constants(ref)
    mass = data["mass_matrix"].detach()
    J = data["foot_jacobians"].detach().reshape(batch, 12, 18)
    Jb = data["base_jacobian"].detach()
    bias = data["bias"].detach()
    # M a = [S^T J^T]x + Jb^T W - h. solve_ex reports a singular
    # mechanics row without poisoning all other environments in its batch.
    selector = ref.new_zeros(18, 12)
    selector[6:] = torch.eye(12, device=ref.device, dtype=ref.dtype)
    rhs = torch.cat((selector.expand(batch, -1, -1), J.transpose(1, 2),
                     Jb.transpose(1, 2), bias[..., None]), -1)
    solved, info = torch.linalg.solve_ex(mass, rhs, check_errors=False)
    mechanics_valid = (info == 0) & torch.isfinite(solved).all(dim=(1,2))
    # Fixed mechanics have no gradient responsibility. Sanitizing a failed
    # factor before multiplying learned W also prevents 0*NaN VJPs.
    solved = torch.nan_to_num(solved.detach(), nan=0., posinf=0., neginf=0.)
    mechanics_map = solved[:,:,:24]
    wrench = _ScaleClipRows.apply(data["wrench_pred_world"],
        self.cfg.gradient_scale_wrench, self.cfg.gradient_clip_wrench, self, "wrench")
    offset = (solved[:,:,24:30] @ wrench[...,None]).squeeze(-1)-solved[:,:,30]
    stance = data["contact_probability"].detach() >= self.cfg.contact_threshold
    if stance_pattern is None:
        # Private builder accepts homogeneous batches for algebra tests.
        if not torch.equal(stance, stance[:1].expand_as(stance)):
            raise ValueError("_build requires a homogeneous stance pattern")
        stance_pattern = sum(int(stance[0, i]) << i for i in range(4))
    feet = [i for i in range(4) if stance_pattern & (1 << i)]
    swing = [i for i in range(4) if not stance_pattern & (1 << i)]
    tau = _ScaleClipRows.apply(ref, self.cfg.gradient_scale_tau,
                              self.cfg.gradient_clip_tau, self, "tau_nom")
    force = _ScaleClipRows.apply(data["force_pred_world"],
        self.cfg.gradient_scale_grf, self.cfg.gradient_clip_grf, self, "grf")
    # Mask ONLY the tracking reference. Raw supervised predictions stay
    # unbounded, and optimized swing forces are constrained separately.
    force = torch.where(stance[..., None], force, torch.zeros_like(force)).flatten(1)
    target = torch.cat((tau, force), -1)
    weights = ref.new_tensor([self.cfg.torque_tracking_weight] * 12
                             + [self.cfg.force_tracking_weight] * 12)
    diagonal = 2 * weights / scale.square()
    Q = torch.diag(diagonal).expand(batch, -1, -1).clone()
    p = -diagonal * target

    def add_residual(C, e, weight):
        # w||C x+e||^2 -> Q += 2w C^T C, p += 2w C^T e.
        nonlocal Q, p
        Q = Q + 2 * weight * C.transpose(1, 2) @ C
        p = p + 2 * weight * (C.transpose(1, 2) @ e[..., None]).squeeze(-1)

    if feet and self.cfg.contact_acceleration_weight:
        rows = [3 * foot + axis for foot in feet for axis in range(3)]
        contact_J = J[:, rows]
        C = contact_J @ mechanics_map / self.cfg.contact_acceleration_scale_m_s2
        e = ((contact_J @ offset[..., None]).squeeze(-1)
             + data["foot_acceleration_bias"].detach().flatten(1)[:, rows])
        add_residual(C, e / self.cfg.contact_acceleration_scale_m_s2,
                     self.cfg.contact_acceleration_weight)

    if self.cfg.attitude_weight:
        # H maps canonical acceleration to yaw-local PHYSICAL angular
        # acceleration (not Euler-angle second derivatives). In a free
        # flyer Jb_angular*v = R_WB*w_B and Jdotb_angular*v =
        # R_WB*(w_B cross w_B)=0. Project into instantaneous yaw axes;
        # we do not differentiate the yaw coordinate frame.
        q = data["base_quaternion"].detach()  # canonical xyzw
        yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),
                          1-2*(q[:,1].square()+q[:,2].square()))
        c, s = yaw.cos(), yaw.sin()
        R = ref.new_zeros(batch, 2, 3)
        R[:,0,0], R[:,0,1] = c, s
        R[:,1,0], R[:,1,1] = -s, c
        up = torch.stack((2*(q[:,0]*q[:,2]+q[:,3]*q[:,1]),
                          2*(q[:,1]*q[:,2]-q[:,3]*q[:,0]),
                          1-2*(q[:,0].square()+q[:,1].square())), -1)
        # z_world cross z_body is a restoring tilt-error rotation vector
        # near upright: [roll,pitch] in yaw-local horizontal axes.
        tilt_world = torch.stack((-up[:,1], up[:,0], torch.zeros_like(up[:,0])), -1)
        tilt = (R @ tilt_world[...,None]).squeeze(-1)
        omega = (R @ data["base_angular_velocity_world"].detach()[...,None]).squeeze(-1)
        desired = -self.cfg.attitude_kp * tilt - self.cfg.attitude_kd * omega
        H = R @ Jb[:,3:6]
        C = H @ mechanics_map / self.cfg.attitude_acceleration_scale_rad_s2
        e = ((H @ offset[...,None]).squeeze(-1) - desired)
        add_residual(C, e / self.cfg.attitude_acceleration_scale_rad_s2,
                     self.cfg.attitude_weight)

    limits, qmin, qmax, vmax = self._limits(ref)
    dt = data["dt"].detach().reshape(-1, 1)
    previous = data["previous_torque"].detach()
    lower = torch.maximum(-limits, previous - self.cfg.torque_rate_limit_nm_s * dt)
    upper = torch.minimum(limits, previous + self.cfg.torque_rate_limit_nm_s * dt)
    q, v = data["joint_position"].detach(), data["joint_velocity"].detach()
    amax = ref.new_tensor(self.cfg.joint_acceleration_limits_rad_s2).reshape(1,12)
    beta = self.cfg.position_integration_coefficient
    alower = torch.maximum(-amax, torch.maximum((-vmax-v)/dt, (qmin-q-dt*v)/(beta*dt.square())))
    aupper = torch.minimum(amax, torch.minimum((vmax-v)/dt, (qmax-q-dt*v)/(beta*dt.square())))
    joint_map, joint_offset = mechanics_map[:,6:], offset[:,6:]
    # Torque 24 rows, acceleration intersection 24 rows. beta=1 matches
    # semi-implicit Genesis/PhysX; beta=.5 is the constant-a convention.
    G = [eye[:12].expand(batch,-1,-1), -eye[:12].expand(batch,-1,-1),
         joint_map, -joint_map]
    h = [upper, -lower, aupper-joint_offset, joint_offset-alower]
    for foot in feet:
        block = ref.new_zeros(5,24)
        col = 12+3*foot
        block[0,col+2] = -1
        block[1,col], block[2,col] = 1, -1
        block[3,col+1], block[4,col+1] = 1, -1
        block[1:,col+2] = -self.cfg.friction_coefficient
        G.append(block.expand(batch,-1,-1))
        h.append(ref.new_zeros(batch,5))
    physical_G, physical_h = torch.cat(G,1), torch.cat(h,1)
    swing_columns = [12+3*foot+axis for foot in swing for axis in range(3)]
    physical_A = eye[swing_columns].expand(batch,-1,-1)
    physical_b = ref.new_zeros(batch,len(swing_columns))
    G, h, gs = _row_scale(physical_G * scale, physical_h)
    A, b, es = _row_scale(physical_A * scale, physical_b)
    Q = Q * scale[:,None] * scale[None,:]
    Q = .5 * (Q + Q.transpose(1,2))
    Q = Q + self.cfg.q_regularization * eye
    p = p * scale
    native_lower = ref.new_full((batch,24), -torch.inf)
    native_upper = ref.new_full((batch,24), torch.inf)
    native_lower[:,:12], native_upper[:,:12] = lower/scale[:12], upper/scale[:12]
    return _QPBuild(Q,p,G,h,A,b,scale,physical_G,physical_h,physical_A,physical_b,
                    es,gs,lower,upper,alower,aupper,native_lower,native_upper,
                    mechanics_map,offset,mechanics_valid)
