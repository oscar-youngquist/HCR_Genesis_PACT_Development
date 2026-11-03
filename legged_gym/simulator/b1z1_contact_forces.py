"""Ground-filtered PhysX contact reduction; forces N, points world m.

The installed tensor API returns normal impulses/dt and friction impulses/dt
separately. One thread per sensor/filter pair visits only its live contacts;
there is no padded contact tensor, host transfer, or cross-environment prefix sum.
"""
import torch
import warp as wp


@wp.kernel
def _reduce_contacts(forces: wp.array2d(dtype=wp.float32),
                     points: wp.array2d(dtype=wp.float32),
                     normals: wp.array2d(dtype=wp.float32),
                     counts: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                     capacity: int, normal_mode: bool,
                     summed: wp.array2d(dtype=wp.float32),
                     moment: wp.array2d(dtype=wp.float32), errors: wp.array(dtype=wp.int32)):
    row = wp.tid()
    count = counts[row]
    start = starts[row]
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    error = int(0)
    if count < 0 or (count > 0 and (start < 0 or start + count > capacity)):
        error = 1
    else:
        for i in range(start, start + count):
            f = wp.vec3(0.0)
            if normal_mode:
                f = forces[i, 0] * wp.vec3(normals[i, 0], normals[i, 1], normals[i, 2])
            else:
                f = wp.vec3(forces[i, 0], forces[i, 1], forces[i, 2])
            p = wp.vec3(points[i, 0], points[i, 1], points[i, 2])
            if not wp.isfinite(f[0]) or not wp.isfinite(f[1]) or not wp.isfinite(f[2]):
                error = 2
            if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
                error = 2
            force = force + f
            torque = torque + wp.cross(p, f)
    for axis in range(3):
        summed[row, axis] = force[axis]
        moment[row, axis] = torque[axis]
    errors[row] = error


class GroundContactForces:
    """Reusable device buffers in PhysX sensor order; fail closed on lost labels."""
    def __init__(self, view, dt, device):
        if view.filter_count < 1 or view.max_contact_data_count < 1:
            raise ValueError('Total GRFs require valid ground filters and nonzero contact capacity')
        if not callable(getattr(view, 'get_friction_data', None)):
            raise RuntimeError('Installed PhysX has no friction API; select explicit normal_only mode for legacy runs')
        if dt <= 0:
            raise ValueError('Contact conversion requires positive physics dt')
        wp.init()  # Idempotent; also permits simulator-independent CPU validation.
        self.view, self.dt = view, float(dt)
        self.normal = torch.zeros(view.sensor_count, 3, device=device)
        self.tangential = torch.zeros_like(self.normal)
        self.total = torch.zeros_like(self.normal)
        self.normal_moment = torch.zeros_like(self.normal)
        self.tangential_moment = torch.zeros_like(self.normal)
        self.moment = torch.zeros_like(self.normal)
        self.errors = torch.zeros(view.sensor_count * view.filter_count, dtype=torch.int32, device=device)
        self._pair_force = torch.zeros(len(self.errors), 3, device=device) if view.filter_count > 1 else None
        self._pair_moment = torch.zeros_like(self._pair_force) if self._pair_force is not None else None

    def _reduce(self, forces, points, normals, counts, starts, normal):
        counts, starts = counts.reshape(-1).to(torch.int32), starts.reshape(-1).to(torch.int32)
        if counts.numel() != self.errors.numel() or starts.shape != counts.shape:
            raise RuntimeError('PhysX contact pair shape differs from sensor/filter layout')
        # PhysX raises on backend query failure. Exact capacity exhaustion is also
        # rejected conservatively: it cannot certify that no contacts were dropped.
        destination = self.normal if normal else self.tangential
        moment = self.normal_moment if normal else self.tangential_moment
        pair_force = destination if self.view.filter_count == 1 else self._pair_force
        pair_moment = moment if self.view.filter_count == 1 else self._pair_moment
        stream = wp.stream_from_torch(self.normal.device) if self.normal.is_cuda else None
        wp.launch(_reduce_contacts, dim=len(counts), inputs=[
            wp.from_torch(forces), wp.from_torch(points), wp.from_torch(normals),
            wp.from_torch(counts), wp.from_torch(starts), self.view.max_contact_data_count, normal,
            wp.from_torch(pair_force), wp.from_torch(pair_moment), wp.from_torch(self.errors)],
            device=str(self.normal.device), stream=stream)
        if self.view.filter_count != 1:
            torch.sum(pair_force.view(self.view.sensor_count, self.view.filter_count, 3), dim=1, out=destination)
            torch.sum(pair_moment.view(self.view.sensor_count, self.view.filter_count, 3), dim=1, out=moment)
        valid = (self.errors == 0).all() & (counts.sum(dtype=torch.int64) < self.view.max_contact_data_count)
        torch._assert_async(valid, 'Invalid/nonfinite or exhausted ground contact data; increase sim.grf.contact_capacity_per_body if exhausted')

    def update(self):
        force, points, normals, _, counts, starts = self.view.get_contact_data(dt=self.dt)
        # get_friction_data REUSES normal count/start storage in this installed
        # PhysX version. Reduce normal data before requesting the friction buffers.
        self._reduce(force, points, normals, counts.clone(), starts.clone(), True)
        friction, anchors, counts, starts = self.view.get_friction_data(dt=self.dt)
        self._reduce(friction, anchors, normals, counts, starts, False)
        torch.add(self.normal, self.tangential, out=self.total)  # Already N; no second dt division.
        torch.add(self.normal_moment, self.tangential_moment, out=self.moment)

    def reset(self, sensor_ids):
        for value in (self.normal, self.tangential, self.total, self.normal_moment, self.tangential_moment, self.moment):
            value[sensor_ids] = 0


def contact_semantics(mode, dt, capacity, filters):
    return dict(version=2, mode=mode, frame='world', force_units='N', moment_units='N*m',
        physics_dt=float(dt), source=('PhysX RigidContactView.get_contact_data + get_friction_data'
            if mode == 'total' else 'Isaac Lab filtered normal force matrix (explicit legacy mode)'),
        normal_source='scalar normal force times actual contact normal',
        tangential_available=mode == 'total', filter_patterns=filters,
        capacity_per_body=capacity, overflow='fail on API error, invalid indices or capacity exhaustion',
        moment_reference=('sum of contact/anchor point cross force; shifted to whole-body CoM at each substep'
            if mode == 'total' else 'link-origin point-force approximation; normal-only'),
        legacy='Absent version denotes historical normal-only sensor recordings, not total GRF')


def momentum_residual(momentum_start, momentum_end, mass, gravity, contact, external, dt):
    """Ns residual; contact/external are completed-substep world N, [steps, env, 3].

    Include non-foot ground contacts and every known nonground external force.
    Internal actuation cancels in whole-body linear momentum, not torso momentum.
    """
    impulse = (contact + external + mass[:, None] * gravity).sum(0) * dt
    return momentum_end - momentum_start - impulse
