"""Synthetic PhysX contracts: no Kit application or GPU required."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import warp as wp

PATH = Path(__file__).resolve().parents[1] / 'legged_gym/simulator/b1z1_contact_forces.py'
spec = importlib.util.spec_from_file_location('b1z1_contact_forces_tested', PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def warp_cache(tmp_path):
    previous = wp.config.kernel_cache_dir
    wp.config.kernel_cache_dir = str(tmp_path / 'warp')
    yield
    wp.config.kernel_cache_dir = previous


class View:
    sensor_count, filter_count, max_contact_data_count = 4, 1, 12

    def __init__(self, dt=.005):
        self.dt, self.queries = dt, []
        self.counts = torch.zeros(4, 1, dtype=torch.int32)
        self.starts = torch.zeros_like(self.counts)
        self.force = torch.zeros(12, 1)
        self.force[[0, 1, 4, 6], 0] = torch.tensor([10., 20., 30., 40.]) * dt
        self.normal = torch.zeros(12, 3)
        self.normal[:, 2] = 1
        self.normal[4] = torch.tensor([.6, 0., .8])  # Normal != world-z on a slope.
        self.points = torch.zeros(12, 3)
        self.points[:, 0] = 1
        self.friction = torch.zeros(12, 3)
        self.friction[[8, 2, 3, 5], 0] = torch.tensor([-3., 2., 4., -8.]) * dt
        self.anchors = torch.zeros(12, 3)
        self.anchors[:, 1] = 2

    def get_contact_data(self, dt):
        self.queries.append(('normal', dt))
        self.counts[:, 0] = torch.tensor([2, 0, 1, 1])
        self.starts[:, 0] = torch.tensor([0, 2, 4, 6])
        return self.force/dt, self.points, self.normal, torch.zeros(12, 1), self.counts, self.starts

    def get_friction_data(self, dt):
        self.queries.append(('friction', dt))
        # Deliberately reuse normal index storage, with independent anchor ranges.
        self.counts[:, 0] = torch.tensor([1, 2, 0, 1])
        self.starts[:, 0] = torch.tensor([8, 2, 0, 5])
        return self.friction/dt, self.anchors, self.counts, self.starts


@pytest.mark.parametrize('dt', [.002, .005, .01])
def test_independent_anchors_scaling_normals_and_moments(dt):
    view = View(dt)
    forces = module.GroundContactForces(view, dt, 'cpu')
    forces.update()
    normal = torch.tensor([[0., 0., 30.], [0., 0., 0.], [18., 0., 24.], [0., 0., 40.]])
    friction = torch.tensor([[-3., 0., 0.], [6., 0., 0.], [0., 0., 0.], [-8., 0., 0.]])
    torch.testing.assert_close(forces.normal, normal)
    torch.testing.assert_close(forces.tangential, friction)
    torch.testing.assert_close(forces.total, normal + friction)
    torch.testing.assert_close(forces.normal_moment, torch.linalg.cross(torch.tensor([[1., 0., 0.]]).expand(4, -1), normal))
    torch.testing.assert_close(forces.tangential_moment, torch.linalg.cross(torch.tensor([[0., 2., 0.]]).expand(4, -1), friction))
    assert view.queries == [('normal', dt), ('friction', dt)]


def test_empty_friction_and_partial_reset():
    view = View()
    original = view.get_friction_data
    def no_contact(dt):
        f, p, c, s = original(dt)
        c.zero_()
        s.fill_(-1)  # Empty ranges may have sentinel starts.
        return f, p, c, s
    view.get_friction_data = no_contact
    forces = module.GroundContactForces(view, .005, 'cpu')
    forces.update()
    assert not forces.tangential.any()
    before = forces.total.clone()
    forces.reset(torch.tensor([0, 1]))
    assert not forces.total[:2].any()
    torch.testing.assert_close(forces.total[2:], before[2:])


@pytest.mark.parametrize('problem', ['capacity', 'range', 'nonfinite', 'unavailable'])
def test_bad_contact_data_fails_closed(problem):
    view = View()
    original = view.get_friction_data
    def bad(dt):
        if problem == 'unavailable':
            raise RuntimeError('Backend friction unavailable')
        f, p, c, s = original(dt)
        if problem == 'capacity': c[0] = view.max_contact_data_count
        if problem == 'range': s[0] = view.max_contact_data_count
        if problem == 'nonfinite': f[8, 0] = float('nan')
        return f, p, c, s
    view.get_friction_data = bad
    with pytest.raises(RuntimeError):
        module.GroundContactForces(view, .005, 'cpu').update()


def test_filters_required_and_semantics_explicit():
    view = View()
    view.filter_count = 0
    with pytest.raises(ValueError, match='filter'):
        module.GroundContactForces(view, .005, 'cpu')
    assert module.contact_semantics('total', .005, 8, ['ground'])['tangential_available']
    assert not module.contact_semantics('normal_only', .005, 8, ['ground'])['tangential_available']


def test_multiple_ground_filters_are_summed_per_sensor():
    view = View()
    view.sensor_count, view.filter_count = 2, 2
    forces = module.GroundContactForces(view, .005, 'cpu')
    forces.update()
    torch.testing.assert_close(forces.total, torch.tensor([[3., 0., 30.], [10., 0., 64.]]))


@pytest.mark.parametrize('acceleration', [0., 2.])
def test_whole_body_momentum_includes_gravity_and_external_loads(acceleration):
    mass, gravity, dt = torch.tensor([2., 3.]), torch.tensor([0., 0., -9.81]), .005
    contact = torch.zeros(8, 2, 3)
    external = torch.zeros_like(contact)
    external[..., 0] = -10.
    external[..., 2] = -25.
    contact[..., 0] = 10. + mass*acceleration
    contact[..., 2] = 25. + mass*9.81
    start, end = torch.zeros(2, 3), torch.zeros(2, 3)
    end[:, 0] = mass*acceleration*len(contact)*dt
    torch.testing.assert_close(module.momentum_residual(start, end, mass, gravity, contact, external, dt), start, atol=1e-6, rtol=0.)
    # A normal-only reference misses the horizontal impulse even in equilibrium.
    contact[..., 0] = 0.
    assert (module.momentum_residual(start, end, mass, gravity, contact, external, dt)[:, 0] > 0).all()
