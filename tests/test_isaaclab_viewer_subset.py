"""USD render visibility must not disable simulated bodies or contacts."""
from types import SimpleNamespace

import pytest

import legged_gym.envs
from legged_gym.simulator.isaaclab_viewer import configure_robot_visibility
from legged_gym.simulator.isaaclab_viewer import select_rendered_envs
from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT


@pytest.mark.parametrize('selected,visible',[(None,[0,1,2]),([] ,[]),([1],[1]),([0,2,20],[0,2])])
def test_subset_preserves_active_bodies_and_clone_visibility(selected,visible):
    Usd=pytest.importorskip('pxr.Usd',reason='USD requires Isaac Sim startup')
    UsdGeom=pytest.importorskip('pxr.UsdGeom')
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World/envs/env_0')
    UsdGeom.Xform.Define(stage,'/World/envs/env_0/go2')
    UsdGeom.Cube.Define(stage,'/World/envs/env_0/go2/visual')
    for i in (1,2):
        clone=UsdGeom.Xform.Define(stage,f'/World/envs/env_{i}')
        clone.GetPrim().GetInherits().AddInherit('/World/envs/env_0')
    ground=UsdGeom.Cube.Define(stage,'/World/ground')
    configure_robot_visibility(stage,3,'go2',selected)
    for i in range(3):
        robot=stage.GetPrimAtPath(f'/World/envs/env_{i}/go2')
        mesh=UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/envs/env_{i}/go2/visual'))
        assert robot.IsActive()
        assert (mesh.ComputeVisibility()!=UsdGeom.Tokens.invisible)==(i in visible)
    assert ground.ComputeVisibility()==UsdGeom.Tokens.inherited


@pytest.mark.parametrize('indices',[[-1],[.5],[True]])
def test_bad_indices_rejected(indices):
    with pytest.raises(ValueError,match='nonnegative integer'):
        configure_robot_visibility(None,3,'go2',indices)


@pytest.mark.parametrize('headless',[False,True])
@pytest.mark.parametrize('enabled',[False,True])
def test_debug_switch_is_independent_and_headless_safe(headless,enabled):
    sim=SimpleNamespace(_headless=headless,_cfg=SimpleNamespace(viewer=SimpleNamespace(debug_vis=enabled)))
    assert IsaacLabSimulator_PACT._contact_sensor_debug_vis(sim)==(enabled and not headless)
    assert IsaacLabSimulator._contact_sensor_debug_vis(sim)==(not headless)


def test_random_display_subset_is_unique_reproducible_and_rng_isolated():
    import random
    import torch
    cfg=SimpleNamespace(rendered_envs_count=250,rendered_envs_seed=12,rendered_envs_idx=[0])
    py_state=random.getstate();torch_state=torch.random.get_rng_state()
    selected=select_rendered_envs(cfg,4096)
    assert len(selected)==len(set(selected))==250
    assert all(0<=i<4096 for i in selected)
    assert selected==select_rendered_envs(cfg,4096)
    assert random.getstate()==py_state
    assert torch.equal(torch.random.get_rng_state(),torch_state)
    cfg.rendered_envs_seed=13
    assert selected!=select_rendered_envs(cfg,4096)
    assert select_rendered_envs(cfg,3)==[0,1,2]
    cfg.rendered_envs_count=0
    assert select_rendered_envs(cfg,4096)==[]
    cfg.rendered_envs_count=None
    assert select_rendered_envs(cfg,4096)==[0]


@pytest.mark.parametrize('count',[-1,1.5,True])
def test_random_display_rejects_invalid_counts(count):
    with pytest.raises(ValueError,match='rendered_envs_count'):
        select_rendered_envs(SimpleNamespace(rendered_envs_count=count),4096)
