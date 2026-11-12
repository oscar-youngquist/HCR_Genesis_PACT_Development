"""Bounded hardware checks; uses real PhysX, never a mock or scripted tray."""
import torch
from rsl_rl.beyond_robustness_payload import spawn_state


@torch.no_grad()
def physics_checks(env):
    s=env.simulator; n=env.num_envs; device=env.device
    assert n>=2
    zeros=torch.zeros(n,12,device=device)
    env.reset()
    # Observe is pure, including RNG/history/properties.
    rng=s.payload_rng.get_state().clone(); before=env.observe().clone()
    again=env.observe()
    for x,y in zip(before.inputs(),again.inputs()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert torch.equal(rng,s.payload_rng.get_state())
    # Initial/intermediate/final envelopes: changing progress alone must not
    # change active actors. Only the selected reset applies the new sample.
    stages=[]
    saved_progress=env.domain_rand_curriculum.progress['disturbance']
    for progress in (0.,.5,1.):
        prior=s.payload_mass.clone()
        s.set_payload_progress(progress)
        torch.testing.assert_close(prior,s.payload_mass,rtol=0,atol=0)
        env._reset(torch.tensor([0],device=device))
        lo,hi=s.payload_cfg.ranges(progress)['mass']
        assert lo<=s.payload_mass[0]<=hi
        torch.testing.assert_close(prior[1:],s.payload_mass[1:],rtol=0,atol=0)
        stages.append(float(s.payload_mass[0]))
    s.set_payload_progress(saved_progress); env.reset()
    tray=s._contact_sensors.body_names.index('br_tray')
    support=[]; ground=[]
    for _ in range(100):
        s.step(zeros); s.post_physics_step()
        support.append(s._contact_sensors.data.net_forces_w[:,tray,2].clone())
        ground.append(s.ground_total_forces_world[:,tray].norm(dim=-1).clone())
    # Stable relative height near floor top; tray receives box reaction while
    # the pair-filtered ground channel excludes that normal support.
    expected=s.payload_cfg.tray_height+s.payload_cfg.tray_size[2]/2+s.box_size[2]/2
    labels=s.payload_labels(); support=torch.stack(support[-30:]); ground=torch.stack(ground[-30:])
    assert (labels[:,2]-expected).abs().max()<.025, labels
    assert support.mean()<-.2 and ground.max()<.1, (support.mean(),ground.max())
    support_mean=float(support.mean())
    # Actual material combine mode is multiply; tray coefficient is restored
    # after generic robot friction randomization.
    ids=torch.arange(n,device=device)
    s._randomize_friction(ids)
    material=s._robot.root_physx_view.get_material_properties()
    torch.testing.assert_close(material[:,s._tray_shape_slice,:2],
        torch.full_like(material[:,s._tray_shape_slice,:2],s.payload_cfg.tray_friction))
    # Reset one robot/load only; all other world states, samples, and histories
    # remain bitwise unchanged (no physics step in reset).
    root=s._robot.data.root_link_state_w.clone(); box=s.payload.data.root_link_state_w.clone()
    mass=s.payload_mass.clone(); params=env.parameters.clone(); actions=env.previous_action.clone()
    env._reset(torch.tensor([0],device=device))
    for x,y in ((root,s._robot.data.root_link_state_w),(box,s.payload.data.root_link_state_w),
                (mass,s.payload_mass),(params,env.parameters),(actions,env.previous_action)):
        torch.testing.assert_close(x[1:],y[1:],rtol=0,atol=0)
    # Boundary relocation is a shared translation, not manufactured load motion.
    root=s._robot.data.root_link_state_w[:1].clone()
    box=s.payload.data.root_link_state_w[:1].clone()
    shift=s._terrain_x_range[1]+1-root[0,0]
    root[:,0]+=shift; box[:,0]+=shift
    one=torch.tensor([0],device=device)
    s._robot.write_root_link_state_to_sim(root,one); s.payload.write_root_link_state_to_sim(box,one)
    relative=s.payload_labels()[0].clone()
    s.post_physics_step()
    torch.testing.assert_close(relative,s.payload_labels()[0],atol=5e-6,rtol=1e-5)
    # Finite-duration inertial slide: give box a relative velocity once, not a
    # pose-following/scripted motion; lower/higher friction pair readback.
    env.reset()
    for _ in range(60): s.step(zeros); s.post_physics_step()
    d=s._robot.data
    rel=s.payload_offset.clone(); rel[:,:2]=0
    speed=torch.zeros(n,3,device=device); speed[:,0]=.6
    state=spawn_state(d.root_link_quat_w,d.root_link_pos_w,d.root_link_lin_vel_w,d.root_link_ang_vel_w,rel,speed)
    s.payload.write_root_state_to_sim(state)
    props=s.payload.root_physx_view.get_material_properties().clone()
    props[0,:,:2]=.02; props[1:,:,:2]=.8
    s.payload.root_physx_view.set_material_properties(props,torch.arange(n))
    initial=s.payload_labels()[:,:3].clone()
    for _ in range(6): s.step(zeros); s.post_physics_step()
    displacement=s.payload_labels()[:,:3]-initial
    assert displacement[0,0]>.01, displacement
    assert displacement[0,0]>displacement[1,0]+.005, displacement
    # No claim of calibrated friction identification: readback plus a matched
    # short physical slide establishes material installation and relative motion.
    assert s.payload.root_physx_view.get_material_properties()[0,0,0]<.03
    assert s.payload.root_physx_view.get_material_properties()[1,0,0]>.79
    # An instantaneous torso velocity push leaves the independent load behind,
    # generating real relative motion (no kinematic tray or external box force).
    prior=s.payload_labels()[:,:3].clone()
    velocity=s._robot.data.root_link_vel_w.clone(); velocity[:,1]+=1.
    s._robot.write_root_link_velocity_to_sim(velocity)
    for _ in range(4): s.step(zeros); s.post_physics_step()
    push_displacement=s.payload_labels()[:,:3]-prior
    assert push_displacement[:,:2].norm(dim=-1).max()>.01
    # Cross-env collision isolation: move box 0 to the same pose as box 1 with
    # identical velocity. Neither box may receive a box-box separating impulse.
    env.reset(); pose=s.payload.data.root_state_w.clone()
    pose[0]=pose[1]; pose[0,2]+=1.; pose[1,2]+=1.; pose[:2,7:]=0
    s.payload.write_root_state_to_sim(pose)
    for _ in range(3): s.step(zeros); s.post_physics_step()
    torch.testing.assert_close(s.payload.data.root_com_lin_vel_w[0],s.payload.data.root_com_lin_vel_w[1],atol=2e-4,rtol=1e-4)
    # Geometry-based escape, simultaneous timeout: failure must win; terminal
    # box label remains owned after automatic selected-row reset.
    env.reset(); pose=s.payload.data.root_state_w[:1].clone(); pose[:,0]+=1.
    s.payload.write_root_state_to_sim(pose,torch.tensor([0],device=device))
    env.episode_steps[0]=env.max_episode_length-1
    result=env.step(zeros)
    assert result.dones[0] and not result.timeouts[0]
    assert env.last_failure_reasons['payload_lost'][0]
    assert result.terminal.load_labels[0,0]>.5 and result.observation.load_labels[0,0].abs()<.2, (result.terminal.load_labels[0],result.observation.load_labels[0])
    # Genuine tray/torso-ground contact survives pair filtering. Place the
    # upside-down robot low enough that its tray strikes the terrain.
    env.reset(); root=s._robot.data.root_link_state_w[:1].clone()
    root[:,2]=s.env_origins[:1,2]+.13; root[:,3:7]=torch.tensor([0.,1.,0.,0.],device=device)
    root[:,7:]=0
    s._robot.write_root_link_state_to_sim(root,torch.tensor([0],device=device))
    impact=0.
    for _ in range(10):
        s.step(zeros); s.post_physics_step()
        impact=max(impact,float(s.ground_total_forces_world[0,env.collision_ids].norm(dim=-1).max()))
    assert impact>1., impact
    env.reset()
    return dict(support_reaction_z_n=support_mean,filtered_tray_ground_n=float(ground.max()),
        mass_initial_half_final_kg=stages,
        slide_x_m=displacement[:2,0].tolist(),push_relative_motion_m=float(push_displacement[:,:2].norm(dim=-1).max()),ground_collision_peak_n=impact,
        partial_reset=True,terminal_ownership=True,failure_precedence=True,inter_env_isolation=True,
        paired_boundary_relocation=True)
