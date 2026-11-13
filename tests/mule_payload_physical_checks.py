"""Bounded real-PhysX experiments, isolated nominal resets (not training defaults)."""
import torch
from legged_gym.envs.go2.go2_mule.mule_schema import MULECommand
from legged_gym.envs.go2.go2_mule.rolling_payload import ball_spawn_state


@torch.no_grad()
def physics_checks(env):
    s=env.simulator; n=env.num_envs; device=env.device
    assert n>=2 and len(s.balls)==4
    original=(env._reset_dofs,env._reset_root_states)
    def joints(ids):
        s.reset_dofs(ids,s.default_dof_pos.expand(n,-1)[ids],torch.zeros(len(ids),12,device=device))
    def root(ids):
        s.reset_root_states(ids,s.base_init_pos.expand(len(ids),-1)+s.env_origins[ids],
            s.base_init_quat.expand(len(ids),-1),torch.zeros(len(ids),3,device=device),torch.zeros(len(ids),3,device=device))
    env._reset_dofs,env._reset_root_states=joints,root
    target=torch.tensor(env.cfg.standing_position,device=device).expand(n,12)
    zero=torch.zeros(n,12,device=device)
    command=MULECommand(zero,zero,zero,zero,target)
    def advance(count):
        for _ in range(count): s.step_position_targets(target); s.post_physics_step()
    try:
        env.reset(); ids=torch.arange(n,device=device)
        s.ball_mass.fill_(.15); s._write_ball_properties(ids)
        tray=s._contact_sensors.body_names.index('br_tray')
        support=[]; ground=[]
        for _ in range(80):
            advance(1)
            support.append(s.payload_reaction_world[:,2].clone())
            ground.append(s.ground_total_forces_world[:,tray].norm(dim=-1).clone())
        support=torch.stack(support[-20:]); ground=torch.stack(ground[-20:])
        r,v=s.payload_motion(); expected=s.payload_cfg.tray_height+s.payload_cfg.tray_size[2]/2+s.ball_radius
        assert (r[:,:,2]-expected).abs().max()<.04,(r,expected)
        assert support.mean()<-.5 and ground.max()<.1,(support,ground)
        # Robot/tray truly receive the payload reaction; compare independent
        # robot aggregate contact sensor with ball->robot pair-filtered sum.
        reaction=s._contact_sensors.data.net_forces_w[:,tray,2]
        assert reaction.mean()<-.5,reaction
        # Joint column mapping against the full world Jacobian, with base
        # columns included exactly once. This detects order/offset mistakes.
        d=s._robot.data; j=s._robot.root_physx_view.get_jacobians()
        predicted=(j[:,s._mule_feet,:3,:]@torch.cat((d.root_com_lin_vel_w,d.root_com_ang_vel_w,d.joint_vel),-1)[:,None,:,None]).squeeze(-1)
        torch.testing.assert_close(predicted,d.body_com_lin_vel_w[:,s._mule_feet],atol=.02,rtol=.05)
        # Explicit material readback and tray protection after generic DR.
        s._randomize_friction(ids)
        material=s._robot.root_physx_view.get_material_properties()
        torch.testing.assert_close(material[:,s._tray_shape_slice,:2],
            torch.full_like(material[:,s._tray_shape_slice,:2],s.payload_cfg.tray_friction))
        # Two balls collide in mid-air, no robot/ground forces. Equal masses,
        # opposite velocities, a real closing gap; angular states unconstrained.
        env.reset(); s.ball_mass.fill_(.2); s._write_ball_properties(ids)
        for k,ball in enumerate(s.balls):
            state=ball.data.root_state_w.clone()
            state[:,:3]=s._robot.data.root_link_pos_w+torch.tensor([0.,0.,1.5+k],device=device)
            state[:,7:]=0
            if k<2:
                state[:,2]=s._robot.data.root_link_pos_w[:,2]+1.5
                state[:,0]+=(1 if k==0 else -1)*(s.ball_radius[:,k]+.015)
                state[:,7]=-.7 if k==0 else .7
            ball.write_root_state_to_sim(state)
        advance(4)
        relative_closing=s.balls[0].data.root_com_lin_vel_w[:,0]-s.balls[1].data.root_com_lin_vel_w[:,0]
        assert (relative_closing>-.5).all(),relative_closing
        # Independent roll/slide after a finite velocity impulse; no pose follower.
        env.reset(); advance(40)
        before,_=s.payload_motion()
        ball=s.balls[0]; velocity=ball.data.root_com_vel_w.clone(); velocity[:,0]+=.5
        ball.write_root_com_velocity_to_sim(velocity)
        advance(6); after,_=s.payload_motion()
        motion=(after[:,0]-before[:,0]).norm(dim=-1)
        assert motion.max()>.01,motion
        assert ball.data.root_com_ang_vel_w.norm(dim=-1).max()>.1
        # Cross-environment overlap must not generate a separating impulse.
        env.reset(); ball=s.balls[0]; state=ball.data.root_state_w.clone()
        state[0]=state[1]; state[:2,2]+=2; state[:2,7:]=0
        ball.write_root_state_to_sim(state); advance(3)
        torch.testing.assert_close(ball.data.root_com_lin_vel_w[0],ball.data.root_com_lin_vel_w[1],atol=3e-4,rtol=1e-4)
        # Pure relocation preserves relative motion of all four balls.
        env.reset(); one=torch.tensor([0],device=device)
        root_state=s._robot.data.root_link_state_w[:1].clone()
        delta=s._terrain_x_range[1]+1-root_state[0,0]; root_state[:,0]+=delta
        s._robot.write_root_link_state_to_sim(root_state,one)
        for ball in s.balls:
            state=ball.data.root_state_w[:1].clone(); state[:,0]+=delta; ball.write_root_state_to_sim(state,one)
        prior=s.payload_motion(); s.post_physics_step(); relocated=s.payload_motion()
        for x,y in zip(prior,relocated): torch.testing.assert_close(x[0],y[0],atol=1e-5,rtol=1e-5)
        # Any escaped ball terminates; failure beats simultaneous timeout.
        env.reset(); ball=s.balls[0]; state=ball.data.root_state_w[:1].clone(); state[:,0]+=1
        ball.write_root_state_to_sim(state,one); env.episode_steps[0]=env.max_episode_length-1
        old_id=env.observe().episode_id.clone(); result=env.step(command)
        assert result.terminated[0] and not result.timed_out[0] and result.failure_reasons['payload_lost'][0]
        assert result.terminal.episode_id[0]==old_id[0] and result.observation.episode_id[0]==old_id[0]+1
        owned=result.terminal.actor.clone(); env.reset(one)
        torch.testing.assert_close(result.terminal.actor,owned,rtol=0,atol=0)
        # Genuine inverted robot/tray-to-ground impact remains visible.
        env.reset(); state=s._robot.data.root_link_state_w[:1].clone()
        state[:,2]=s.env_origins[:1,2]+.13; state[:,3:7]=state.new_tensor([0.,1.,0.,0.]); state[:,7:]=0
        s._robot.write_root_link_state_to_sim(state,one); impact=0.
        for _ in range(10):
            advance(1); impact=max(impact,float(s.ground_total_forces_world[0,env.collision_ids].norm(dim=-1).max()))
        assert impact>1.,impact
        env.reset()
        return dict(support_reaction_z_n=float(support.mean()),support_ground_max_n=float(ground.max()),
            rolling_displacement_max_m=float(motion.max()),ball_collision_relative_vx=relative_closing.tolist(),
            real_ground_collision_peak_n=impact,inter_env_isolation=True,boundary_translation=True,
            failure_timeout_precedence=True,terminal_ownership=True,named_jacobian_velocity=True)
    finally:
        env._reset_dofs,env._reset_root_states=original
