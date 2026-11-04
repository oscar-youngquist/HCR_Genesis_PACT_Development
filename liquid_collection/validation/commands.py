"""Validation-only commands, applied before the normal observation builder."""
def install(env, aggressive=False):
    import torch
    from liquid_collection.physics import rotate
    original = env.compute_observations
    def observations():
        age = env.episode_length_buf * env.dt
        forward = (age >= 1.6) & (age < 3.)
        turning = age >= 4.2
        if aggressive:
            # One second settling, then ten seconds of bounded aggressive commands.
            phase=((age-1.).clamp_min(0).floor().long() % 5)
            active=age>=1.
            forward=active & ((phase==0)|(phase==3)|(phase==4))
            turning=active & ((phase==3)|(phase==4))
        bounds = env.command_ranges
        env.commands[:, 0] = torch.where(forward | turning, .4, 0.).clamp(*bounds['lin_vel_x'])
        env.commands[:, 1] = max(bounds['lin_vel_y'][0], min(0., bounds['lin_vel_y'][1]))
        yaw=torch.where(turning, .4, 0.)
        if aggressive:
            env.commands[:,0]=torch.where(forward,.5,torch.where(active & (phase==2),-.5,0.)).clamp(*bounds['lin_vel_x'])
            env.commands[:,1]=torch.where(turning,torch.where(phase==3,.25,-.25),0.).clamp(*bounds['lin_vel_y'])
            yaw=torch.where(turning,torch.where(phase==3,.8,-.8),0.)
        if env.cfg.commands.heading_command:
            q = env.simulator._robot.get_quat()
            axis = torch.zeros((env.num_envs, 3), device=env.device); axis[:, 0] = 1
            direction = rotate(q, axis)
            heading = torch.atan2(direction[:, 1], direction[:, 0])
            target = heading + 2*yaw
            target = (target + torch.pi) % (2*torch.pi) - torch.pi
            env.commands[:, 3] = target.clamp(*bounds['heading'])
            error = (env.commands[:, 3]-heading+torch.pi) % (2*torch.pi)-torch.pi
            env.commands[:, 2] = (.5*error).clamp(*bounds['ang_vel_yaw'])
        else:
            env.commands[:, 2] = yaw.clamp(*bounds['ang_vel_yaw'])
        original()
    env.compute_observations = observations
