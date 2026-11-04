"""Shared evaluation-only settings; retain each robot's PACT task contracts."""


def liquid_evaluation_config(robot, mount_height):
    class LiquidConfig(robot):
        class env(robot.env):
            num_envs = 1
            env_spacing = 2.0
            use_liquid = True
            liquid_payload = True

        class liquid:
            liquid_type = 'water'
            liquid_volume = 4  # nominal legacy fill preset, litres
            liquid_tank = 'default'
            particle_size = .010  # same resolution/contact settings as collection
            substeps = 4  # internal physics/SPH substeps per control.dt / control.decimation
            contact_geometry = 'boxes'
            cavity_coupling = 'planes'
            contact_softness_m = .005
            # Internal timestep = control.dt / control.decimation / substeps.
            lid_overlap = .007
            base_link = 'base'
            tank_opacity = .25
            fluid_color = (.05,.35,1.)
            particle_markers = True  # actual-position instances avoid rasterizer glass/particle occlusion

        class terrain(robot.terrain):
            mesh_type = 'plane'
            curriculum = False
            measure_heights = False
            reset_out_of_bounds = True

        class commands(robot.commands):
            curriculum = False
            heading_command = False
            resampling_time = 5.

            class ranges(robot.commands.ranges):
                lin_vel_x = [-.5,.5]
                lin_vel_y = [-.2,.2]
                ang_vel_yaw = [-.5,.5]

        class control(robot.control):
            randomize_pact_weights = False
            use_tradeoff_curriculum = False

        class domain_rand(robot.domain_rand):
            use_domainrand_curriculum = False
            randomize_base_mass = False
            randomize_com_displacement = False
            randomize_friction = False
            randomize_pd_gain = False
            randomize_motor_strength = False
            randomize_ctrl_delay = False
            # Use the simulator property APIs with degenerate midpoint ranges,
            # resolved from the selected checkpoint's saved training config.
            randomize_joint_armature = True
            randomize_joint_friction = True
            randomize_joint_stiffness = True
            randomize_joint_damping = True
            push_robots = False

        class viewer(robot.viewer):
            rendered_envs_idx = [0]
            add_camera = False

    LiquidConfig.liquid.mount_height = mount_height
    return LiquidConfig
