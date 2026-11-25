"""Opt-in Lab props. Explicit force-only grasp; no constraint reaction is reapplied."""
import numpy as np
import torch

from legged_gym.utils.pilot_figure1 import partner_motion


class InteractionScene:
    def __init__(self, simulator, config, cameras=False):
        from isaaclab.assets import RigidObject, RigidObjectCfg
        import isaaclab.sim as sim
        from pxr import UsdPhysics
        self.sim=simulator; self.config=config; self.active=False; self.scenario='nominal'
        self.enabled=False; self.time=0.; self.camera=None
        self.payload=RigidObject(RigidObjectCfg(prim_path='/World/Figure1Payload',
            spawn=sim.CuboidCfg(size=(config['handle_spacing_m'],.18,.07),
                rigid_props=sim.RigidBodyPropertiesCfg(), mass_props=sim.MassPropertiesCfg(mass=config['mass_kg']),
                collision_props=sim.CollisionPropertiesCfg(),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.15,.6,.35))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.,3.,.1))))
        # Only explicit grasp forces connect robot and box. Prevent an unlogged
        # second interaction through robot/payload collision geometry.
        pairs=UsdPhysics.FilteredPairsAPI.Apply(simulator._stage.GetPrimAtPath('/World/Figure1Payload'))
        pairs.CreateFilteredPairsRel().AddTarget('/World/envs/env_0/'+simulator._cfg.asset.name)
        self.proxy=RigidObject(RigidObjectCfg(prim_path='/World/Figure1Partner',
            spawn=sim.SphereCfg(radius=.045, rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                mass_props=sim.MassPropertiesCfg(mass=.1), collision_props=sim.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.85,.2,.12))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.,3.,1.))))
        sim.DomeLightCfg(intensity=1700.).func('/World/Figure1Light',sim.DomeLightCfg(intensity=1700.))
        if cameras:
            self.camera='/World/Figure1Camera'
            camera_cfg=sim.PinholeCameraCfg(focal_length=24.,clipping_range=(.05,100.))
            camera_cfg.func(self.camera,camera_cfg)
        self.robot_force=torch.zeros(1,3,device=simulator._device)
        self.partner_force=torch.zeros_like(self.robot_force)
        self.handles=torch.zeros(1,2,3,device=simulator._device)
        self.partner_position=torch.zeros_like(self.robot_force)
        self.partner_velocity=torch.zeros_like(self.robot_force)
        self.substep_records=[]

    def reset(self, ids):
        self.active=False
        self.robot_force.zero_(); self.partner_force.zero_()
        self.substep_records.clear()
        self.payload.reset(ids); self.proxy.reset(ids)
        self.payload.write_root_state_to_sim(self.payload.data.default_root_state.clone())
        self.proxy.write_root_state_to_sim(self.proxy.data.default_root_state.clone())
        self.payload.update(0.);self.proxy.update(0.)

    def _geometry(self):
        from isaaclab.utils.math import quat_apply
        state=self.payload.data.root_state_w
        local=state.new_tensor([[-self.config['handle_spacing_m']/2,0,0],
                                [self.config['handle_spacing_m']/2,0,0]])[None]
        offsets=quat_apply(state[:,None,3:7].expand(-1,2,-1),local)
        self.handles=state[:,None,:3]+offsets
        velocities=state[:,None,7:10]+torch.cross(state[:,None,10:13].expand_as(offsets),offsets,dim=-1)
        return state,offsets,velocities

    def attach(self, env):
        from legged_gym.envs.b1z1.directional_compliance import heading, rotate_yaw
        s=self.sim; self.yaw=heading(s.base_quat).clone()
        offset=rotate_yaw(s.ee_pos.new_tensor([[self.config['handle_spacing_m']/2,0,0]]),self.yaw)
        state=self.payload.data.default_root_state.clone()
        state[:,:3]=s.ee_pos+offset; state[:,3:7]=0
        state[:,3]=torch.cos(self.yaw/2);state[:,6]=torch.sin(self.yaw/2);state[:,7:]=0
        self.payload.write_root_state_to_sim(state);self.payload.update(0.)
        self._geometry();self.partner_origin=self.handles[:,1].clone()
        self.active=True

    def begin_interval(self, env, time):
        from legged_gym.envs.b1z1.directional_compliance import rotate_yaw
        self.time=time;self.substep_records.clear()
        if not self.enabled or time<self.config['attach_s']:
            self.robot_force.zero_();self.partner_force.zero_();return
        if not self.active: self.attach(env)
        _,_,velocities=self._geometry()
        displacement,velocity=partner_motion(time,self.scenario,self.config)
        self.partner_position=self.partner_origin+rotate_yaw(self.robot_force.new_tensor(displacement)[None],self.yaw)
        self.partner_velocity=rotate_yaw(self.robot_force.new_tensor(velocity)[None],self.yaw)
        def spring(point,vel,handle,hvel):
            force=self.config['spring_n_m']*(handle-point)+self.config['damping_ns_m']*(hvel-vel)
            return force*(self.config['force_limit_n']/force.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
        self.robot_force=spring(self.sim.ee_pos,self.sim._ee_vel,self.handles[:,0],velocities[:,0])
        self.partner_force=spring(self.partner_position,self.partner_velocity,self.handles[:,1],velocities[:,1])
        pose=self.proxy.data.default_root_state.clone();pose[:,:3]=self.partner_position;pose[:,7:]=0
        self.proxy.write_root_state_to_sim(pose)
        # Zero-order hold at the control boundary: the exact same world force
        # drives ComplianceTask.begin and every executed physics substep.

    def before_substep(self):
        self.payload.permanent_wrench_composer.reset()
        if self.active:
            state,offsets,_=self._geometry()
            forces=torch.stack((-self.robot_force,-self.partner_force),1)
            torque=torch.cross(offsets,forces,dim=-1).sum(1)
            self.payload.permanent_wrench_composer.set_forces_and_torques(
                forces=forces.sum(1)[:,None],torques=torque[:,None],is_global=True)
            tcp=self.sim._robot.data.body_link_pos_w[:,self.sim._gripper_index]
            self.substep_records.append(torch.cat((tcp,self.handles.flatten(1),forces.sum(1),
                                                   torque,self.robot_force,self.partner_force),-1).clone())
        self.payload.write_data_to_sim()

    def after_substep(self, dt):
        self.payload.update(dt);self.proxy.update(dt)

    def image(self, env):
        if self.camera is None: return None
        import ctypes
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_buffer
        if not hasattr(self,'viewport'):
            self.viewport=get_active_viewport()
            if self.viewport is None:
                from omni.kit.viewport.window import ViewportWindow
                self.viewport_window=ViewportWindow('Figure 1',width=960,height=640)
                self.viewport=self.viewport_window.viewport_api
            self.viewport.camera_path=self.camera
            self.viewport.set_texture_resolution((960,640))
            # Lab disables viewport updates headless even when offscreen sensors
            # are enabled; this optional diagnostic viewport needs explicit updates.
            self.viewport.updates_enabled=True
        center=env.simulator.base_pos[0].cpu().numpy().copy();center[2]=.6
        set_camera_view(center+np.array([2.,-3.,1.7]),center+np.array([.5,0,0]),self.camera,self.viewport)
        frames=[]
        def captured(buffer,size,width,height,fmt):
            if size!=width*height*4: raise RuntimeError(f'Unsupported viewport pixel format: {fmt}')
            get_pointer=ctypes.pythonapi.PyCapsule_GetPointer
            get_pointer.restype=ctypes.c_void_p;get_pointer.argtypes=[ctypes.py_object,ctypes.c_char_p]
            array=(ctypes.c_ubyte*size).from_address(get_pointer(buffer,None))
            frames.append(np.frombuffer(array,dtype=np.uint8).reshape(height,width,4)[:,:,:3].copy())
        # Kit viewport capture avoids the installed RGB annotator's uint64
        # dependency-graph bug. Render only: no physics/history/reference calls.
        capture=capture_viewport_to_buffer(self.viewport,captured)
        for _ in range(30):
            self.sim._sim.render()
            if frames: return frames[0]
        raise RuntimeError('Viewport capture did not complete in 30 render-only updates')


class NoPropScene:
    """Nominal validation contains no payload or partner physics/visual prims."""
    def __init__(self, simulator, config, cameras=False):
        self.sim=simulator;self.payload=None;self.camera=None;self.enabled=False
        self.robot_force=torch.zeros(1,3,device=simulator._device)
        self.partner_force=torch.zeros_like(self.robot_force)
        self.partner_position=torch.full_like(self.robot_force,float('nan'))
        self.partner_velocity=self.partner_position.clone()
        self.handles=torch.full((1,2,3),float('nan'),device=simulator._device)
        self.substep_records=[]
        if cameras:
            import isaaclab.sim as sim
            sim.DomeLightCfg(intensity=1700.).func('/World/Figure1Light',sim.DomeLightCfg(intensity=1700.))
            self.camera='/World/Figure1Camera'
            camera_cfg=sim.PinholeCameraCfg(focal_length=24.)
            camera_cfg.func(self.camera,camera_cfg)
    def reset(self, ids): self.robot_force.zero_()
    def begin_interval(self, env, time): self.robot_force.zero_()
    def before_substep(self): pass
    def after_substep(self, dt): pass
    image=InteractionScene.image
