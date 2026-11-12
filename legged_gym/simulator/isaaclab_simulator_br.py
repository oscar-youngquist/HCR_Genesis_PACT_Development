"""Opt-in BR scene: fixed-joint dynamic tray and independent sliding rigid box.

No QP, learned controller, or physics-loss code. Tray is an articulation link,
not a pose follower. Its real mass/inertia are independent of torso mass DR.
Box dimensions vary across a seeded bank BEFORE physics initialization;
mass, inertia, friction and initial conditions change only on eligible resets.
"""
import torch
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT
from legged_gym.simulator.reset_randomization_cadence import ResetRandomizationCadence
from rsl_rl.beyond_robustness_payload import (SlidingPayloadConfig, cuboid_inertia,
    load_labels, spawn_state, named_indices, rotate_wxyz, payload_geometry)


class IsaacLabSimulatorBR(IsaacLabSimulator_PACT):
    def __init__(self, cfg, sim_params, device, headless, payload_cfg=None):
        self.payload_cfg = payload_cfg or SlidingPayloadConfig(seed=int(getattr(cfg,'seed',0)))
        self.payload_rng = torch.Generator(device=device).manual_seed(self.payload_cfg.seed)
        self.payload_progress = 0.
        super().__init__(cfg, sim_params, device, headless)
        self.payload_cadence = ResetRandomizationCadence(self._num_envs, device,
            max(1, int(getattr(cfg.domain_rand,'reset_resample_episodes',1))))
        self.payload_mass = torch.zeros(self._num_envs, device=device)
        self.payload_friction = torch.zeros_like(self.payload_mass)
        self.payload_offset = torch.zeros(self._num_envs,3,device=device)
        self.payload_relative_velocity = torch.zeros_like(self.payload_offset)
        self.set_payload_progress(0.)

    def _create_scene_components(self, source_env_path):
        from pxr import UsdPhysics, UsdGeom, Gf
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject, RigidObjectCfg
        c = self.payload_cfg
        root = source_env_path+'/'+self._cfg.asset.name
        base = root+'/'+self._cfg.asset.base_link_name
        if not self._stage.GetPrimAtPath(base).IsValid():
            raise ValueError(f'Cannot attach tray: missing torso {base}')
        tray = root+'/br_tray'
        prim = UsdGeom.Xform.Define(self._stage,tray)
        # Robot prim carries its own spawn transform; tray is in robot frame.
        prim.AddTranslateOp().Set(Gf.Vec3d(0,0,c.tray_height))
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        sim.define_rigid_body_properties(tray, sim.RigidBodyPropertiesCfg(disable_gravity=False))
        material = sim.RigidBodyMaterialCfg(static_friction=c.tray_friction,
            dynamic_friction=c.tray_friction, restitution=0., friction_combine_mode='multiply')
        x,y,z = c.tray_size; t,h = c.wall_thickness,c.wall_height
        pieces = [((x,y,z),(0,0,0)), ((t,y,h),((x-t)/2,0,(z+h)/2)),
                  ((t,y,h),(-(x-t)/2,0,(z+h)/2)),
                  ((x-2*t,t,h),(0,(y-t)/2,(z+h)/2)),
                  ((x-2*t,t,h),(0,-(y-t)/2,(z+h)/2))]
        volumes = torch.tensor([a*b*d for (a,b,d),_ in pieces],dtype=torch.float64)
        masses = c.tray_mass*volumes/volumes.sum()
        centres = torch.tensor([p for _,p in pieces],dtype=torch.float64)
        com = (masses[:,None]*centres).sum(0)/c.tray_mass
        inertia = torch.zeros(3,3,dtype=torch.float64)
        for i,(size,pos) in enumerate(pieces):
            shape = sim.CuboidCfg(size=size,collision_props=sim.CollisionPropertiesCfg(),
                physics_material=material, visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.2,.3,.5)))
            shape.func(tray+f'/part_{i}',shape,translation=pos)
            delta = centres[i]-com
            inertia += cuboid_inertia(masses[i:i+1],size)[0] + masses[i]*(delta.square().sum()*torch.eye(3)-delta[:,None]*delta[None,:])
        mass = UsdPhysics.MassAPI.Apply(prim.GetPrim())
        mass.CreateMassAttr(c.tray_mass)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*com.tolist()))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia.diag().tolist()))
        sim.activate_contact_sensors(tray)
        joint = UsdPhysics.FixedJoint.Define(self._stage,root+'/br_tray_joint')
        joint.CreateBody0Rel().SetTargets([base]); joint.CreateBody1Rel().SetTargets([tray])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0,0,c.tray_height))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
        joint.CreateCollisionEnabledAttr(False)

    def _replicate_scene_physics(self):
        # Installed MultiAssetSpawner requires parsing heterogeneous colliders.
        return False

    def _scene_physx_options(self):
        return {'gpu_total_aggregate_pairs_capacity':self._cfg.sim.physx.gpu_total_aggregate_pairs_capacity}

    def _create_cloned_scene_components(self, prim_paths):
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from pxr import UsdGeom, Gf
        import math
        # The legacy cloner places all source copies at zero until first reset.
        # With heterogeneous bodies this causes O(N²) broadphase pairs during
        # SimulationContext.reset(). Stage separated assets BEFORE initialization;
        # normal robot/payload reset still sets the inherited terrain poses.
        width=math.ceil(math.sqrt(self._num_envs))
        for i,path in enumerate(prim_paths):
            xform=UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
            ops=[op for op in xform.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate]
            op=ops[0] if ops else xform.AddTranslateOp()
            op.Set(Gf.Vec3d(3*(i//width),3*(i%width),5.))
        c=self.payload_cfg
        bank,ids=payload_geometry(c,self._num_envs)
        self.box_size_bank=bank.tolist(); self.box_size_ids=ids.tolist()
        self.box_size=bank[ids].to(self._device)
        shapes=[sim.CuboidCfg(size=tuple(size), rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim.MassPropertiesCfg(mass=c.mass_initial[0]),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.002,rest_offset=0.),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=c.friction_range[0],
                dynamic_friction=c.friction_range[0],restitution=0.,friction_combine_mode='multiply'),
            activate_contact_sensors=True) for size in self.box_size_bank]
        # Supported pre-initialization spawner copies bank geometry, not runtime
        # USD scaling. Its deterministic traversal must match canonical env IDs.
        paths=sim.find_matching_prim_paths('/World/envs/env_.*')
        if paths!=prim_paths: raise RuntimeError('Unexpected multi-asset environment traversal order')
        shape=sim.MultiAssetSpawnerCfg(assets_cfg=shapes,random_choice=False)
        self.payload = RigidObject(RigidObjectCfg(prim_path='/World/envs/env_.*/br_box',spawn=shape,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,1.))))

    def _contact_sensor_options(self):
        # Force ground filtering even for supplied older task configs. Payload
        # support stays physical; only the observation pair filter excludes it.
        options = super()._contact_sensor_options()
        if not self._ground_filters:
            raise ValueError('BR requires terrain-filtered contact sensing (sim.grf.force_mode)')
        return options

    def _randomize_friction(self, env_ids):
        # PhysX packs shapes by link. Installed Isaac Lab events.py uses this
        # same supported per-link rigid view to resolve shape counts.
        if not hasattr(self,'_tray_shape_slice'):
            view = self._robot.root_physx_view
            counts = [self._robot._physics_sim_view.create_rigid_body_view(path).max_shapes
                      for path in view.link_paths[0]]
            if sum(counts) != view.max_shapes: raise RuntimeError('Unexpected articulation shape packing')
            idx = self._robot.body_names.index('br_tray')
            self._tray_shape_slice = slice(sum(counts[:idx]),sum(counts[:idx+1]))
        super()._randomize_friction(env_ids)
        props = self._robot.root_physx_view.get_material_properties().clone()
        ids = env_ids.cpu()
        props[ids,self._tray_shape_slice,:2] = self.payload_cfg.tray_friction
        props[ids,self._tray_shape_slice,2] = 0.
        self._robot.root_physx_view.set_material_properties(props,ids)

    def set_payload_progress(self, progress):
        self.payload_progress = float(progress)
        self.payload_cadence.update_ranges(self.payload_cfg.ranges(progress))

    def _uniform(self, shape, low, high):
        return low+(high-low)*torch.rand(shape,device=self._device,generator=self.payload_rng)

    def reset_payload(self, env_ids):
        """Call AFTER robot reset; active environments retain state and samples."""
        c = self.payload_cfg
        if not getattr(self,'_resume_samples',False): self.payload_cadence.begin_reset(env_ids)
        ranges = c.ranges(self.payload_progress)
        for name in ranges:
            ids = self.payload_cadence.select(name,env_ids)
            if len(ids)==0: continue
            low,high = ranges[name]
            if name=='mass': self.payload_mass[ids] = self._uniform((len(ids),),low,high)
            elif name=='friction': self.payload_friction[ids] = self._uniform((len(ids),),low,high)
            elif name=='displacement':
                usable = torch.tensor(c.tray_size[:2],device=self._device)/2-c.wall_thickness-self.box_size[ids,:2]/2-c.spawn_clearance
                self.payload_offset[ids,:2] = self._uniform((len(ids),2),-1.,1.)*usable.clamp(max=high)
            else: self.payload_relative_velocity[ids,:2] = self._uniform((len(ids),2),-high,high)
        self.payload_offset[env_ids,2] = c.tray_height+c.tray_size[2]/2+self.box_size[env_ids,2]/2+c.spawn_clearance
        view=self.payload.root_physx_view; cpu_ids=env_ids.cpu()
        masses=view.get_masses().clone(); masses[cpu_ids,0]=self.payload_mass[env_ids].cpu()
        inertia=view.get_inertias().clone(); inertia[cpu_ids]=cuboid_inertia(self.payload_mass[env_ids],self.box_size[env_ids]).reshape(-1,9).cpu()
        props=view.get_material_properties().clone(); props[cpu_ids,:,:2]=self.payload_friction[env_ids,None,None].cpu(); props[cpu_ids,:,2]=0.
        view.set_masses(masses,cpu_ids); view.set_inertias(inertia,cpu_ids); view.set_material_properties(props,cpu_ids)
        # Return read-back actual values, not desired labels.
        self.payload_mass[env_ids]=view.get_masses()[cpu_ids,0].to(self._device)
        d=self._robot.data
        state=spawn_state(d.root_link_quat_w[env_ids],d.root_link_pos_w[env_ids],
            d.root_link_lin_vel_w[env_ids],d.root_link_ang_vel_w[env_ids],
            self.payload_offset[env_ids],self.payload_relative_velocity[env_ids])
        self.payload.write_root_state_to_sim(state,env_ids)
        self.payload.reset(env_ids)
        for buf in (self.ground_normal_forces_world,self.ground_total_forces_world): buf[env_ids]=0

    def payload_labels(self):
        d,p=self._robot.data,self.payload.data
        # Isaac Lab 0.54's pose writer updates root_link_pose_w immediately but
        # not an already-read root_com_pose_w cache until time advances. Compose
        # the CoM explicitly so post-reset labels are correct without stepping.
        com=p.root_link_pos_w+rotate_wxyz(p.root_link_quat_w,p.body_com_pos_b[:,0])
        return load_labels(d.root_link_quat_w,d.root_link_pos_w,d.root_link_lin_vel_w,
            d.root_link_ang_vel_w,com,p.root_com_lin_vel_w,self.payload_mass)

    def _check_base_pos_out_of_bound(self):
        before=self._robot.data.root_link_pos_w.clone()
        super()._check_base_pos_out_of_bound()
        delta=self._robot.data.root_link_pos_w-before
        ids=delta.ne(0).any(-1).nonzero().flatten()
        if len(ids):
            pose=self.payload.data.root_link_state_w[ids,:7].clone()
            pose[:,:3]+=delta[ids]
            self.payload.write_root_link_pose_to_sim(pose,ids)

    def step(self, actions):
        super().step(actions)
        self.payload.update(self._control_dt)

    def _compute_torques(self, actions):
        torque=super()._compute_torques(actions)
        # Reward-only request, BEFORE action clipping. Same delayed command
        # identity and pre-integration state as the last executed PD command.
        raw=getattr(self,'_br_raw_actions',actions)
        self.br_requested_torque=self._motor_strength*(self._kp_scale*self._p_gains*
            (raw*self._cfg.control.action_scale+self.default_dof_pos-self.dof_pos)
            -self._kd_scale*self._d_gains*self.dof_vel)
        return torque

    def realized_parameters(self):
        """CPU PhysX readbacks at reset only; caller caches this owned GPU pack."""
        robot=self._robot; view=robot.root_physx_view
        legs=[f'{leg}_{part}' for leg in ('FR','FL','RR','RL') for part in ('hip','thigh','calf')]
        ids=named_indices(robot.body_names,legs); base=robot.body_names.index(self._cfg.asset.base_link_name)
        masses=view.get_masses().to(self._device); com=view.get_coms().to(self._device)
        # _p/_d_gains are initialized in articulation order; labels/controller
        # use the same canonical permutation here and in _compute_torques.
        return dict(joint_kp=self._kp_scale*self._p_gains, joint_kd=self._kd_scale*self._d_gains,
            motor_strength=self._motor_strength,leg_link_mass=masses[:,ids],
            leg_link_com=com[:,ids,:3].reshape(self._num_envs,-1),base_mass=masses[:,base,None],
            base_com=com[:,base,:3],tray_mass=masses[:,robot.body_names.index('br_tray'),None])

    def sampling_state_dict(self):
        """Checkpoint sampled properties, not PhysX contacts/trajectory state."""
        from copy import deepcopy
        view=self._robot.root_physx_view
        return deepcopy(dict(geometry_version=2,box_size_bank=self.box_size_bank,box_size_ids=self.box_size_ids,rng=self.payload_rng.get_state(),
            payload_cadence=self.payload_cadence.__dict__,
            robot_cadence=self._reset_cadence.__dict__ if hasattr(self,'_reset_cadence') else None,
            tensors={key:getattr(self,key).clone() for key in (
                'payload_mass','payload_friction','payload_offset','payload_relative_velocity',
                '_kp_scale','_kd_scale','_motor_strength','_added_base_mass','_base_com_bias',
                '_joint_armature','_joint_friction','_joint_stiffness','_joint_damping','_friction_values')},
            joint_properties={name:getattr(self._robot.data,'joint_'+name).clone()
                for name in ('armature','friction_coeff','stiffness','damping')},
            properties={key:getattr(view,'get_'+key)().clone() for key in ('masses','coms','inertias','material_properties')}))

    def load_sampling_state_dict(self, state):
        from copy import deepcopy
        if (state.get('geometry_version')!=2 or state.get('box_size_bank')!=self.box_size_bank
                or state.get('box_size_ids')!=self.box_size_ids):
            raise ValueError('Payload creation-time geometry differs; recreate with checkpoint payload config/seed')
        self.payload_rng.set_state(state['rng'].cpu())
        self.payload_cadence.__dict__.update(deepcopy(state['payload_cadence']))
        for key,value in state['tensors'].items(): getattr(self,key).copy_(value.to(self._device))
        view=self._robot.root_physx_view; ids=torch.arange(self._num_envs)
        for key,value in state['properties'].items(): getattr(view,'set_'+key)(value.cpu(),ids)
        for name,value in state['joint_properties'].items():
            method='write_joint_'+('friction_coefficient' if name=='friction_coeff' else name)+'_to_sim'
            getattr(self._robot,method)(value.to(self._device))
        if state['robot_cadence'] is not None:
            cadence=self._reset_sampling_cadence()
            if cadence is None: raise ValueError('Reset sampling cadence differs from checkpoint')
            cadence.__dict__.update(deepcopy(state['robot_cadence']))
        self._resume_samples=True

    def reset_idx(self, env_ids):
        if getattr(self,'_resume_samples',False):
            # Fresh-state resume installs saved properties, not stale histories.
            # Prevent first reset from advancing restored episode counters.
            cadence=getattr(self,'_reset_cadence',None)
            if cadence is not None: cadence.started[env_ids]=False
        super().reset_idx(env_ids)

    def _reset_sample_ids(self, name, env_ids):
        if getattr(self,'_resume_samples',False): return env_ids[:0]
        return super()._reset_sample_ids(name,env_ids)

    def _reset_domain_randomization(self, env_ids):
        if not getattr(self,'_resume_samples',False): super()._reset_domain_randomization(env_ids)
