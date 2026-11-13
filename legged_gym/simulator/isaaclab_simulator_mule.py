"""MULE's physical rolling spheres, not BR box physics or HardPACT learning.

The BR subclass supplies only fixed-tray construction, ground filtering,
named property readback and robot reset-cadence handling. Its constructor,
box creation/sampling/labels/step are NOT called. All spheres are independent
rigid bodies. Physics property APIs are CPU-backed: four batched read/write
calls, never one call per environment; mass changes necessarily synchronize.
"""
from copy import deepcopy
import math
import torch
from legged_gym.simulator.isaaclab_simulator_br import IsaacLabSimulatorBR
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT
from legged_gym.envs.go2.go2_mule.rolling_payload import (
    sphere_geometry,sphere_inertia,sample_offsets,ball_spawn_state,ball_motion,due_mass_changes)
from rsl_rl.beyond_robustness_payload import inverse_rotate_wxyz, named_indices


def canonical_leg_jacobians(jacobian, body_indices, joint_indices, quat, *, fixed_base=False):
    """PhysX world COM Jacobian -> named leg COM Jacobians in torso axes.

    Floating base has six leading world-twist columns and includes root body.
    Fixed base omits root body and has only joint columns. The endpoint is the
    named foot link COM, not a contact point inferred from privileged sensors.
    """
    offset=0 if fixed_base else 6
    bodies=torch.as_tensor(body_indices,device=jacobian.device)-(1 if fixed_base else 0)
    cols=torch.as_tensor(joint_indices,device=jacobian.device).reshape(4,3)+offset
    j=jacobian[:,bodies,:3,:]
    j=j.gather(-1,cols[None,:,None,:].expand(j.shape[0],-1,3,-1))
    return inverse_rotate_wxyz(quat[:,None,None,:].expand(-1,4,3,-1),j.transpose(-1,-2)).transpose(-1,-2)


class IsaacLabSimulatorMULE(IsaacLabSimulatorBR):
    def _scene_physx_options(self):
        return {**super()._scene_physx_options(),
            'gpu_found_lost_pairs_capacity':getattr(self._cfg.sim.physx,'gpu_found_lost_pairs_capacity',2**21)}

    def __init__(self,cfg,sim_params,device,headless,payload_cfg):
        self.payload_cfg=payload_cfg
        self.payload_rng=torch.Generator(device=device).manual_seed(payload_cfg.seed)
        self.payload_progress=payload_cfg.initial_severity
        self.balls=[]; self.ball_sensors=[]
        IsaacLabSimulator_PACT.__init__(self,cfg,sim_params,device,headless)
        from legged_gym.simulator.ground_contact_forces import GroundContactForces
        n=self._num_envs
        self.ball_mass=torch.zeros(n,4,device=device)
        self.ball_friction=torch.zeros_like(self.ball_mass)
        self.ball_restitution=torch.zeros_like(self.ball_mass)
        self.ball_offsets=torch.zeros(n,4,3,device=device)
        self.ball_relative_velocity=torch.zeros_like(self.ball_offsets)
        self.ball_relative_omega=torch.zeros_like(self.ball_offsets)
        self.mass_elapsed=torch.zeros(n,device=device)
        self.mass_change_count=torch.zeros(n,dtype=torch.long,device=device)
        self.payload_reaction_world=torch.zeros(n,3,device=device)
        self.payload_active=torch.zeros(n,dtype=torch.bool,device=device)
        self.realized_severity=torch.zeros(n,device=device)
        self._ball_contacts=[GroundContactForces(sensor.contact_physx_view,sim_params['dt'],device)
                             for sensor in self.ball_sensors]
        self._p_gains=self._p_gains[self._dof_indices]; self._d_gains=self._d_gains[self._dof_indices]
        self._mule_feet=named_indices(self._robot.body_names,[f'{f}_foot' for f in ('FR','FL','RR','RL')])
        named_indices(self._robot.joint_names,cfg.asset.dof_names)
        self.last_requested_torque=torch.zeros(n,12,device=device)

    def _create_cloned_scene_components(self,paths):
        from pxr import UsdGeom,Gf,Usd,UsdPhysics
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject,RigidObjectCfg
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        # Same pre-initialization separation precaution as BR: no coincident
        # cloned robots/payloads during PhysX initialization/broadphase building.
        width=math.ceil(math.sqrt(self._num_envs))
        for i,path in enumerate(paths):
            xf=UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
            ops=[op for op in xf.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate]
            (ops[0] if ops else xf.AddTranslateOp()).Set(Gf.Vec3d(3*(i//width),3*(i%width),5.))
        c=self.payload_cfg; bank,ids=sphere_geometry(c,self._num_envs)
        self.radius_bank=bank.tolist(); self.radius_ids=ids.tolist(); self.ball_radius=bank[ids].to(self._device)
        if c.mode=='nominal': return  # no colliders, bodies, or zero-mass substitutes
        if sim.find_matching_prim_paths('/World/envs/env_.*')!=paths: raise RuntimeError('Unexpected environment traversal order')
        # PhysX requires each filter expression to resolve ONE body per env,
        # not a wildcard over all links (which silently gives invalid filters).
        robot_root=paths[0]+'/'+self._cfg.asset.name
        filters=['/World/envs/env_.*'+p.GetPath().pathString[len(paths[0]):]
                 for p in Usd.PrimRange(self._stage.GetPrimAtPath(robot_root)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not filters: raise RuntimeError('No physical robot links for payload contact filtering')
        for ball in range(4):
            shapes=[sim.SphereCfg(radius=float(row[ball]),
                rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim.MassPropertiesCfg(mass=c.mass_floor_kg),
                collision_props=sim.CollisionPropertiesCfg(contact_offset=.002,rest_offset=0.),
                physics_material=sim.RigidBodyMaterialCfg(static_friction=c.friction_range[0],
                    dynamic_friction=c.friction_range[0],restitution=c.restitution_range[0],
                    friction_combine_mode='multiply',restitution_combine_mode='max'),activate_contact_sensors=True)
                for row in self.radius_bank]
            path=f'/World/envs/env_.*/mule_ball_{ball}'
            shape=sim.MultiAssetSpawnerCfg(assets_cfg=shapes,random_choice=False,activate_contact_sensors=True)
            self.balls.append(RigidObject(RigidObjectCfg(prim_path=path,spawn=shape,
                init_state=RigidObjectCfg.InitialStateCfg(pos=((1 if ball<2 else -1)*.12,(1 if ball%2==0 else -1)*.08,.4)))))
            # Only robot/tray contact enters critic disturbance. Ball-ball and
            # terrain forces cannot masquerade as loads applied to the robot.
            self.ball_sensors.append(ContactSensor(ContactSensorCfg(prim_path=path,
                update_period=self._sim_params['dt'],history_length=0,
                filter_prim_paths_expr=filters,
                track_contact_points=True,track_friction_forces=True,max_contact_data_count_per_prim=32)))

    def set_payload_severity(self,value):
        self.payload_cfg.mass_range(value)  # validation; no current property mutation
        self.payload_progress=float(value)
        cadence=self._reset_sampling_cadence()
        if cadence is not None: cadence.update_ranges({'mule_payload':(value,)})

    def _uniform(self,shape,lo,hi):
        return lo+(hi-lo)*torch.rand(shape,device=self._device,generator=self.payload_rng)

    def _write_ball_properties(self,ids,*,material=False,material_ids=None):
        cpu=ids.cpu()
        # Materials follow the persistent cadence, whereas mass/inertia MUST
        # change each episode. Fresh resume explicitly restores both families.
        material_ids=ids if material_ids is None else material_ids
        material=material and len(material_ids)>0
        # Batch GPU/CPU crossings across the four balls; retain PhysX readback
        # as the authoritative realized mass/material labels.
        masses_cpu=self.ball_mass[ids].cpu()
        inertias_cpu=sphere_inertia(self.ball_mass[ids],self.ball_radius[ids]).flatten(-2).cpu()
        realized_masses=[]; realized_materials=[]
        if material:
            material_cpu=material_ids.cpu()
            friction_cpu=self.ball_friction[material_ids].cpu()
            restitution_cpu=self.ball_restitution[material_ids].cpu()
        for k,ball in enumerate(self.balls):
            view=ball.root_physx_view
            mass=view.get_masses().clone(); mass[cpu,0]=masses_cpu[:,k]
            inertia=view.get_inertias().clone()
            inertia[cpu]=inertias_cpu[:,k]
            view.set_masses(mass,cpu); view.set_inertias(inertia,cpu)
            if material:
                props=view.get_material_properties().clone()
                props[material_cpu,:,:2]=friction_cpu[:,k,None,None]
                props[material_cpu,:,2]=restitution_cpu[:,k,None]
                view.set_material_properties(props,material_cpu)
            realized_masses.append(view.get_masses()[cpu,0].clone())
            if material:
                realized_materials.append(view.get_material_properties()[material_cpu,0,1:3].clone())
        self.ball_mass[ids]=torch.stack(realized_masses,1).to(self._device)
        if material:
            realized=torch.stack(realized_materials,1).to(self._device)
            self.ball_friction[material_ids]=realized[:,:,0]
            self.ball_restitution[material_ids]=realized[:,:,1]

    def reset_payload(self,ids):
        if not len(ids): return
        self.payload_reaction_world[ids]=0
        if not self.balls: return
        c=self.payload_cfg; p=self.payload_progress
        if not getattr(self,'_resume_samples',False):
            due=self._reset_sample_ids('mule_payload',ids)
            self.payload_active[due]=p>0
            self.realized_severity[due]=p
            self.ball_friction[due]=self._uniform((len(due),4),*c.friction_at(p))
            self.ball_restitution[due]=self._uniform((len(due),4),*c.restitution_range)
            self.ball_offsets[due]=sample_offsets(c,self.ball_radius[due],self.payload_rng,p)
            self.ball_relative_velocity[due]=self._uniform((len(due),4,3),-1.,1.)*c.relative_linear_velocity_max*p
            self.ball_relative_omega[due]=self._uniform((len(due),4,3),-1.,1.)*c.relative_angular_velocity_max*p
            # Episode-initial masses are not persistent material properties.
            # A periodic (larger-range) draw must never leak into a new episode.
            severity=self.realized_severity[ids,None]
            lo=(c.light_mass_range[0]+severity*(c.reset_mass_range[0]-c.light_mass_range[0])).clamp_min(c.mass_floor_kg)
            hi=(c.light_mass_range[1]+severity*(c.reset_mass_range[1]-c.light_mass_range[1])).clamp_min(c.mass_floor_kg)
            self.ball_mass[ids]=lo+(hi-lo)*self._uniform((len(ids),4),0.,1.)
            self.mass_elapsed[ids]=0; self.mass_change_count[ids]=0
        material_ids=ids if getattr(self,'_resume_samples',False) or not getattr(self,'collection_optimizations',True) else due
        self._write_ball_properties(ids,material=True,material_ids=material_ids)
        d=self._robot.data
        states=ball_spawn_state(d.root_link_quat_w[ids],d.root_link_pos_w[ids],d.root_link_lin_vel_w[ids],
            d.root_link_ang_vel_w[ids],self.ball_offsets[ids],self.ball_relative_velocity[ids],self.ball_relative_omega[ids])
        for k,(ball,sensor,contact) in enumerate(zip(self.balls,self.ball_sensors,self._ball_contacts)):
            # Supported PhysX gravity flags + nonoverlapping parking well below
            # terrain: positive-mass bodies, no robot/terrain/ball interaction.
            # Activation changes only on reset, never during an active episode.
            view=ball.root_physx_view; cpu=ids.cpu()
            disabled=view.get_disable_gravities().clone()
            disabled[cpu,0]=(~self.payload_active[ids]).to(device='cpu',dtype=disabled.dtype)
            view.set_disable_gravities(disabled,cpu)
            parked=~self.payload_active[ids]
            parking=states[:,k].clone(); parking[:,2]=-100.-k; parking[:,7:]=0
            states[:,k]=torch.where(parked[:,None],parking,states[:,k])
            ball.write_root_state_to_sim(states[:,k],ids); ball.reset(ids); sensor.reset(ids); contact.reset(ids)

    def periodic_mass_update(self,dt):
        if not self.balls: return
        due,elapsed=due_mass_changes(self.mass_elapsed,dt,self.payload_cfg.mass_change_interval_s)
        self.mass_elapsed.copy_(torch.where(self.payload_active,elapsed,0.))
        ids=(due&self.payload_active).nonzero().flatten()
        if len(ids):
            c=self.payload_cfg; p=self.realized_severity[ids,None]
            lo=(c.light_mass_range[0]+p*(c.periodic_mass_range[0]-c.light_mass_range[0])).clamp_min(c.mass_floor_kg)
            hi=(c.light_mass_range[1]+p*(c.periodic_mass_range[1]-c.light_mass_range[1])).clamp_min(c.mass_floor_kg)
            self.ball_mass[ids]=lo+(hi-lo)*self._uniform((len(ids),4),0.,1.)
            self._write_ball_properties(ids)
            self.mass_change_count[ids]+=1
        # Discontinuous parameter change, NOT momentum conserving: pose and
        # linear/angular velocity remain unchanged; no extra physical impulse.

    def _compute_torques(self,actions):
        if getattr(self,'_mule_position_target',None) is None:
            torque=IsaacLabSimulator_PACT._compute_torques(self,actions)
        else:
            # Already scaled ONCE by compose_actions; fresh measured PD each
            # physics substep. No reverse scaling or extra action clipping.
            torque=self._motor_strength*(self._kp_scale*self._p_gains*(self._mule_position_target-self.dof_pos)
                -self._kd_scale*self._d_gains*self.dof_vel)
        self.last_requested_torque=torque.detach().clone()
        return torque

    def step_position_targets(self,targets):
        self._mule_position_target=targets
        try: self.step(targets)
        finally: self._mule_position_target=None

    def step(self,actions):
        self.periodic_mass_update(self._control_dt)  # before the next physics interval
        IsaacLabSimulator_PACT.step(self,actions)
        self.payload_reaction_world.zero_()
        for ball,sensor,contact in zip(self.balls,self.ball_sensors,self._ball_contacts):
            ball.update(self._control_dt); sensor.update(self._control_dt)
            contact.update()
            self.payload_reaction_world.sub_(contact.total.reshape(self._num_envs,3))

    def leg_jacobians_b(self,ids=None):
        rows=slice(None) if ids is None else ids
        return canonical_leg_jacobians(self._robot.root_physx_view.get_jacobians()[rows],self._mule_feet,
            self._dof_indices,self._robot.data.root_link_quat_w[rows],fixed_base=self._robot.is_fixed_base)

    def payload_motion(self):
        if not self.balls:
            z=torch.zeros_like(self.ball_offsets); return z,z.clone()
        d=self._robot.data
        state=torch.stack([ball.data.root_state_w for ball in self.balls],1)
        return ball_motion(d.root_link_quat_w,d.root_link_pos_w,d.root_link_lin_vel_w,d.root_link_ang_vel_w,state)

    def _check_base_pos_out_of_bound(self):
        from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
        before=self._robot.data.root_link_pos_w.clone()
        IsaacLabSimulator._check_base_pos_out_of_bound(self)
        delta=self._robot.data.root_link_pos_w-before; ids=delta.ne(0).any(-1).nonzero().flatten()
        if len(ids):
            for ball in self.balls:
                pose=ball.data.root_link_state_w[ids,:7].clone(); pose[:,:3]+=delta[ids]
                ball.write_root_link_pose_to_sim(pose,ids)

    def sampling_state_dict(self):
        view=self._robot.root_physx_view
        keys=('ball_mass','ball_friction','ball_restitution','ball_offsets','ball_relative_velocity','ball_relative_omega',
              'mass_elapsed','mass_change_count','payload_active','realized_severity','_kp_scale','_kd_scale','_motor_strength','_added_base_mass',
              '_base_com_bias','_joint_armature','_joint_friction','_joint_stiffness','_joint_damping','_friction_values')
        return deepcopy(dict(version=2,radius_bank=self.radius_bank,radius_ids=self.radius_ids,
            severity=self.payload_progress,rng=self.payload_rng.get_state(),
            tensors={k:getattr(self,k).clone() for k in keys},
            robot_cadence=self._reset_cadence.__dict__ if hasattr(self,'_reset_cadence') else None,
            joint_properties={n:getattr(self._robot.data,'joint_'+n).clone() for n in ('armature','friction_coeff','stiffness','damping')},
            robot_properties={k:getattr(view,'get_'+k)().clone() for k in ('masses','coms','inertias','material_properties')}))

    def validate_sampling_state_dict(self,state):
        if state['version']!=2 or state['radius_bank']!=self.radius_bank or state['radius_ids']!=self.radius_ids:
            raise ValueError('Incompatible creation-time sphere geometry')
        self.payload_cfg.mass_range(state['severity'])
        torch.Generator(device=self._device).set_state(state['rng'].cpu())
        current=self.sampling_state_dict()
        for group in ('tensors','joint_properties','robot_properties'):
            if state[group].keys()!=current[group].keys(): raise ValueError('Sampling fields differ')
            for k,v in state[group].items():
                if v.shape!=current[group][k].shape or v.dtype!=current[group][k].dtype or not torch.isfinite(v).all():
                    raise ValueError('Invalid sampled property: '+k)
        if self.balls and (state['tensors']['ball_mass']<self.payload_cfg.mass_floor_kg).any(): raise ValueError('Nonpositive sampled mass')
        if ((state['tensors']['realized_severity']<0)|(state['tensors']['realized_severity']>1)).any():
            raise ValueError('Invalid realized payload severity')

    def load_sampling_state_dict(self,state):
        self.validate_sampling_state_dict(state)
        self.payload_rng.set_state(state['rng'].cpu()); self.payload_progress=state['severity']
        for k,v in state['tensors'].items(): getattr(self,k).copy_(v.to(self._device))
        view=self._robot.root_physx_view; ids=torch.arange(self._num_envs)
        for k,v in state['robot_properties'].items(): getattr(view,'set_'+k)(v.cpu(),ids)
        for name,value in state['joint_properties'].items():
            method='write_joint_'+('friction_coefficient' if name=='friction_coeff' else name)+'_to_sim'
            getattr(self._robot,method)(value.to(self._device))
        if state['robot_cadence'] is not None:
            self._reset_sampling_cadence().__dict__.update(deepcopy(state['robot_cadence']))
        self._resume_samples=True
