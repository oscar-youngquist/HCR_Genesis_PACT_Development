"""BR adapted with shared auxiliary rewards (not an exact author reproduction).

All coefficients below are reward-rate coefficients; multiply by control dt ONCE,
including event and potential terms, matching the shared task convention. No
positive-total clipping. Snapshot state is clean, endpoint/pre-reset simulator
truth, after instantaneous pushes. Histories are independent of controller delay.
"""
from copy import deepcopy
from types import SimpleNamespace
import math
import torch

VERSION = 3  # Fixed BR tracking and joint-limit counts; incompatible with v2.
PAPER = 'https://arxiv.org/html/2507.07825v1#S3.T1'
# BR fixed tracking/count formulas; audited shared conventions for other overlaps.
# coefficient, formula, required state, units (before coefficient/dt)
ORIGINAL = {
    'tracking_lin_vel': (2., 'exp(-4*sum((command_xy-body_velocity_xy)^2))', 'command, body velocity', '1'),
    'tracking_ang_vel': (.5, 'exp(-4*(command_yaw-body_yaw_rate)^2)', 'command, body angular velocity', '1'),
    'lin_vel_z': (-2., 'v_B_z^2', 'body velocity', 'm^2/s^2'),
    'ang_vel_xy': (-.05, 'sum(omega_B_xy^2)', 'body angular velocity', 'rad^2/s^2'),
    'dof_acc': (-2.5e-7, 'sum(((dq-dq_previous)/control_dt)^2)', 'current/previous joint velocity', 'rad^2/s^4'),
    'joint_power': (-2e-5, 'sum(abs(tau_executed*dq))', 'executed torque, joint velocity', 'W'),
    'torques': (-1e-5, 'sum(tau_executed^2)', 'executed torque', 'Nm^2'),
    'base_height': (-1.2, '(mean(base_z-terrain_samples)-height_target)^2', 'terrain-relative height', 'm^2'),
    'action_rate': (-.02, 'sum((a-a_previous)^2)', 'clipped issued position actions before delay', 'normalized action^2'),
    'action_smoothness': (-.001, 'sum((a-2*a_previous+a_previous2)^2)', 'two issued action histories', 'normalized action^2'),
    'collision': (-1., 'count(norm(ground_force)>0.1 N)', 'penalized non-foot links including tray', 'count'),
    'dof_pos_limits': (-2., 'sum((q<soft_lower) OR (q>soft_upper))', 'canonical joint positions and simulator soft limits', 'joint count'),
    'feet_air_time': (1., 'sum((air_time-0.5)*first_filtered_contact)*I(norm(command_xy)>0.1)', 'ground contacts and air-time history', 's'),
    'feet_contact_forces': (-2., 'sum(relu(norm(ground_foot_force)-max_contact_force))', 'ground-only foot forces', 'N'),
    'load_motion': (2., '1/(1+norm(load_relative_velocity_B))', 'ground-truth derivative of body-relative load CoM', '1'),
}
# Explicit allowlist; zero coefficients remain disabled. Helpers are audited below.
ADDITIONS = {
    'dof_vel_limits': ('sum(clamp(abs(dq)-soft_fraction*asset_speed_limit,0,1))','dq, named speed limits','rad/s'),
    'torque_limits': ('sum(relu(abs(requested_tau)-soft_fraction*torque_limit))','unclipped delayed requested torque','Nm'),
    'dof_close_to_default': ('sum((q-q_default)^2)','q, default pose','rad^2'),
    'dof_vel_stand_still': ('sum(abs(dq))*I(norm(command)<0.1)','dq, command','rad/s'),
    'dof_pos_stand_still': ('sum((q-q_default)^2)*I(norm(command)<0.1)','q, command','rad^2'),
    'stand_still_contact': ('I(all(foot_Fz>0.1))*I(norm(command)<0.1)','ground foot forces, command','1'),
    'alive_bonus': ('I(not done)','termination masks','1'),
    'orientation': ('sum(projected_gravity_xy^2)','projected gravity','1'),
    'dof_tracking': ('exp(-4*sum((q_delayed_target-q)^2))','executed delayed position target, q','1'),
    'joint_power_dist': ('var(tau_executed*dq, correction=1)','executed torque, dq','W^2'),
    'hip_pos': ('sum((q_hip-q_default_hip)^2)','canonical hip positions','rad^2'),
    'pbrs_orientation': ('phi_next-phi_previous; phi=exp(-sum(gravity_xy^2)/0.5)','gravity, previous potential, failure mask','1'),
    'pbrs_height': ('phi_next-phi_previous; phi=exp(-(height-target)^2/0.5)','terrain-relative height, previous potential, failure mask','1'),
    'support_polygon': ('shared stance center/radius (>=3 feet), segment distance (2); exp(-excess^2/(sigma+1e-6))','base/feet world positions, ground contacts','1'),
    'vhip_angle': ('HardPACT relu(acos(clamp(base_world_z/norm(base-cop),-1,1))-0.1)','ground-force-weighted CoP, base position','rad'),
    'vhip_angular_acc': ('HardPACT relu(abs(-9.81*sin(theta)/norm(base-cop))-0.001)','ground-force-weighted CoP, base position','rad/s^2'),
    'front_foot_overreach': ('sum(I(Fz>5)*relu(abs(foot_B_x-front_nominal)-margin)^2)','front feet body positions, ground contacts','m^2'),
    'rear_foot_overreach': ('sum(I(Fz>5)*relu(abs(foot_B_x-rear_nominal)-margin)^2)','rear feet body positions, ground contacts','m^2'),
    'foot_clearance_terrain_aware': ('exp(-sum(swing*(height_error^2+excess_weight*relu(excess)^2))/sigma)','foot height, local maximum terrain height, ground contacts','1'),
    'foot_slip': ('sum((I(Fz>1)*sum(foot_world_velocity_xy))^2)','ground contacts, world foot velocity','m^2/s^2'),
    'stumble': ('sum(I(abs(Fz)<=1)*I(norm(Fxy)>4*abs(Fz))*I(norm(Fxy)>5))','ground foot forces','count'),
    'feet_near_edge': ('count(near_edge and ground_contact)','terrain edge mask, ground contacts','count'),
    'edge_swing_clearance': ('sum(I(near_edge and swing)*relu(max_height_ahead+margin-foot_z)^2)','terrain edge/height, foot position/contact','m^2'),
    'swing_foot_collision_edge': ('sum(edge_swing_face_mask*relu(velocity_xy dot normal_xy-min_speed)^2)','terrain normals, feet velocity/contact','m^2/s^2'),
    'feet_regulation': ('sum(foot_speed_xy^2*exp(-foot_height/(0.025*height_target)))','base/feet height, feet velocity, gravity','m^2/s^2'),
    'termination': ('I(done and not timeout)','failure mask','1'),
    'dof_act_limits': ('sum(position_target_excess_outside_hard_limits)','delayed position action, hard joint limits','rad'),
}
PARAMETERS = ('base_height_target','max_contact_force','contact_force_threshold',
    'foot_clearance_target','foot_height_offset','foot_clearance_excess_margin',
    'foot_clearance_excess_weight','foot_clearance_tracking_sigma','front_foot_x_nominal',
    'foot_x_margin','rear_foot_x_nominal','rear_foot_x_margin','support_polygon_sigma',
    'edge_swing_clearance_margin','swing_collision_max_normal_z','swing_collision_min_speed',
    )


def resolve_reward_spec(shared, dt, overrides=None):
    """Filter source curriculum explicitly; reject protected terms in BR overrides."""
    overrides=deepcopy(overrides or {})
    if set(overrides)-{'mode','additional_coefficients','curriculum'}:
        raise ValueError('Unknown BR reward override')
    mode=overrides.get('mode','merged')
    if mode not in ('merged','smoke_tracking'): raise ValueError('Unknown BR reward mode')
    coefficients={k:float(getattr(shared.scales,k,0.)) for k in ADDITIONS}
    changes=overrides.get('additional_coefficients',{})
    if set(changes)-ADDITIONS.keys(): raise ValueError('Only allowlisted BR additions may override coefficients')
    coefficients.update(changes)
    c=shared.reward_curriculum
    inherited_keys=list(c.curr_reward_keys)
    schedule=dict(enabled=bool(shared.use_reward_curriculum),warmup=int(c.warmup_steps),
        duration=int(c.curr_steps),bounds={k:list(c.curr_reward_bounds[k]) for k in inherited_keys
            if k in ADDITIONS and coefficients[k]!=0})
    schedule.update(overrides.get('curriculum',{}))
    if set(schedule)-{'enabled','warmup','duration','bounds'} or set(schedule['bounds'])-ADDITIONS.keys():
        raise ValueError('Reward curriculum must target compatible additions, never protected BR terms')
    if schedule['warmup']<0 or schedule['duration']<0: raise ValueError('Negative reward schedule duration')
    if any(len(v)!=2 or not all(math.isfinite(x) for x in v) for v in schedule['bounds'].values()):
        raise ValueError('Invalid reward curriculum bounds')
    if not all(math.isfinite(v) for v in coefficients.values()): raise ValueError('Nonfinite reward coefficient')
    terms={}
    for name,(weight,formula,state,units) in ORIGINAL.items():
        terms[name]=dict(group='original_br',coefficient=weight,formula=formula,state=state,units=units,
            curriculum_eligible=False,coefficient_source=PAPER,
            source=PAPER if name in ('load_motion','tracking_lin_vel','tracking_ang_vel','dof_pos_limits') else 'Go2HardPACT/Go2PACT shared reward helper; BR coefficient',
            time_scaling='coefficient * control_dt once')
    for name,(formula,state,units) in ADDITIONS.items():
        terms[name]=dict(group='shared_addition',coefficient=coefficients[name],formula=formula,state=state,units=units,
            curriculum_eligible=True,source='Go2PACT/Go2HardPACT audited helper; BR adaptation',
            time_scaling='coefficient * control_dt once')
    excluded={k:('BR owns this quantity' if k in ORIGINAL else 'channel-specific, duplicate, or not allowlisted')
        for k in dir(shared.scales) if not k.startswith('_') and k not in ADDITIONS}
    return dict(version=VERSION,mode=mode,control_dt=dt,terms=terms,curriculum=schedule,
        excluded_shared=excluded,protected_source_curriculum_ignored=[k for k in inherited_keys if k in ORIGINAL],
        assumptions={
            'load_motion':'Author paper/project page do not specify scalar reduction. Use 1/(1+L2 norm(dr_B/dt)) in m/s; not verified author implementation.',
            'air_time':'Calls HardPACT inherited feet_air_time helper: Fz>1 N, current OR previous contact; landing event sum(air_time-0.5 s), planar command norm>0.1 m/s. Reset contacts initialize filter.',
            'contact_force':'Calls HardPACT inherited feet_contact_forces helper: sum force-norm excess above inherited max_contact_force (N), not squared.',
            'joint_limit':'Count joints strictly outside simulator dof_pos_limits: URDF midpoint +/- half-range * inherited soft_dof_pos_limit; equality is not a violation. Threshold choice is an implementation assumption.',
            'tracking':'Fixed exp(-4*error^2), independent of terrain, commands and wrench sigma. Raw curriculum evidence uses the same unweighted formula.',
            'timing':'Endpoint after push, before reset; executed last-substep torque and endpoint dq for physical power. Joint acceleration is control-dt finite difference. Root impulse also propagates to cached foot velocities at fixed q/dq. Contacts remain the last completed physics measurement. No additional physics.',
            'action_history':'Clipped issued normalized position actions before delay. First rate and first two smoothness differences omitted after reset; dq history initialized from reset state.',
            'smoothness_sign':'Uses HardPACT a_t-2*a_(t-1)+a_(t-2), intentionally replacing the printed Table-I minus. Constant actions have zero steady-state smoothness penalty.',
            'shaping':'Shared undiscounted potential difference; absorbing potential zero on physical terminal, actual terminal potential on timeout. All terms dt-scaled, no total clamp.',
            'slip':'Uses existing HardPACT square(sum(vxy)) literally; opposite x/y components can cancel. No BR-only formula correction.',
            'vhip':'Uses existing HardPACT world-z numerator and CoP formula literally, with ground-filtered contact tensors and correct sensor indices. No BR-only height/support formula correction.',
            'method':'BR fixed tracking/count terms and coefficients with audited shared conventions elsewhere, plus shared auxiliary rewards. Not exact author-code reproduction.'})


def tracking(velocity, omega, commands):
    return torch.exp(-4*(commands[:,:2]-velocity[:,:2]).square().sum(-1)), torch.exp(-4*(commands[:,2]-omega[:,2]).square())


class BRRewards:
    def __init__(self, spec, cfg, num_envs, device):
        self.spec=deepcopy(spec); self.cfg=cfg; self.dt=spec['control_dt']; self.iteration=0
        self.actions=torch.zeros(num_envs,2,12,device=device)
        self.age=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.velocity=torch.zeros(num_envs,12,device=device)
        self.air=torch.zeros(num_envs,4,device=device)
        self.contacts=torch.zeros(num_envs,4,dtype=torch.bool,device=device)
        self.potential=torch.zeros(num_envs,2,device=device)
        self.last_raw={}; self.last_weighted={}

    def potentials(self,s):
        height=(s.base_pos[:,2,None]-s.measured_heights).mean(-1)
        return torch.stack((torch.exp(-s.projected_gravity[:,:2].square().sum(-1)/.5),
            torch.exp(-(height-self.cfg.rewards.base_height_target).square()/.5)),-1)

    def reset(self, ids, s):
        self.actions[ids]=0; self.age[ids]=0; self.air[ids]=0
        self.velocity[ids]=s.dof_vel[ids]
        self.contacts[ids]=s.link_contact_forces[ids][:,s.feet_contact_indices,2]>1.
        self.potential[ids]=self.potentials(s)[ids]

    def begin_iteration(self,iteration):
        self.iteration=int(iteration)

    def coefficients(self):
        values={k:v['coefficient'] for k,v in self.spec['terms'].items()}
        c=self.spec['curriculum']
        if c['enabled']:
            p=1. if c['duration']==0 and self.iteration>=c['warmup'] else min(1.,max(0.,(self.iteration-c['warmup'])/max(1,c['duration'])))
            ramp=.5*(1-math.cos(math.pi*p))
            for k,(lo,hi) in c['bounds'].items():
                if values[k]!=0: values[k]=lo+(hi-lo)*ramp
        return values

    @torch.no_grad()
    def evaluate(self,s,action,delayed,commands,load_velocity,done,timeout,*,tracking_context=None):
        r=self.cfg.rewards; foot=s.link_contact_forces[:,s.feet_contact_indices]
        contact=foot[:,:,2]>1.
        raw=dict(load_motion=1/(1+load_velocity.norm(dim=-1)))
        coefficients=self.coefficients()
        # Call audited stateless shared additions with explicit dependencies;
        # never point contact access at aggregate tray/payload support forces.
        from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
        from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
        # All state and histories refer to the current owned snapshot.
        if not hasattr(self,'_shared_context'): self._shared_context=SimpleNamespace()
        context=self._shared_context
        context.__dict__.update(simulator=s,cfg=self.cfg,commands=commands.detach().clone(),actions=action,
            reset_buf=done,time_out_buf=timeout,device=str(action.device),dt=self.dt,
            non_failure_reset_buf=torch.zeros_like(done),
            grf_processor=SimpleNamespace(contacts=foot[:,:,2]>r.contact_force_threshold))
        context.last_actions=self.actions[:,0]; context.llast_actions=self.actions[:,1]
        context.feet_air_time=self.air.clone(); context.last_contacts=self.contacts.clone()
        s.last_dof_vel=self.velocity; s.penalized_contact_indices=s.penalised_contact_indices
        context._legacy_task_class=Go2PACT
        raw['tracking_lin_vel'],raw['tracking_ang_vel']=tracking(s.base_lin_vel,s.base_ang_vel,commands)
        raw['dof_pos_limits']=((s.dof_pos<s.dof_pos_limits[:,0]) |
                               (s.dof_pos>s.dof_pos_limits[:,1])).float().sum(-1)
        for key in ORIGINAL:
            if key in raw: continue
            raw[key]=getattr(Go2PACT,'_reward_'+key)(context).float()
        # Keep BR's reset-safe history availability masks; no cross-episode terms.
        raw['action_rate']*=self.age>0
        raw['action_smoothness']*=self.age>1
        context.actions=delayed  # Position tracking uses the actually issued delayed target.
        context._feet_contact_mask=lambda: foot[:,:,2]>r.contact_force_threshold
        context._feet_near_edge_mask=lambda: s.near_edge
        s._link_contact_forces=s.link_contact_forces
        context._compute_vhip_angle=lambda: Go2PACT._compute_vhip_angle(context)
        context._compute_vhip_acceleration=lambda: Go2PACT._compute_vhip_acceleration(context)
        special={'pbrs_orientation','pbrs_height','torque_limits'}
        for key in ADDITIONS:
            if coefficients[key]==0 or key in special: continue
            cls=Go2HardPACT if key in ('dof_vel_limits','foot_clearance_terrain_aware','front_foot_overreach','rear_foot_overreach') else Go2PACT
            raw[key]=getattr(cls,'_reward_'+key)(context).float()
        potential=self.potentials(s)
        for i,key in enumerate(('pbrs_orientation','pbrs_height')):
            if coefficients[key]!=0: raw[key]=torch.where(done&~timeout,0.,potential[:,i])-self.potential[:,i]
        if coefficients['torque_limits']!=0:
            raw['torque_limits']=(s.requested_torque.abs()-s.torque_limits*r.soft_torque_limit).clamp_min(0).sum(-1)
        weighted={k:v*coefficients[k]*self.dt for k,v in raw.items()}
        # Keep current-step views for focused formula tests and episode logging.
        # No raw/weighted/coefficient TensorBoard series or old per-step reductions.
        self.last_raw={k:v.detach() for k,v in raw.items()}
        self.last_weighted={k:v.detach() for k,v in weighted.items() if coefficients[k]!=0}
        original=sum(weighted[k] for k in ORIGINAL)
        additional=sum((v for k,v in weighted.items() if k in ADDITIONS),torch.zeros_like(original))
        total=original+additional
        self.actions[:,1].copy_(self.actions[:,0]); self.actions[:,0].copy_(action); self.age+=1
        self.velocity.copy_(s.dof_vel); self.air.copy_(context.feet_air_time); self.contacts.copy_(context.last_contacts)
        self.potential.copy_(potential)
        return total

    def state_dict(self):
        return dict(version=VERSION,spec=deepcopy(self.spec),iteration=self.iteration)

    def load_state_dict(self,state):
        if state['version']!=VERSION or state['spec']!=self.spec: raise ValueError('Incompatible BR reward contract')
        self.begin_iteration(state['iteration'])  # Episode-local buffers rebuilt by fresh reset.
