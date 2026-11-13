"""MULE Table I coefficients with explicitly selected HardPACT definitions.

No author training code is available. Table I specifies coefficients, not the
non-GRF formulas: those use the existing HardPACT/Go2PACT helpers at the user's
request, except clearance: its negative coefficient requires a positive cost.
Terrain-relative speed-weighted squared clearance error is an explicit
reproduction assumption, not a verified author-code formula.
"""
from copy import deepcopy
from types import SimpleNamespace as NS
import math
import torch
from rsl_rl.beyond_robustness_rewards import ADDITIONS as SHARED_ADDITIONS, PARAMETERS

VERSION = 2
PAPER = 'https://arxiv.org/html/2505.00488v1#S2.T1'
# Explicit zeros are part of the contract, not absent/default inherited terms.
NATIVE = {
    'tracking_lin_vel': (1., 0.), 'tracking_ang_vel': (.5, 0.),
    'lin_vel_z': (-2., -2.), 'ang_vel_xy': (-.05, -.05),
    'orientation': (-.2, -.2), 'dof_acc': (-2.5e-7, -2.5e-7),
    'joint_power': (-2e-5, 0.), 'base_height': (-2., -2.),
    'foot_clearance': (-.01, -.01), 'action_rate': (-.001, -.01),
    'action_smoothness': (-.01, -.01), 'grf_tracking': (0., 2.),
}
FORMULAS = {
    'tracking_lin_vel': ('exp(-4*sum((cmd_xy-v_B_xy)^2))','1'),
    'tracking_ang_vel': ('exp(-4*(cmd_yaw-omega_B_z)^2)','1'),
    'lin_vel_z': ('v_B_z^2','m^2/s^2'), 'ang_vel_xy': ('sum(omega_B_xy^2)','rad^2/s^2'),
    'orientation': ('sum(projected_gravity_xy^2)','1'),
    'dof_acc': ('sum(((dq-dq_previous)/control_dt)^2)','rad^2/s^4'),
    'joint_power': ('sum(abs(tau_executed*dq))','W'),
    'base_height': ('(mean(base_z-terrain_samples)-height_target)^2','m^2'),
    'foot_clearance': ('sum(norm(foot_v_world_xy)*(foot_z-mean(local_terrain_z)-height_offset-clearance_target)^2)','m^3/s'),
    'action_rate': ('sum((a_stream-a_previous_stream)^2)','normalized action^2'),
    'action_smoothness': ('sum((a_stream-2*a_previous_stream+a_previous2_stream)^2)','normalized action^2'),
    'grf_tracking': ('.75*I(h>h_target)+.5*I(h<h_target)*I(sum(norm(force_proxy_i))>total_realized_mass*9.81)','1'),
}
# Additions belong ONLY to nominal: no aliases restore adaptive tracking/power.
EXCLUDED_OVERLAPS = {'orientation','joint_power_dist','foot_clearance_terrain_aware',
                    'feet_regulation','pbrs_orientation','pbrs_height'}
ADDITIONS = {k:v for k,v in SHARED_ADDITIONS.items() if k not in EXCLUDED_OVERLAPS}


def resolve_reward_spec(shared, dt, overrides=None):
    overrides=deepcopy(overrides or {})
    if set(overrides)-{'additional_coefficients','curriculum'}: raise ValueError('Unknown MULE reward setting')
    coefficients={k:float(getattr(shared.scales,k,0.)) for k in ADDITIONS}
    changes=overrides.get('additional_coefficients',{})
    if set(changes)-ADDITIONS.keys(): raise ValueError('Native MULE terms/zeros are protected')
    coefficients.update(changes)
    c=shared.reward_curriculum
    schedule=dict(enabled=bool(shared.use_reward_curriculum),warmup=int(c.warmup_steps),
        duration=int(c.curr_steps),bounds={k:list(c.curr_reward_bounds[k]) for k in c.curr_reward_keys
            if k in ADDITIONS and coefficients[k]!=0})
    schedule.update(overrides.get('curriculum',{}))
    if (set(schedule)-{'enabled','warmup','duration','bounds'} or set(schedule['bounds'])-ADDITIONS.keys()
            or schedule['warmup']<0 or schedule['duration']<0): raise ValueError('Invalid/protected reward curriculum')
    if not all(math.isfinite(v) for v in coefficients.values()) or any(
            len(v)!=2 or not all(math.isfinite(x) for x in v) for v in schedule['bounds'].values()):
        raise ValueError('Invalid reward coefficients/bounds')
    terms={}
    for k,weights in NATIVE.items():
        formula,units=FORMULAS[k]
        terms[k]=dict(coefficients=dict(zip(('nominal','adaptive'),weights)),formula=formula,units=units,
            source=PAPER if k=='grf_tracking' else ('terrain-relative squared-error reproduction assumption' if k=='foot_clearance' else 'Go2PACT helper used by HardPACT; MULE Table I coefficient'),
            coefficient_source=PAPER,curriculum_eligible=False,reason='native MULE; explicit zeros protected')
    for k,(formula,state,units) in ADDITIONS.items():
        terms[k]=dict(coefficients=dict(nominal=coefficients[k],adaptive=0.),formula=formula,state=state,units=units,
            source='audited HardPACT shared addition',curriculum_eligible=True,
            reason='nominal locomotion only; disabled source coefficients remain disabled')
    return dict(version=VERSION,control_dt=dt,terms=terms,curriculum=schedule,
        excluded_shared={k:('native overlap' if k in NATIVE or k in EXCLUDED_OVERLAPS else
            'channel-specific, duplicate or not allowlisted') for k in dir(shared.scales)
            if not k.startswith('_') and k not in ADDITIONS},
        assumptions=dict(formulas='Unspecified native formulas use HardPACT except clearance: positive speed-weighted squared error relative to the mean of nine local terrain samples, with shared foot offset/target. No exponential or sigma. Not verified author code.',
            actions='Separate raw normalized samples before combined clipping/delay/scaling. HardPACT difference formulas; no action-scale factor. First one/two unavailable differences masked.',
            timing='Clean endpoint after pushes, before reset. Last executed substep torque; dq derivative uses control_dt. Multiply each coefficient by control_dt exactly once; no total clipping.',
            grf='Each |f_i| is L2 norm of existing torque/Jacobian proxy (body axes), not simulator GRF. All four estimates must be valid for the low-height force test; invalid estimate does not earn that branch. Strict inequalities; equality gives zero.',
            mass='Realized articulation mass already includes tray and robot randomization; add only active sphere masses once. Parked spheres contribute zero.',
            additions='Nominal-only compatibility merge, not original MULE. No adaptive tracking/power aliases or duplicate orientation/height/clearance shaping.'))


def clearance_cost(foot_position, foot_velocity, terrain_height, target, offset):
    """Positive cost; invariant to a common world-z translation of feet/terrain."""
    error=foot_position[...,2]-terrain_height.mean(-1)-offset-target
    return (foot_velocity[...,:2].norm(dim=-1)*error.square()).sum(-1)


def grf_reward(height, target, force, force_valid, robot_with_tray_mass, active_ball_mass):
    total=robot_with_tray_mass.reshape(-1)+active_ball_mass.sum(-1)
    valid=force_valid.all(-1)&torch.isfinite(force).all(dim=(-1,-2))&torch.isfinite(total)&(total>0)
    magnitude=torch.where(valid[:,None,None],force,0.).norm(dim=-1).sum(-1)
    return .75*(height>target).float()+.5*((height<target)&valid&(magnitude>total*9.81)).float()


class MULERewards:
    def __init__(self,spec,cfg,n,device):
        self.spec=deepcopy(spec); self.cfg=cfg; self.dt=spec['control_dt']; self.iteration=0
        self.actions=torch.zeros(n,2,2,12,device=device) # stream, previous/two-back
        self.age=torch.zeros(n,device=device,dtype=torch.long)
        self.velocity=torch.zeros(n,12,device=device)
        self.counts=torch.zeros(4,device=device) # rows, valid state, valid force proxy, nonfinite native rows
        self.last_raw={}; self.last_weighted={}

    def reset(self,ids,s):
        self.actions[ids]=0; self.age[ids]=0; self.velocity[ids]=s.dof_vel[ids]

    def begin_iteration(self,iteration):
        self.iteration=int(iteration); self.counts.zero_()

    def coefficients(self):
        result={k:dict(v['coefficients']) for k,v in self.spec['terms'].items()}
        c=self.spec['curriculum']
        if c['enabled']:
            p=float(self.iteration>=c['warmup']) if c['duration']==0 else min(1.,max(0.,(self.iteration-c['warmup'])/c['duration']))
            ramp=.5*(1-math.cos(math.pi*p))
            for k,(lo,hi) in c['bounds'].items():
                if result[k]['nominal']!=0: result[k]['nominal']=lo+(hi-lo)*ramp
        return result

    @torch.no_grad()
    def evaluate(self,s,command,delayed,commands,force,robot_mass,ball_mass,done,timeout,valid):
        from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
        from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
        foot=s.link_contact_forces[:,s.feet_contact_indices]
        context=NS(simulator=s,cfg=self.cfg,commands=commands,device=str(s.dof_pos.device),dt=self.dt,
            reset_buf=done,time_out_buf=timeout,non_failure_reset_buf=torch.zeros_like(done),
            grf_processor=NS(contacts=foot[:,:,2]>self.cfg.rewards.contact_force_threshold))
        s.last_dof_vel=self.velocity; s.penalized_contact_indices=s.penalised_contact_indices
        common={k:getattr(Go2PACT,'_reward_'+k)(context) for k in NATIVE
            if k not in ('action_rate','action_smoothness','grf_tracking','foot_clearance')}
        common['foot_clearance']=clearance_cost(s.feet_pos,s.feet_vel,s._height_around_feet,
            self.cfg.rewards.foot_clearance_target,self.cfg.rewards.foot_height_offset)
        height=(s.base_pos[:,2,None]-s.measured_heights).mean(-1)
        common['grf_tracking']=grf_reward(height,self.cfg.rewards.base_height_target,
            force.force_b,force.valid,robot_mass,ball_mass)
        self.counts.add_(torch.stack((valid.new_tensor(len(valid),dtype=torch.float),valid.sum().float(),
            (valid&force.valid.all(-1)&torch.isfinite(force.force_b).all(dim=(-1,-2))).sum().float(),
            (~torch.stack([torch.isfinite(v) for v in common.values()]).all(0)).sum().float())))
        context.actions=(delayed-s.default_dof_pos)/self.cfg.control.action_scale
        context._feet_contact_mask=lambda:foot[:,:,2]>self.cfg.rewards.contact_force_threshold
        context._feet_near_edge_mask=lambda:s.near_edge
        s._link_contact_forces=s.link_contact_forces
        context._compute_vhip_angle=lambda:Go2PACT._compute_vhip_angle(context)
        context._compute_vhip_acceleration=lambda:Go2PACT._compute_vhip_acceleration(context)
        coefficients=self.coefficients(); extra={}
        for k in ADDITIONS:
            if not coefficients[k]['nominal']: continue
            if k=='torque_limits':
                extra[k]=(s.requested_torque.abs()-s.torque_limits*self.cfg.rewards.soft_torque_limit).clamp_min(0).sum(-1)
            else:
                cls=Go2HardPACT if k in ('dof_vel_limits','front_foot_overreach','rear_foot_overreach') else Go2PACT
                extra[k]=getattr(cls,'_reward_'+k)(context)
        self.last_raw={}; self.last_weighted={}; totals=[]
        for i,(stream,action) in enumerate(zip(('nominal','adaptive'),(command.nominal_raw,command.adaptive_raw))):
            context.actions=action; context.last_actions=self.actions[:,i,0]; context.llast_actions=self.actions[:,i,1]
            raw={**common,**(extra if i==0 else {})}
            raw['action_rate']=Go2PACT._reward_action_rate(context)*(self.age>0)
            raw['action_smoothness']=Go2PACT._reward_action_smoothness(context)*(self.age>1)
            # Explicitly mask invalid measurements, never return nonfinite PPO rewards.
            raw={k:torch.where(valid&torch.isfinite(v),v,0.) for k,v in raw.items()}
            weighted={k:v*coefficients[k][stream]*self.dt for k,v in raw.items() if coefficients[k][stream]!=0}
            self.last_raw[stream]=raw; self.last_weighted[stream]=weighted
            totals.append(sum(weighted.values(),torch.zeros_like(height)))
        self.actions[:,:,1].copy_(self.actions[:,:,0])
        self.actions[:,0,0].copy_(command.nominal_raw); self.actions[:,1,0].copy_(command.adaptive_raw)
        self.age+=1; self.velocity.copy_(s.dof_vel)
        return tuple(totals)

    def state_dict(self): return dict(version=VERSION,spec=deepcopy(self.spec),iteration=self.iteration)
    def load_state_dict(self,state):
        if state['version']!=VERSION or state['spec']!=self.spec: raise ValueError('Incompatible MULE reward contract')
        self.begin_iteration(state['iteration']) # Fresh resets rebuild episode-local histories.
