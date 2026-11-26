"""Normal PILOT interaction curriculum; no actor inputs or additional forces."""
import torch
from .pilot_continuation import walking_mask


class InteractionCfg:
    enabled = True  # PACT family; FACET explicitly opts out, UniFP does not inherit.
    version = 1
    continuous_clock = True
    reference_air_time = True
    max_offset_norm = .30
    ee_gain = [.005, .005, .005]
    planar_base_gain = [.060, .040]
    mixture_weights = [.25, .25, .15, .15, .20]
    mixture_start_iteration = 0
    mixture_ramp_iterations = 5000
    smoke_full_mixture = False
    coefficient_range = [.8, 1.]
    force_range = [40., 50.]
    plateau_range = [1.5, 2.]
    onset_s = 8.
    ramp_s = .5
    recovery_s = 3.
    base_coefficient = .5
    diagnostics = False
    sample_scenarios = True  # Evaluation owns its scenarios, but keeps clock/cap semantics.
    source_hashes = None  # Populated once in the resolved/checkpoint contract.


FAMILIES = ('yielding', 'protected', 'ample', 'restricted', 'native')


def prepare_restore(cfg, saved_environment):
    """Old archived config classes lack this section; restore it as a section."""
    saved=saved_environment.get('pilot_interactions')
    if saved is not None:
        if saved.get('version') != InteractionCfg.version:
            raise ValueError('Unsupported PILOT interaction configuration version')
        cfg.pilot_interactions=InteractionCfg()


def configure(cfg):
    if not getattr(getattr(cfg,'pilot_interactions',None),'enabled',False):
        return
    if not cfg.compliance.enabled:
        raise ValueError('PILOT interactions require differential compliance')
    c=cfg.pilot_interactions
    cfg.compliance.max_offset_norm = c.max_offset_norm
    cfg.compliance.ee_gain = list(c.ee_gain)
    cfg.compliance.base_gain = [*c.planar_base_gain,cfg.compliance.base_gain[2]]
    if c.source_hashes is None:
        from pathlib import Path
        import hashlib
        root=Path(__file__).resolve().parents[2]
        paths=('envs/b1z1/pilot_interactions.py','envs/b1z1/pilot_continuation.py',
            'envs/b1z1/b1z1_pact/b1z1_pact.py','envs/b1z1/compliance_task.py',
            'envs/b1z1/directional_compliance.py','envs/b1z1/compliance_bounds.py',
            'simulator/isaaclab_simulator_b1z1.py')
        c.source_hashes={'legged_gym/'+p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def mixture(cfg, iteration, device):
    w=torch.as_tensor(cfg.mixture_weights,device=device,dtype=torch.float32)
    if w.shape!=(5,) or not torch.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Five nonnegative finite interaction weights required')
    w=w/w.sum()
    fraction=(1. if cfg.smoke_full_mixture else min(1.,max(0.,
        (iteration-cfg.mixture_start_iteration)/max(1,cfg.mixture_ramp_iterations))))
    w=w*fraction;w[-1]+=1-fraction
    return w


def envelope(age, plateau, cfg):
    # One episode-local pulse. No reload before recovery or command resampling.
    t=age-cfg.onset_s
    return torch.minimum((t/cfg.ramp_s).clamp(0,1),
        ((cfg.ramp_s+plateau+cfg.ramp_s-t)/cfg.ramp_s).clamp(0,1))


def descriptors(cfg, n, device, iteration):
    """Adjacent reset rows share a descriptor, not privileged policy inputs.

    Default equal yielding/protected weights produce exact counterpart pairs.
    Physical reset/randomization distributions are unchanged, not forced equal.
    """
    count=(n+1)//2;index=torch.arange(n,device=device)//2
    w=mixture(cfg,iteration,device)
    group=torch.multinomial(torch.stack((w[0]+w[1],w[2],w[3],w[4])),count,True)[index]
    parity=(torch.arange(n,device=device)%2)^torch.randint(2,(count,),device=device)[index]
    if cfg.mixture_weights[0]==cfg.mixture_weights[1]:
        paired_family=parity
    else:
        paired_family=(torch.rand(n,device=device)>=w[0]/(w[0]+w[1]).clamp_min(1e-12)).long()
    family=torch.where(group==0,paired_family,group+1)
    def draw(bounds):return torch.empty(count,device=device).uniform_(*bounds)[index]
    return dict(family=family,axis=torch.randint(3,(count,),device=device)[index],
        sign=torch.where(torch.rand(count,device=device)<.5,-1.,1.)[index],
        force=draw(cfg.force_range),plateau=draw(cfg.plateau_range),
        coefficient=draw(cfg.coefficient_range),base_off=parity.bool(),pair_index=index)


class Interactions:
    def __init__(self, env):
        self.env=env;self.cfg=env.cfg.pilot_interactions
        c=self.cfg
        if c.recovery_s<3 or c.ramp_s<=0 or c.onset_s<=0:
            raise ValueError('Require >=3 seconds recovery and positive pulse timing')
        for lo,hi in (c.coefficient_range,c.force_range,c.plateau_range):
            if not 0<=lo<=hi:raise ValueError('Invalid interaction range')
        if c.coefficient_range[1]>1:raise ValueError('Compliance exceeds one')
        n=env.num_envs;dev=env.device
        self.family=torch.full((n,),4,device=dev,dtype=torch.long)
        self.axis=torch.zeros(n,device=dev,dtype=torch.long)
        self.sign=torch.ones(n,device=dev)
        self.force=torch.zeros(n,device=dev)
        self.plateau=torch.zeros(n,device=dev)
        self.setup_rejected=torch.zeros(n,device=dev,dtype=torch.bool)
        self.descriptor_id=torch.full((n,),-1,device=dev,dtype=torch.long)
        self.next_descriptor=0
        self.counts=torch.zeros(5,device=dev)
        # Compact cumulative sufficient statistics; family IDs are explicit.
        self.metrics=('samples','active','offset_ge15','offset_ge20','offset_ge25',
            'cap_violation','cap_active','recovery_failed','base_clipped',
            'ample_gate_nonzero','ample_base_nonzero','restricted_gate_active',
            'release_samples','release_offset_sum','release_base_sum','failed')
        self.stats=torch.zeros(5,len(self.metrics),device=dev)
        self.max_offset=torch.zeros((),device=dev)
        if c.diagnostics:env.compliance.reference.collect_diagnostics=True

    @property
    def controlled(self):return self.family<4

    def keep_commands(self, ids):
        e=self.env;selected=ids[self.controlled[ids]]
        e.commands[selected,:3]=0
        if e.cfg.commands.heading_command:
            e.heading_commands[selected]=e.simulator.base_euler[selected,2]

    def reset(self, ids):
        e=self.env;c=self.cfg;n=len(ids)
        if not c.sample_scenarios:
            self.family[ids]=4
            return
        # Descriptor sampling is independent of yielding/protected membership:
        # force sign cannot predict compliance. Both draw identical goal/base
        # and physical-initialization distributions; paired tests reuse descriptors.
        data=descriptors(c,n,e.device,getattr(e,'training_iteration',0))
        for name in ('axis','sign','force','plateau','family'):
            getattr(self,name)[ids]=data[name]
        self.descriptor_id[ids]=data['pair_index']+self.next_descriptor
        self.next_descriptor+=(n+1)//2
        self.counts.scatter_add_(0,self.family[ids],torch.ones(n,device=e.device))
        chosen=ids[self.controlled[ids]];m=len(chosen)
        self.setup_rejected[ids]=False
        if not m:return
        axis=self.axis[chosen];sign=self.sign[chosen];family=self.family[chosen]
        p=e.compliance.profile
        p[chosen]=0
        coefficient=data['coefficient'][self.controlled[ids]]
        p[chosen,2*axis+(sign>0).long()]=torch.where(family==1,0.,coefficient)
        p[chosen,6:10]=c.base_coefficient
        # Same distribution for paired yielding/protected; independent base switch
        # within restricted scenarios, not zeroing direct-torso compliance globally.
        off=(family==3)&data['base_off'][self.controlled[ids]]
        p[chosen[off],6:]=0
        goal=e.ee_goal_sphere[chosen].clone()
        goal[:]=goal.new_tensor([.55,0.,0.])
        goal[(axis==0)&(sign<0),0]=.75
        # Outward radial demand near the radius bound (both enabled/disabled base).
        restricted=family==3
        self.axis[chosen[restricted]]=0;self.sign[chosen[restricted]]=1
        p[chosen[restricted],:6]=0;p[chosen[restricted],1]=coefficient[restricted]
        goal[restricted]=goal.new_tensor([.88,0.,0.])
        old=e.ee_goal_sphere[chosen].clone();e.ee_goal_sphere[chosen]=goal
        invalid=e.collision_check(chosen)|~((goal>=e.compliance.bounds[:,0])&(goal<=e.compliance.bounds[:,1])).all(-1)
        self.setup_rejected[chosen]=invalid
        e.ee_goal_sphere[chosen]=torch.where(invalid[:,None],old,goal)
        # Keep original interpolation, but never change goals during pulse/recovery.
        end=c.onset_s+2*c.ramp_s+self.plateau[chosen]+c.recovery_s
        e.traj_total_timesteps[chosen]=torch.maximum(e.traj_total_timesteps[chosen],end/e.dt)
        self.keep_commands(chosen)
        if e.compliance.coverage.weights is not None:e.compliance.coverage.ids[chosen]=2

    def apply(self, output, ids, ee):
        e=self.env;selected=ids[self.controlled[ids]]
        if not ee:
            # Controlled experiments exclude torso streams; native episodes retain
            # all combined loading. The one parent application call remains intact.
            output[selected]=0
            return
        if not e.cfg.commands.apply_ee_external_forces:return
        from .directional_compliance import rotate_yaw,heading
        amplitude=self.force*self.sign*envelope(e.episode_length_buf*e.dt,self.plateau,self.cfg)*e.external_force_scale
        f=torch.zeros_like(output);f.scatter_(1,self.axis[:,None],amplitude[:,None])
        output[selected]=rotate_yaw(f,heading(e.simulator.base_quat))[selected]

    def record(self):
        e=self.env;r=e.compliance.reference;t=e.compliance.target
        if t is None:return
        offset=r.delta.norm(dim=-1);active=e.ee_force_ext_world.norm(dim=-1)>.5
        self.max_offset.copy_(torch.maximum(self.max_offset,offset.max()))
        released=(e.episode_length_buf*e.dt>=self.cfg.onset_s+2*self.cfg.ramp_s+self.plateau)&self.controlled
        gate=t.gates.amax(-1)
        values=torch.stack((torch.ones_like(offset),active,active&(offset>=.15),active&(offset>=.20),
            active&(offset>=.25),offset>e.cfg.compliance.max_offset_norm+2e-6,r.cap_active,e.compliance.recovery_failed,
            (t.base.abs()>=e.compliance.command_limits[:,1].abs()-1e-5).any(-1),
            active&(gate>.05),active&(t.base.abs().amax(-1)>.02),active&(gate>.5),
            released,released*offset,released*t.base.norm(dim=-1),e.fail_buf.bool()),-1).float()
        self.stats.index_add_(0,self.family,values)

    def summary(self):
        return {'families':FAMILIES,'episode_counts':self.counts.tolist(),
                'metrics':self.metrics,'sums':self.stats.tolist(),'max_offset_m':self.max_offset.item()}
