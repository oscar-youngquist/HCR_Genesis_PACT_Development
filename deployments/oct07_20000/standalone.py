"""Portable inference and October-7 QP assembly; no simulator/solver imports."""
from typing import Dict
import copy
import torch
from torch import nn, Tensor


class Deployment(nn.Module):
    def __init__(self, actor, env, cfg):
        super().__init__()
        enc=actor.context_encoder
        self.features=nn.Sequential(copy.deepcopy(enc.ce_in),nn.ELU(),copy.deepcopy(enc.ce_h1),
            nn.ELU(),copy.deepcopy(enc.ce_h2),nn.ELU())
        self.mean=nn.Sequential(copy.deepcopy(enc.ce_latmean_h),nn.ELU(),copy.deepcopy(enc.ce_out_mean))
        self.explicit=copy.deepcopy(actor.explicit_estimator.network)
        self.trunk=copy.deepcopy(actor.act_trunk)
        self.position=copy.deepcopy(actor.act_pos_out)
        self.feedforward=copy.deepcopy(actor.act_tau_out)
        self.grf=copy.deepcopy(actor.physics_estimator.grf_head)
        self.wrench=copy.deepcopy(actor.physics_estimator.wrench_head)
        for name,value in [('grf_scale',actor.physics_estimator.grf_scale_n),
            ('wrench_scale',actor.physics_estimator.wrench_scale),('wrench_clip',actor.physics_estimator.wrench_qp_clip)]:
            self.register_buffer(name,value.detach().clone())
        names=env['asset']['dof_names']
        self.register_buffer('default_q',torch.tensor([env['init_state']['default_joint_angles'][n] for n in names]))
        self.register_buffer('variable_scale',torch.tensor([cfg['torque_scale_nm']]*12+[cfg['force_scale_n']]*12))
        self.register_buffer('eye',torch.eye(24))
        self.register_buffer('selector',torch.cat((torch.zeros(6,12),torch.eye(12)),0))
        fr=torch.zeros(4,5,24)
        for foot in range(4):
            j=12+3*foot
            fr[foot,0,j+2]=-1
            fr[foot,1,j]=1;fr[foot,2,j]=-1
            fr[foot,3,j+1]=1;fr[foot,4,j+1]=-1
            fr[foot,1:,j+2]=-cfg['friction_coefficient']
        self.register_buffer('friction',fr)
        self.register_buffer('diagonal',2*torch.tensor([cfg['torque_tracking_weight']]*12+[cfg['force_tracking_weight']]*12)/self.variable_scale.square())
        self.contact_epsilon=float(actor.explicit_estimator.contact_epsilon)
        self.clip_obs=float(env['normalization']['clip_observations'])
        self.clip_actions=float(env['normalization']['clip_actions'])
        self.position_scale=float(env['control']['action_scale'])
        self.torque_scale=float(env['control']['torque_scale'])
        scales=env['normalization']['obs_scales']
        self.lin_scale=float(scales['lin_vel']);self.ang_scale=float(scales['ang_vel'])
        self.q_scale=float(scales['dof_pos']);self.v_scale=float(scales['dof_vel'])
        self.threshold=float(cfg['contact_threshold'])
        self.wc=float(cfg['contact_acceleration_weight']);self.sc=float(cfg['contact_acceleration_scale_m_s2'])
        self.wb=float(cfg['attitude_weight']);self.sb=float(cfg['attitude_acceleration_scale_rad_s2'])
        self.kp_tilt=float(cfg['attitude_kp']);self.kd_tilt=float(cfg['attitude_kd'])
        self.wh=float(cfg['height_weight']);self.sh=float(cfg['height_acceleration_scale'])
        self.kp_height=float(cfg['height_kp']);self.kd_height=float(cfg['height_kd']);self.height_target=float(cfg['height_target'])
        self.wxy=float(cfg['planar_velocity_weight']);self.sxy=float(cfg['planar_velocity_scale_m_s'])
        self.wyaw=float(cfg['yaw_rate_weight']);self.syaw=float(cfg['yaw_rate_scale_rad_s'])
        self.horizon=float(cfg['qp_velocity_objective_horizon_s'])
        self.beta=float(cfg['position_integration_coefficient']);self.reg=float(cfg['q_regularization'])
        self.slack_scale=float(cfg['soft_joint_recovery_scale_rad_s2']);self.slack_weight=float(cfg['soft_joint_recovery_weight'])
        assert cfg['torque_rate_constraint_weight']==0 and cfg['endpoint_torque_constraints']

    def forward(self, obs:Tensor, history:Tensor):
        torch._assert(obs.dim()==2 and obs.size(1)==57,'obs must be [B,57]')
        torch._assert(history.dim()==2 and history.size(1)==570,'history must be [B,570]')
        feature=self.features(history)
        z=self.mean(feature)
        raw=self.explicit(feature)
        explicit=torch.cat((raw[:,:3],self.contact_epsilon+(1-2*self.contact_epsilon)*torch.sigmoid(raw[:,3:7]),raw[:,7:]),-1)
        hidden=self.trunk(torch.cat((obs,z,explicit),-1))
        return torch.cat((self.position(hidden),self.feedforward(hidden)),-1),z,explicit

    @torch.jit.export
    def observation(self,command:Tensor,gravity_body:Tensor,omega_body:Tensor,q:Tensor,v:Tensor,previous_action:Tensor):
        packed=torch.cat((command[:,:2]*self.lin_scale,command[:,2:3]*self.ang_scale,gravity_body,
            omega_body*self.ang_scale,(q-self.default_q)*self.q_scale,v*self.v_scale,previous_action),-1)
        # Historical env appends history before applying current-observation clipping.
        return packed.clamp(-self.clip_obs,self.clip_obs),packed

    @torch.jit.export
    def append_history(self,history:Tensor,obs:Tensor):
        return torch.cat((history[:,57:],obs),-1)

    @torch.jit.export
    def nominal(self,action:Tensor,q:Tensor,v:Tensor,kp:Tensor,kd:Tensor,motor:Tensor,feedback_weight:Tensor,feedforward_weight:Tensor,limits:Tensor):
        action=action.clamp(-self.clip_actions,self.clip_actions)
        target=self.default_q+self.position_scale*action[:,:12]
        ff=self.torque_scale*action[:,12:]
        pd=motor*feedback_weight*(kp*(target-q)-kd*v)
        requested=pd+motor*feedforward_weight*ff
        return target,ff,pd,requested,torch.maximum(torch.minimum(requested,limits),-limits)

    @torch.jit.export
    def physics(self,z:Tensor,explicit:Tensor,tau_nom:Tensor):
        grf=self.grf(torch.cat((z,explicit,tau_nom),-1))*self.grf_scale
        raw=self.wrench(torch.cat((z,explicit),-1))*self.wrench_scale
        clipped=torch.maximum(torch.minimum(torch.nan_to_num(raw),self.wrench_clip),-self.wrench_clip)
        return grf.reshape(-1,4,3),raw,clipped

    @torch.jit.export
    def rotate(self,quat:Tensor,vector:Tensor):
        t=2*torch.cross(quat[:,:3],vector,dim=-1)
        return vector+quat[:,3:]*t+torch.cross(quat[:,:3],t,dim=-1)

    @torch.jit.export
    def world_inputs(self,quat:Tensor,grf_yaw:Tensor,wrench_yaw:Tensor):
        yaw=torch.atan2(2*(quat[:,3]*quat[:,2]+quat[:,0]*quat[:,1]),1-2*(quat[:,1].square()+quat[:,2].square()))
        c=yaw.cos();s=yaw.sin()
        rotation=torch.stack((torch.stack((c,-s,torch.zeros_like(c)),-1),torch.stack((s,c,torch.zeros_like(c)),-1),
            torch.stack((torch.zeros_like(c),torch.zeros_like(c),torch.ones_like(c)),-1)),1)
        return grf_yaw@rotation.transpose(1,2),torch.cat(((rotation@wrench_yaw[:,:3,None]).squeeze(-1),
            (rotation@wrench_yaw[:,3:,None]).squeeze(-1)),-1)

    def residual(self,Q:Tensor,p:Tensor,C:Tensor,e:Tensor,w:float):
        return Q+2*w*C.transpose(1,2)@C,p+2*w*(C.transpose(1,2)@e.unsqueeze(-1)).squeeze(-1)

    def scale_rows(self,G:Tensor,h:Tensor):
        scale=torch.maximum(G.square().sum(-1).sqrt(),h.abs()).clamp_min(1.)
        return G/scale.unsqueeze(-1),h/scale,scale

    @torch.jit.export
    def primary(self,d:Dict[str,Tensor])->Dict[str,Tensor]:
        tau=d['tau_nom'];n=tau.size(0)
        M=d['mass_matrix'];J=d['foot_jacobians'].reshape(n,12,18);Jb=d['base_jacobian']
        stance=d['contact_probability']>=self.threshold
        mask=stance.repeat_interleave(3,dim=1).to(tau.dtype)
        rhs=torch.cat((self.selector.expand(n,-1,-1),J.transpose(1,2)*mask[:,None,:],Jb.transpose(1,2),d['bias'].unsqueeze(-1)),-1)
        sol,info=torch.linalg.solve_ex(M,rhs,check_errors=False)
        mechanics_valid=(info==0)&torch.isfinite(sol).all(dim=2).all(dim=1)
        sol=torch.nan_to_num(sol,nan=0.,posinf=0.,neginf=0.)
        A=sol[:,:,:24];b=(sol[:,:,24:30]@d['wrench_pred_world'].unsqueeze(-1)).squeeze(-1)-sol[:,:,30]
        f=torch.where(stance[:,:,None],d['force_pred_world'].reshape(n,4,3),torch.zeros_like(d['force_pred_world'].reshape(n,4,3))).flatten(1)
        Q=torch.diag(self.diagonal).expand(n,-1,-1).clone()
        p=-self.diagonal*torch.cat((tau,f),-1)
        Jc=J*mask[:,:,None]
        Q,p=self.residual(Q,p,Jc@A/self.sc,((Jc@b.unsqueeze(-1)).squeeze(-1)+d['foot_acceleration_bias'].flatten(1)*mask)/self.sc,self.wc)
        q=d['base_quaternion'];omega=d['base_angular_velocity_world']
        yaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square()))
        c=yaw.cos();s=yaw.sin();zero=torch.zeros_like(c)
        R=torch.stack((torch.stack((c,s,zero),-1),torch.stack((-s,c,zero),-1)),1)
        up=torch.stack((2*(q[:,0]*q[:,2]+q[:,3]*q[:,1]),2*(q[:,1]*q[:,2]-q[:,3]*q[:,0]),1-2*(q[:,0].square()+q[:,1].square())),-1)
        tilt=torch.stack((-up[:,1],up[:,0],zero),-1)
        desired=-self.kp_tilt*(R@tilt.unsqueeze(-1)).squeeze(-1)-self.kd_tilt*(R@omega.unsqueeze(-1)).squeeze(-1)
        H=R@Jb[:,3:6]
        Q,p=self.residual(Q,p,H@A/self.sb,((H@b.unsqueeze(-1)).squeeze(-1)-desired)/self.sb,self.wb)
        conjugate=torch.cat((-q[:,:3],q[:,3:]),-1)
        linear=self.rotate(conjugate,d['base_linear_velocity_world']);angular=self.rotate(conjugate,omega)
        C=self.horizon*torch.cat((A[:,:2],A[:,5:6]),1)
        e=torch.cat((linear[:,:2],angular[:,2:3]),-1)+self.horizon*torch.cat((b[:,:2],b[:,5:6]),-1)-d['velocity_command']
        Q,p=self.residual(Q,p,C[:,:2]/self.sxy,e[:,:2]/self.sxy,self.wxy)
        Q,p=self.residual(Q,p,C[:,2:]/self.syaw,e[:,2:]/self.syaw,self.wyaw)
        vz=d['estimated_base_linear_velocity_world']
        height_desired=self.kp_height*(self.height_target-d['estimated_height'].reshape(-1))-self.kd_height*vz[:,2]
        Hh=Jb[:,2:3]
        eh=(Hh@b.unsqueeze(-1)).squeeze(-1)+torch.cross(omega,vz,dim=-1)[:,2:3]-height_desired[:,None]
        Q,p=self.residual(Q,p,Hh@A/self.sh,eh/self.sh,self.wh)
        dt=d['dt'].reshape(-1,1);v=d['joint_velocity'];position=d['joint_position']
        limits=d['torque_limits'].expand_as(tau);qmin=d['position_lower'];qmax=d['position_upper'];vmax=d['velocity_limits']
        lower=torch.maximum((-vmax-v)/dt,(qmin-position-dt*v)/(self.beta*dt.square()))
        upper=torch.minimum((vmax-v)/dt,(qmax-position-dt*v)/(self.beta*dt.square()))
        I=self.eye[:12].expand(n,-1,-1)
        G=torch.cat((I,-I,A[:,6:],-A[:,6:]),1)
        h=torch.cat((limits,limits,upper-b[:,6:],b[:,6:]-lower),1)
        for foot in range(4):
            G=torch.cat((G,self.friction[foot].expand(n,-1,-1)*stance[:,foot,None,None]),1)
            h=torch.cat((h,(~stance[:,foot,None]).to(tau.dtype).expand(-1,5)),1)
        factor=d['effective_kp']*self.beta*dt.square()+d['effective_kd']*dt
        endpoint=I-factor[:,:,None]*A[:,6:]
        offset=-d['effective_kp']*dt*v-factor*b[:,6:]
        G=torch.cat((G,endpoint,-endpoint),1);h=torch.cat((h,limits-offset,limits+offset),1)
        physical_G=G;physical_h=h
        G,h,rs=self.scale_rows(G*self.variable_scale,h)
        Q=Q*self.variable_scale[None,:,None]*self.variable_scale[None,None,:]
        Q=.5*(Q+Q.transpose(1,2))+self.reg*self.eye;p=p*self.variable_scale
        native_lower=torch.cat((-limits/self.variable_scale[:12],torch.full_like(tau,-float('inf'))),-1)
        native_upper=torch.cat((limits/self.variable_scale[:12],torch.full_like(tau,float('inf'))),-1)
        return {'Q':Q,'p':p,'G':G,'h':h,'physical_G':physical_G,'physical_h':physical_h,'row_scale':rs,
            'variable_scale':self.variable_scale,'native_lower':native_lower,'native_upper':native_upper,
            'acceleration_map':A,'acceleration_offset':b,'mechanics_valid':mechanics_valid,
            'joint_lower':lower,'joint_upper':upper,'tau_lower':-limits,'tau_upper':limits,
            'valid':mechanics_valid&(lower<=upper).all(-1)}

    @torch.jit.export
    def recovery(self,m:Dict[str,Tensor])->Dict[str,Tensor]:
        result=torch.jit.annotate(Dict[str,Tensor],{})
        for key,value in m.items(): result[key]=value
        n=m['p'].size(0);rows=m['G'].size(1)
        Q=m['Q'].new_zeros((n,36,36));Q[:,:24,:24]=m['Q']
        eye=self.eye[:12,:12]
        Q[:,24:,24:]=(2*self.slack_weight+self.reg)*eye
        scale=torch.cat((m['variable_scale'],m['p'].new_full((12,),self.slack_scale)))
        G=m['Q'].new_zeros((n,rows+12,36));G[:,:rows,:24]=m['physical_G']
        G[:,24:36,24:]=-eye;G[:,36:48,24:]=-eye;G[:,rows:,24:]=-eye
        h=torch.cat((m['physical_h'],m['p'].new_zeros((n,12))),1)
        result['Q']=Q;result['p']=torch.cat((m['p'],m['p'].new_zeros((n,12))),1)
        result['physical_G']=G;result['physical_h']=h
        gs,hs,rs=self.scale_rows(G*scale,h)
        result['G']=gs;result['h']=hs;result['row_scale']=rs;result['variable_scale']=scale
        result['native_lower']=torch.cat((m['native_lower'],m['p'].new_zeros((n,12))),1)
        result['native_upper']=torch.cat((m['native_upper'],m['p'].new_full((n,12),float('inf'))),1)
        result['acceleration_map']=torch.cat((m['acceleration_map'],m['p'].new_zeros((n,18,12))),2)
        result['valid']=m['mechanics_valid']
        return result

    @torch.jit.export
    def candidate(self,z:Tensor,m:Dict[str,Tensor],contact:Tensor):
        x=z*m['variable_scale']
        torque=torch.maximum(torch.minimum(x[:,:12],m['tau_upper']),m['tau_lower'])
        f=x[:,12:24].reshape(-1,4,3)
        f=torch.where((contact>=self.threshold)[:,:,None],f,torch.zeros_like(f)).flatten(1)
        x=torch.cat((torque,f,x[:,24:]),-1)
        normalized=x/m['variable_scale']
        residual=((m['G']@normalized.unsqueeze(-1)).squeeze(-1)-m['h']).clamp_min(0).amax(-1)
        feasible=torch.isfinite(z).all(-1)&torch.isfinite(x).all(-1)&(residual<=.001)&m['valid']
        return x,feasible,residual

    @torch.jit.export
    def fallback(self,nominal:Tensor,limits:Tensor):
        return torch.maximum(torch.minimum(torch.nan_to_num(nominal,nan=0.,posinf=0.,neginf=0.),limits),-limits)

    @torch.jit.export
    def feedforward_command(self,selected_torque:Tensor,physical_pd:Tensor,motor:Tensor,feedforward_weight:Tensor):
        gain=motor*feedforward_weight;available=gain!=0
        command=(selected_torque-physical_pd)/torch.where(available,gain,torch.ones_like(gain))
        return torch.where(available,command,torch.zeros_like(command)),available
