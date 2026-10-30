"""Portable diagnostics on recorded Exp2 windows; no model/runner/simulator construction.

Only the saved original FiLM MLP and linear heads are evaluated, using NumPy.
Local input probes hold recorded pre-features/state fixed, not a new trajectory.
"""
from pathlib import Path
import json
import shutil
import tarfile

import numpy as np
import pandas as pd
from scipy import signal, stats, special

import exp1_analysis as shared
import exp3_film_oscillation as film
from exp2_task_diagnostics import amplitude, ratio, stages
from exp3_oscillation_physical import REQUIRED

FIELDS = tuple(dict.fromkeys((*film.REQUIRED, *film.OPTIONAL,
    'commands_t','command_nominal_t','base_position_world','source_action_id',
    'action_clipped','delayed_action_interval_mean','pd_weighted_motor_mean_square',
    'ff_weighted_motor_mean_square','executed_torque_mean_square',
    'torque_request_motor_scaled_interval_mean','ee_reference_velocity_world',
    'phase','task_stage_interval','task_stage_successor','load_event_index','base_command_event_index',
    'command_nominal','environment_id','ee_reference_sphere')))
GROUP = ['task','checkpoint','profile','loading','stage','motion','scheduled_load_n']
IDENTITY = GROUP + ['evaluation_seed','training_seed','trial_id','run','window_id','start_s','end_s']


def detrended(x, time):
    """OLS per signed component: x = bias + slope*(t-mean(t)) + residual."""
    x=np.asarray(x,float); t=np.asarray(time,float); t=t-t.mean()
    if x.ndim==1: x=x[:,None]
    if len(t)<3 or np.any(np.diff(time)<=0) or not np.isfinite(x).all():
        raise ValueError('Need finite, ordered component signals for detrending')
    bias=x.mean(0); slope=t@(x-bias)/(t@t)
    trend=bias+t[:,None]*slope
    return dict(bias=bias,slope=slope,trend=trend,residual=x-trend,
                rms=np.sqrt(np.mean((x-trend)**2,0)),drift_rms=np.sqrt(np.mean((trend-bias)**2,0)))


def input_agreement(estimated, measured, epsilon):
    """Same normalized command-minus-state coordinates; no divisions near zero."""
    e,m=np.asarray(estimated,float),np.asarray(measured,float)
    en,mn=np.linalg.norm(e,axis=1),np.linalg.norm(m,axis=1)
    valid=mn>epsilon; directional=valid&(en>epsilon)
    out=dict(measured_rms=float(np.sqrt(np.mean(m*m))),estimated_rms=float(np.sqrt(np.mean(e*e))),
             magnitude_ratio=float(np.mean(en[valid]/mn[valid])) if valid.any() else np.nan,
             direction_cosine=float(np.mean(np.sum(e[directional]*m[directional],1)/(en[directional]*mn[directional]))) if directional.any() else np.nan,
             magnitude_samples=int(valid.sum()),direction_samples=int(directional.sum()),
             measured_near_zero=int((~valid).sum()),estimated_near_zero=int((en<=epsilon).sum()))
    for i in range(m.shape[1]):
        active=abs(m[:,i])>epsilon
        out.update({f'axis{i}_bias':float((e[:,i]-m[:,i]).mean()),
                    f'axis{i}_rms_ratio':ratio(np.sqrt(np.mean(e[:,i]**2)),np.sqrt(np.mean(m[:,i]**2)),epsilon),
                    f'axis{i}_gain':float(m[:,i]@e[:,i]/(m[:,i]@m[:,i])) if np.sqrt(np.mean(m[:,i]**2))>epsilon else np.nan,
                    f'axis{i}_sign_agreement':float(np.mean(np.sign(e[active,i])==np.sign(m[active,i]))) if active.any() else np.nan})
    return out


def rotate(v,q):
    return film.inverse_rotate(v,np.asarray(q)*np.array([-1.,-1.,-1.,1.]))


def motion_split(d,dt):
    """Exact endpoint displacement identity, world m/s; not a causal arm/leg split.

    R1(r1-r0) includes all EE motion relative to the torso. No Jacobian, estimated
    base acceleration, or inconsistent root-COM velocity is needed here.
    """
    p0,p1=d['ee_position_t_world'],d['ee_position_world']
    b0,b1=d['base_position_t'],d['base_position_world']
    q0,q1=d['base_quaternion_t'],d['base_quaternion_xyzw']
    r0=film.inverse_rotate(p0-b0,q0); r1=film.inverse_rotate(p1-b1,q1)
    trans=(b1-b0)/dt; rotation=(rotate(r0,q1)-rotate(r0,q0))/dt
    relative=rotate(r1-r0,q1)/dt
    ref=(d['ee_reference_world']-d['ee_reference_t_world'])/dt
    return dict(base_translation=trans,base_rotation=rotation,torso_relative=relative,
                reference_velocity=ref,reference_relative=(p1-p0)/dt-ref,
                closure=(p1-p0)/dt-trans-rotation-relative)


def torque_channels(action,d,meta,delayed=False):
    """Same recorded state for all candidates, branch/motor weighted, Nm.

    This is an interval-START request, not substep execution. Delayed comparisons
    use recorded source IDs; absent sources are NaN, never wrapped/padded.
    """
    cfg=meta['resolved_environment']; n=cfg['env']['num_actions']
    a=np.asarray(action,float)
    valid=np.ones(len(a),bool)
    if delayed:
        ids=np.searchsorted(d['step'],d['source_action_id'])
        valid=(ids<len(a)) & (d['step'][np.minimum(ids,len(a)-1)]==d['source_action_id'])
        a=a[np.minimum(ids,len(a)-1)].copy(); a[~valid]=np.nan
    a=np.clip(a,-cfg['normalization']['clip_actions'],cfg['normalization']['clip_actions'])
    qscale,tscale=film.action_scales(meta)
    defaults=np.array([cfg['init_state']['default_joint_angles'][k] for k in meta['dof_names']])
    target=np.broadcast_to(defaults,d['joint_position_t'].shape).copy(); target[:,:n]+=a[:,:n]*qscale
    pd=(d['nominal_kp']*d['mechanics_kp_scale']*(target-d['joint_position_t'])-
        d['nominal_kd']*d['mechanics_kd_scale']*d['joint_velocity_t'])
    ff=np.zeros_like(pd)
    if a.shape[1]==2*n: ff[:,:n]=a[:,n:]*tscale
    elif a.shape[1]!=n: raise ValueError('Unknown action channel layout')
    # B1Z1 position-only execution ignores coupled-controller branch weights.
    pd*=d['mechanics_motor_strength']*(d['mechanics_feedback_tau_weight'] if a.shape[1]==2*n else 1.)
    ff*=d['mechanics_motor_strength']*(d['mechanics_feedforward_tau_weight'] if a.shape[1]==2*n else 1.)
    request=pd+ff; limit=d['actual_clipping_limits']
    return dict(pd=pd,ff=ff,total=request,clipped=np.clip(request,-limit,limit),valid=valid,target=target)


def channel_coupling(pd,ff,time):
    """Covariance is retained: variances of coupled channels are NOT additive."""
    p=detrended(pd,time)['residual']; f=detrended(ff,time)['residual']
    vp,vf=np.mean(p*p),np.mean(f*f); cross=np.mean(p*f); total=np.mean((p+f)**2)
    return dict(pd_variance=vp,ff_variance=vf,cross_covariance=cross,total_variance=total,
                reinforcement=2*cross/(vp+vf) if vp+vf>1e-20 else np.nan,
                variance_closure=total-vp-vf-2*cross,
                cancellation_nm=float(np.mean((abs(pd)+abs(ff)-abs(pd+ff))/2)),
                residual_cancellation_nm=float(np.mean((abs(p)+abs(f)-abs(p+f))/2)))


def conditioning_network(x,weights,activation):
    """Replay ONLY the original saved unnormalized MLP; reject V02/LayerNorm."""
    indices=sorted(int(k.split('.')[0]) for k in weights if k.endswith('.weight'))
    if not indices: raise ValueError('Original conditioning network weights unavailable')
    for j,i in enumerate(indices):
        w=weights[f'{i}.weight']; b=weights[f'{i}.bias']
        if w.ndim!=2: raise ValueError('Not the original actor-feature conditioning MLP')
        x=np.asarray(x)@w.T+b
        if j<len(indices)-1:
            if activation=='elu': x=np.where(x>0,x,np.expm1(np.minimum(x,0)))
            elif activation=='relu': x=np.maximum(x,0)
            elif activation=='tanh': x=np.tanh(x)
            elif activation=='sigmoid': x=special.expit(x)
            elif activation in ('lrelu','leaky_relu'): x=np.where(x>=0,x,.01*x)
            else: raise ValueError('Unsupported recorded activation: '+activation)
    return np.split(x,2,axis=1)


def head_actions(features,heads):
    return np.concatenate([features@heads[k][0].T+heads[k][1] for k in ('position','torque') if k in heads],axis=1)


def action_variants(d,heads):
    pre=d['film_features_pre_t']
    fixed=(1+d['film_gamma_t'].mean(0))*pre+d['film_beta_t'].mean(0)
    return {k:head_actions(x,heads) for k,x in
            [('original',d['film_features_post_t']),('fixed',fixed),('bypass',pre)]}


def probes(condition,epsilon,magnitude_step,direction_step):
    """Central coordinate/radial/directional probes in OLD normalized coordinates."""
    if condition.shape[1]!=6: raise ValueError('Requires original base(3)+EE(3) conditioning')
    for group,sl in [('base',slice(0,3)),('ee',slice(3,6))]:
        frozen=condition.copy(); frozen[:,sl]=condition[:,sl].mean(0)
        yield group+'.freeze',frozen,None,1.
        for i in range(3):
            plus,minus=condition.copy(),condition.copy()
            plus[:,sl.start+i]+=epsilon; minus[:,sl.start+i]-=epsilon
            yield group+f'.coordinate{i}',plus,minus,2*epsilon
        plus,minus=condition.copy(),condition.copy()
        plus[:,sl]*=1+magnitude_step; minus[:,sl]*=1-magnitude_step
        yield group+'.magnitude',plus,minus,2*magnitude_step
        v=condition[:,sl]; norm=np.linalg.norm(v,axis=1,keepdims=True)
        axis=np.eye(3)[np.argmin(abs(v),axis=1)]; tangent=np.cross(v,axis)
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
        plus,minus=condition.copy(),condition.copy()
        plus[:,sl]=np.cos(direction_step)*v+np.sin(direction_step)*norm*tangent
        minus[:,sl]=np.cos(direction_step)*v-np.sin(direction_step)*norm*tangent
        yield group+'.direction',plus,minus,2*direction_step


def spectral_curves(signals,dt,cfg):
    """Identical component detrending/Hann Welch bins; export complex CSD numerically."""
    names=[k for k,x in signals.items() if np.asarray(x).ndim==1 and np.isfinite(x).all()]
    if not names: return {}
    plan=film.spectral_plan(len(signals[names[0]]),dt,cfg)
    if not plan['available']: return {}
    x=np.column_stack([signals[k] for k in names]); x=signal.detrend(x,axis=0,type='linear')
    kw=dict(fs=1/dt,nperseg=plan['nperseg'],noverlap=int(plan['nperseg']*cfg['overlap_fraction']),detrend=False,window='hann')
    freq,psd=signal.welch(x,axis=0,**kw)
    pairs=[(i,j) for i,n in enumerate(names) for j,m in enumerate(names)
           if n in ('feature_pc1','conditioning_ee_pitch','ff_arm_joint') and m in ('ee_x','ee_y','ee_z')]
    cross=[]; coherences=[]; phases=[]
    for i,j in pairs:
        _,c=signal.csd(x[:,i],x[:,j],**kw); den=psd[:,i]*psd[:,j]
        coh=np.divide(abs(c)**2,den,out=np.full_like(den,np.nan),where=den>1e-25)
        cross.append(c); coherences.append(coh)
        phases.append(np.where((coh>=cfg['coherence_threshold'])&(freq>=plan['usable_low_hz'])&(freq<=plan['usable_high_hz']),np.angle(c),np.nan))
    return dict(frequency_hz=freq,signal_names=np.asarray(names),psd=psd,
                pairs=np.asarray([(names[i],names[j]) for i,j in pairs]).reshape(-1,2),
                cross_spectrum=np.asarray(cross),coherence=np.asarray(coherences),phase_rad=np.asarray(phases),
                segments=np.asarray(plan['segments']))


def analyze_window(d,meta,cfg,heads=None,basis=None):
    t=d['time_seconds']; out={'samples':len(t)}; arrays=dict(d); missing=[]
    err=film.inverse_rotate(d['ee_position_t_world']-d['ee_reference_t_world'],d['base_quaternion_t'],True)
    fit=detrended(err,t); arrays.update(ee_error=err,ee_trend=fit['trend'],ee_residual=fit['residual'])
    for i,axis in enumerate('xyz'):
        for key,unit in [('bias','m'),('slope','m_s'),('rms','m'),('drift_rms','m')]: out[f'ee.{axis}.{key}_{unit}']=fit[key][i]
        out[f'ee.{axis}.centered_rms_m']=np.sqrt(np.mean((err[:,i]-fit['bias'][i])**2))
    out['ee.mean_error_m']=np.linalg.norm(err,axis=1).mean()
    out['ee.vector_rms_m']=np.sqrt(np.mean(np.sum(err*err,axis=1)))
    out['ee.energy_closure_m2']=out['ee.vector_rms_m']**2-np.sum(fit['bias']**2+fit['drift_rms']**2+fit['rms']**2)
    out['ee.adherence']=np.mean(np.linalg.norm(err,axis=1)<=meta['evaluation']['ee_tolerance_m'])
    if 'ee_force_applied_world_interval_mean' in d:
        loads=np.linalg.norm(d['ee_force_applied_world_interval_mean'],axis=1)
        out['load.mean_n']=loads.mean(); out['load.peak_n']=loads.max()
    spectral={f'ee_{a}':err[:,i] for i,a in enumerate('xyz')}
    groups=film.joint_groups(meta,cfg); n=meta['resolved_environment']['env']['num_actions']
    qscale,tscale=film.action_scales(meta)
    if 'action_raw' in d:
        arrays['recorded_position_command']=d['action_raw'][:,:n]*qscale
        if d['action_raw'].shape[1]==2*n: arrays['recorded_ff_command']=d['action_raw'][:,n:]*tscale
    for group,ids in groups.items():
        active=[i for i in ids if i<n]
        for channel in ('position','ff'):
            key='recorded_'+channel+'_command'
            if active and key in arrays: out[f'recorded.action.{group}.{channel}.rms']=amplitude(arrays[key][:,active])
        if not ids: continue
        for channel,key in [('pd','pd_weighted_motor'),('ff','ff_weighted_motor'),('executed','executed_torque')]:
            if key+'_interval_mean' in d: out[f'recorded.{group}.{channel}.interval_rms_nm']=amplitude(d[key+'_interval_mean'][:,ids])
            if key+'_mean_square' in d: out[f'recorded.{group}.{channel}.substep_rms_nm']=np.sqrt(np.mean(d[key+'_mean_square'][:,ids]))
        if 'opposing_channel_cancellation_mean' in d: out[f'recorded.{group}.substep_cancellation_nm']=np.mean(d['opposing_channel_cancellation_mean'][:,ids])
    if all(k in d for k in ('pd_weighted_motor_interval_mean','ff_weighted_motor_interval_mean','torque_request_motor_scaled_interval_mean')):
        out['recorded.torque_accounting_max_nm']=np.max(abs(d['pd_weighted_motor_interval_mean']+d['ff_weighted_motor_interval_mean']-d['torque_request_motor_scaled_interval_mean']))
        if out['recorded.torque_accounting_max_nm']>cfg.get('recorded_torque_atol_nm',1e-3):
            missing.append('Recorded PD+FF does not match requested interval torque; inspect accounting error')
    try:
        body_v=film.inverse_rotate(d['base_velocity_world_t'],d['base_quaternion_t'])
        body_w=film.inverse_rotate(d['base_angular_velocity_world_t'],d['base_quaternion_t'])
        cmd=d.get('command_nominal_t',d.get('commands_t'))[:,:3]
        base_error=np.column_stack((body_v[:,:2],body_w[:,2]))-cmd
        arrays['base_error']=base_error; out['base.forward_bias_m_s']=base_error[:,0].mean()
        out['base.adherence']=np.mean((np.linalg.norm(base_error[:,:2],axis=1)<=meta['evaluation']['velocity_tolerance_m_s']) & (abs(base_error[:,2])<=meta['evaluation']['yaw_tolerance_rad_s']))
        spectral['base_forward_error']=base_error[:,0]
    except (KeyError,TypeError) as exc: missing.append('base tracking: '+str(exc))
    try:
        for name,x in motion_split(d,meta['dt']).items():
            arrays['motion_'+name]=x
            for i,a in enumerate('xyz'): out[f'motion.{name}.{a}.rms_m_s']=np.sqrt(np.mean(x[:,i]**2))
    except KeyError as exc: missing.append('endpoint motion split: '+str(exc))
    try:
        if 'film_condition_t' not in d: raise KeyError('FiLM inputs not recorded/applicable')
        measured,_=film.measured_condition(d,meta); estimated=d['film_condition_t']
        if estimated.shape!=measured.shape: raise ValueError('Not the original six conditioning coordinates')
        arrays['condition_measured']=measured
        if all(k in d for k in ('prediction_base_velocity','prediction_ee_position')):
            expected=d['obs_t'][:,-6:]-np.c_[d['prediction_base_velocity'],d['prediction_ee_position']]
            out['input.recorded_contract_max_error']=float(np.max(abs(expected-estimated)))
        for group,sl in [('base',slice(0,3)),('ee',slice(3,6))]:
            out.update({f'input.{group}.'+(k.replace('_','.',1) if k.startswith('axis') else k):v
                        for k,v in input_agreement(estimated[:,sl],measured[:,sl],cfg.get('input_zero_epsilon',1e-3)).items()})
        for i in range(6):
            out[f'input.measured{i}.mean']=measured[:,i].mean(); out[f'input.estimated{i}.mean']=estimated[:,i].mean()
        spectral['conditioning_ee_pitch']=estimated[:,4]
    except (KeyError,ValueError) as exc: missing.append('conditioning agreement: '+str(exc))
    if basis and 'film_features_pre_t' in d:
        for k in ('pre','post'):
            arrays['feature_'+k+'_pc']=(d['film_features_'+k+'_t']-basis['center'])@np.asarray(basis['axes']).T
        spectral['feature_pc1']=arrays['feature_post_pc'][:,0]
    variants={}; torques={}
    if heads:
        try:
            variants=action_variants(d,heads)
            error=abs(variants['original']-d['action_raw']); out['action_reconstruction_max_normalized']=error.max()
            if not np.allclose(variants['original'],d['action_raw'],atol=cfg.get('action_reference_atol',.005),rtol=cfg.get('action_reference_rtol',.01)):
                raise ValueError('Large head mismatch; see reconstruction error')
        except (KeyError,ValueError) as exc: missing.append('action references: '+str(exc)); variants={}
    else: missing.append('original frozen heads unavailable/not applicable')
    for variant,a in variants.items():
        arrays['action_'+variant]=a
        for group,ids in groups.items():
            ids=[i for i in ids if i<n]
            if not ids: continue
            channels=[('position',a[:,:n]*qscale)]
            if a.shape[1]==2*n: channels.append(('ff',a[:,n:]*tscale))
            for channel,x in channels:
                if x.shape[1]!=n: continue
                out[f'action.{variant}.{group}.{channel}.rms']=amplitude(x[:,ids])
                out[f'action.{variant}.{group}.{channel}.centered_rms']=amplitude(x[:,ids],'constant')
        try:
            torques[variant]=torque_channels(a,d,meta)
            delayed=torque_channels(a,d,meta,True)
            out['delay.valid_fraction']=delayed['valid'].mean()
            for k,x in delayed.items(): arrays[f'delayed_{variant}_{k}']=x
            for k,x in torques[variant].items(): arrays[f'torque_{variant}_{k}']=x
            for group,ids in groups.items():
                if not ids: continue
                for channel in ('pd','ff','total','clipped'):
                    out[f'torque.{variant}.{group}.{channel}.rms_nm']=amplitude(torques[variant][channel][:,ids])
                    # Only a contiguous suffix is expected for a fixed episode delay.
                    keep=delayed['valid']
                    if keep.any() and np.all(keep[np.flatnonzero(keep)[0]:]):
                        out[f'delayed.{variant}.{group}.{channel}.rms_nm']=amplitude(delayed[channel][keep][:,ids])
                out.update({f'coupling.{variant}.{group}.{k}':v for k,v in channel_coupling(torques[variant]['pd'][:,ids],torques[variant]['ff'][:,ids],t).items()})
        except (KeyError,ValueError) as exc: missing.append('torque accounting: '+str(exc))
    for reference in ('fixed','bypass'):
        if reference not in variants: continue
        for group,ids in groups.items():
            ids=[i for i in ids if i<n]
            if not ids: continue
            for channel in ('position','ff'):
                k=f'action.original.{group}.{channel}.rms'; r=f'action.{reference}.{group}.{channel}.rms'
                if k in out and r in out:
                    out[f'ratio.{reference}.{group}.{channel}']=ratio(out[k],out[r],cfg.get('ratio_epsilon',1e-6))
                    if not np.isfinite(out[f'ratio.{reference}.{group}.{channel}']):
                        missing.append(f'ratio {reference}/{group}/{channel}: near-zero/nonfinite denominator')
            if reference in torques and 'original' in torques:
                dp=torques['original']['pd'][:,ids]-torques[reference]['pd'][:,ids]
                df=torques['original']['ff'][:,ids]-torques[reference]['ff'][:,ids]
                out.update({f'delta.{reference}.{group}.{k}':v for k,v in channel_coupling(dp,df,t).items()})
    if groups.get('arm') and 'original' in variants and variants['original'].shape[1]==2*n:
        # Named first active arm joint, fixed from metadata, not correlation-selected.
        i=groups['arm'][0]
        for v,a in variants.items(): spectral['ff_arm_joint' if v=='original' else 'ff_arm_'+v]=a[:,n+i]*tscale[i]
    if variants and cfg.get('conditioning_sensitivity',True):
        try:
            sensitivity(d,meta,cfg,heads,groups,variants,torques,out,arrays)
        except (KeyError,ValueError) as exc: missing.append('conditioning sensitivity: '+str(exc))
    curves=spectral_curves(spectral,meta['dt'],cfg)
    # Correlations are on detrended signed signals, not norms; keep weak results.
    for source in ('feature_pc1','ff_arm_joint','conditioning_ee_pitch'):
        if source not in spectral: continue
        x=signal.detrend(spectral[source])
        for axis in 'xyz':
            y=signal.detrend(spectral['ee_'+axis])
            out[f'correlation.{source}.ee_{axis}']=float(np.corrcoef(x,y)[0,1]) if min(np.std(x),np.std(y))>1e-12 else np.nan
    out['missing']=json.dumps(sorted(set(missing))); out['spectra_available']=bool(curves)
    return out,arrays,curves


def sensitivity(d,meta,cfg,heads,groups,variants,torques,out,arrays):
    c=d['film_condition_t']; pre=d['film_features_pre_t']; weights=heads['_conditioning']
    activation=meta['resolved_training']['policy']['activation']
    def evaluate(condition):
        g,b=conditioning_network(condition,weights,activation)
        a=head_actions((1+g)*pre+b,heads)
        return g,b,a
    g,b,base=evaluate(c)
    if not np.allclose(np.c_[g,b],np.c_[d['film_gamma_t'],d['film_beta_t']],atol=cfg.get('action_reference_atol',.005),rtol=cfg.get('action_reference_rtol',.01)):
        raise ValueError('Saved conditioning MLP does not match recorded gamma/beta')
    out['conditioning_reconstruction_max']=float(np.max(abs(np.c_[g,b]-np.c_[d['film_gamma_t'],d['film_beta_t']])))
    n=meta['resolved_environment']['env']['num_actions']; qs,ts=film.action_scales(meta)
    for name,plus,minus,den in probes(c,cfg.get('sensitivity_epsilon',.01),cfg.get('sensitivity_magnitude_step',.05),cfg.get('sensitivity_direction_step',.05)):
        gp,bp,ap=evaluate(plus); gm,bm,am=evaluate(minus) if minus is not None else (g,b,base)
        sl=slice(0,3) if name.startswith('base.') else slice(3,6)
        keep=np.linalg.norm(c[:,sl],axis=1)>cfg.get('input_zero_epsilon',1e-3) if name.endswith('.direction') else np.ones(len(c),bool)
        out['sensitivity.'+name+'.samples']=int(keep.sum())
        def rms(x): return np.sqrt(np.mean(x[keep]**2)) if keep.any() else np.nan
        out['sensitivity.'+name+'.gamma_rms']=rms((gp-gm)/den)
        out['sensitivity.'+name+'.beta_rms']=rms((bp-bm)/den)
        da=(ap-am)/den; arrays['sensitivity_action_'+name]=da
        for group,ids in groups.items():
            ids=[i for i in ids if i<n]
            if not ids: continue
            channels=[('position',da[:,:n]*qs)]
            if da.shape[1]==2*n: channels.append(('ff',da[:,n:]*ts))
            for channel,x in channels:
                if x.shape[1]==n: out[f'sensitivity.{name}.{group}.{channel}_rms']=rms(x[:,ids])
        if torques:
            tp,tm=torque_channels(ap,d,meta),torque_channels(am,d,meta)
            for channel in ('pd','ff','total','clipped'):
                diff=(tp[channel]-tm[channel])/den
                for group,ids in groups.items():
                    if ids: out[f'sensitivity.{name}.{group}.{channel}_nm_rms']=rms(diff[:,ids])
    for source in ('base','ee'):
        for group in groups:
            for channel in ('position','ff','pd_nm','ff_nm','total_nm','clipped_nm'):
                values=[out.get(f'sensitivity.{source}.coordinate{i}.{group}.{channel}_rms',np.nan) for i in range(3)]
                out[f'sensitivity.{source}.jacobian.{group}.{channel}_rms']=np.sqrt(np.sum(np.square(values)))


def aggregate_windows(frame):
    """Windows equally within trials; trials equally within evaluation seed."""
    from plot_exp2_film_package import aggregate_windows as shared_aggregate
    return shared_aggregate(frame)


def associations(trials):
    pairs=[(f'ee.{axis}.{part}_m',target) for axis in 'xyz' for part in ('bias','drift_rms','rms')
           for target in ('ee.mean_error_m','ee.adherence')]+[
           ('ratio.bypass.arm.ff','ee.z.rms_m'),('input.ee.axis1.gain','ee.z.bias_m'),
           ('motion.base_rotation.z.rms_m_s','ee.z.rms_m')]
    rows=[]
    for identity,g in trials.groupby(GROUP+['evaluation_seed'],dropna=False):
        for x,y in pairs:
            if x not in g or y not in g: continue
            valid=g[[x,y]].dropna(); n=len(valid)
            r=float(stats.spearmanr(valid[x],valid[y]).statistic) if n>=3 and valid[x].nunique()>1 and valid[y].nunique()>1 else np.nan
            rows.append(dict(zip(GROUP+['evaluation_seed'],identity),x=x,y=y,spearman=r,trials=n))
    return pd.DataFrame(rows)


def select_examples(frame,budget):
    """Low/median/high AMONG TRIAL amplitudes, then nearest window, deterministic."""
    picks=[]
    for key,g in frame.groupby(GROUP,dropna=False,sort=True):
        ts=g.groupby(['evaluation_seed','trial_id','run'])['ee.z.rms_m'].mean().reset_index()
        ts=ts[np.isfinite(ts['ee.z.rms_m'])]
        for level,q in [('median',.5),('low',.1),('high',.9)]:
            if ts.empty: continue
            selected=ts.assign(distance=abs(ts['ee.z.rms_m']-ts['ee.z.rms_m'].quantile(q))).sort_values(['distance','evaluation_seed','trial_id','run']).iloc[0]
            windows=g[(g.evaluation_seed==selected.evaluation_seed)&(g.trial_id==selected.trial_id)&(g.run==selected.run)]
            w=windows.assign(distance=abs(windows['ee.z.rms_m']-selected['ee.z.rms_m'])).sort_values(['distance','window_id']).iloc[0]
            picks.append(dict(w.to_dict(),selection_level=level))
    # Balance methods/tasks, load/motion phases, then amplitude levels. Chronological
    # truncation would spend a small budget only on standing/unloaded examples.
    result=[]; counts=[{}, {}, {}, {}, {}, {}]
    def keys(r): return [r['task'],r['profile'],(r['task'],r['profile']),
                        (str(r['scheduled_load_n']),r['motion'],r.get('phase','')),r['selection_level'],r['checkpoint']]
    while picks and len(result)<budget:
        chosen=min(picks,key=lambda r:tuple(c.get(k,0) for c,k in zip(counts,keys(r)))+
                   (r['selection_level']!='median',r['window_id']))
        result.append(chosen)
        for count,key in zip(counts,keys(chosen)): count[key]=count.get(key,0)+1
        picks=[r for r in picks if r['window_id']!=chosen['window_id']]
    return result


def metric_units(name):
    """Units follow explicit metric namespaces, not the table's display rounding."""
    if name.startswith('sensitivity.'):
        output='Nm' if '_nm_' in name or '.ff_' in name else 'rad' if '.position_' in name else '1'
        if name.endswith('.samples'): return 'samples'
        divisor='normalized-input unit' if '.coordinate' in name or '.jacobian.' in name else 'fraction' if '.magnitude.' in name else 'rad of input-direction rotation' if '.direction.' in name else None
        return output+('/'+divisor if divisor else '')
    if name.startswith(('ratio.','correlation.')) or name.endswith(('adherence','valid_fraction','reinforcement')): return '1'
    if name.startswith('load.'): return 'N'
    if name.startswith('motion.'): return 'm/s (world)'
    if name.endswith('_m2'): return 'm^2'
    if name.endswith('_m_s'): return 'm/s'
    if name.startswith(('ee.','base.')): return 'm' if name.startswith('ee.') else 'm/s'
    if name.startswith('action.'): return 'rad' if '.position.' in name else 'Nm'
    if name.startswith('recorded.'): return 'rad' if '.position.' in name else 'Nm'
    if name.startswith(('coupling.','delta.')): return 'Nm' if 'cancellation' in name else 'Nm^2'
    if name.startswith(('torque.','delayed.')): return 'Nm'
    if name.startswith('input.'):
        return 'samples' if name.endswith(('samples','zero')) else 'normalized coordinate' if name.endswith(('.bias','.mean','_rms','error')) else '1'
    return 'normalized coordinate' if 'reconstruction' in name else 'count/identifier; see column name'


def array_units(name):
    base=name.split('__')[0]
    if base in ('time_seconds','successor_time_seconds','dt'): return 's'
    if base in ('actual_clipping_limits','rated_torque_limits'): return 'Nm'
    if base.startswith('sensitivity_action_'): return 'normalized action / probe unit (see sensitivity convention)'
    if base.startswith(('action_','film_','feature_','condition_','obs_','prediction_')): return 'native normalized coordinate; observation scales in environment metadata'
    if base.startswith('motion_'): return 'm/s world'
    if base.startswith(('torque_','delayed_')):
        return 'rad' if base.endswith('_target') else 'boolean' if base.endswith('_valid') else 'Nm'
    if 'quaternion' in base: return 'unit quaternion xyzw'
    if 'angular_velocity' in base: return 'rad/s'
    if 'velocity' in base: return 'rad/s' if 'joint' in base else 'm/s'
    if 'position' in base or base.startswith(('ee_error','ee_trend','ee_residual')): return 'rad' if 'joint' in base or 'command' in base else 'm'
    if 'reference' in base: return '[m,rad,rad]' if 'sphere' in base else 'm'
    if 'force' in base: return 'N'
    if 'torque' in base or 'ff_command' in base: return 'Nm^2' if 'square' in base else 'Nm'
    if 'commands' in base or 'command_nominal' in base or base=='base_error': return '[m/s,m/s,rad/s] (leading three components)'
    if base=='nominal_kp': return 'Nm/rad'
    if base=='nominal_kd': return 'Nm/(rad/s)'
    return 'dimensionless/index/flag; recorded field name and mechanics metadata apply'


class PackageBuilder:
    """All window metrics/spectra; bounded traces reread once after selection."""
    def __init__(self,cfg,output):
        for key,value in dict(conditioning_sensitivity=True,sensitivity_epsilon=.01,sensitivity_magnitude_step=.05,
            sensitivity_direction_step=.05,input_zero_epsilon=1e-3,package_example_windows=24,
            feature_pairs_per_window=16,spectra_windows_per_part=128,action_reference_atol=.005,action_reference_rtol=.01).items():
            cfg.setdefault(key,value)
        for key in ('sensitivity_epsilon','sensitivity_magnitude_step','sensitivity_direction_step','input_zero_epsilon'):
            if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError(key+' must be finite and positive')
        self.cfg=cfg; self.output=Path(output)/'offline_package'; self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'COMPLETE').unlink(missing_ok=True)
        self.runs={}; self.heads={}; self.rows=[]; self.metric_parts=[]; self.parts=[]; self.spectra={}; self.files=[]; self.dictionary={}

    def register(self,run,is_film):
        meta=run['meta']; key=str(run['path']); self.runs[key]=run
        checkpoint=meta['checkpoint_sha256']
        # Current Pos can also have the ORIGINAL actor-feature FiLM, with only a
        # position head. Select by recorded architecture/fields, not task spelling.
        policy=meta['resolved_training']['policy']
        is_film=is_film or (set(film.FILM)<=set(run['records']['fields']) and not policy.get('film_v02_enabled',False)
                           and policy.get('conditioning_mode','film')=='film')
        if is_film and checkpoint not in self.heads:
            try: self.heads[checkpoint]=film.load_heads(meta,self.cfg,include_conditioning=True)
            except (ValueError,OSError,RuntimeError,KeyError) as exc: self.heads[checkpoint]=(None,str(exc))
        if is_film and not self.cfg.get('feature_bases',{}).get(checkpoint):
            fitted=film.fit_feature_bases([run],self.cfg)
            for basis in fitted.values(): basis['source_recordings']=[str(run['path'])]
            self.cfg.setdefault('feature_bases',{}).update(fitted)

    def accept(self,d,meta,identity,spec):
        row=dict(identity,profile=meta['arguments']['profile'],loading=meta['arguments']['loading'],start_s=spec['start'],end_s=spec['end'],stage=spec['stage'],
                 motion=spec['motion_phase'],scheduled_load_n=spec['scheduled_load_n'])
        row['phase']=next((s['phase'] for s in stages(meta) if s['start_s']<=spec['start']<s['end_s']),'unknown')
        heads,reason=self.heads.get(meta['checkpoint_sha256'],(None,'Not an original FiLM policy'))
        values,_,spectra=analyze_window(d,meta,self.cfg,heads,self.cfg.get('feature_bases',{}).get(meta['checkpoint_sha256']))
        row.update(values); row['head_status']=reason or 'available'; self.rows.append(row)
        if len(self.rows)>=128: self.flush_metrics()
        if spectra:
            self.spectra.update({f'w{row["window_id"]:06d}__{k}':v for k,v in spectra.items()})
            if len(self.spectra)>=self.cfg.get('spectra_windows_per_part',128)*len(spectra): self.flush_spectra()

    def flush_spectra(self):
        if not self.spectra: return
        name=f'spectra-{len(self.parts):04d}.npz'; np.savez_compressed(self.output/name,**self.spectra)
        self.parts.append(name); self.files.append(name); self.spectra={}

    def flush_metrics(self):
        if not self.rows: return
        path=self.output/f'.metrics-{len(self.metric_parts):04d}.csv.gz'
        frame=pd.DataFrame(self.rows)
        frame=frame.astype({k:'float64' for k in frame.select_dtypes(include=np.floating)})
        frame.to_csv(path,index=False); self.metric_parts.append(path); self.rows=[]

    def finish(self):
        self.flush_spectra(); self.flush_metrics()
        frame=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in self.metric_parts],ignore_index=True) if self.metric_parts else pd.DataFrame()
        if frame.empty: raise ValueError('No eligible complete windows for offline package')
        # Preserve full float32 values when serializing: pandas otherwise prints
        # float32 columns with too few digits for exact CSV round-trip aggregation.
        frame=frame.astype({k:'float64' for k in frame.select_dtypes(include=np.floating)}).copy()
        for path in self.metric_parts: path.unlink()
        trial,seed,summary=aggregate_windows(frame); assoc=associations(trial)
        for name,table in [('windows',frame),('trials',trial),('seeds',seed),('summary',summary),('associations',assoc)]:
            table.to_csv(self.output/(name+'.csv.gz'),index=False); self.files.append(name+'.csv.gz')
        compact=summary[summary.metric.str.startswith(('ee.','base.','input.ee.axis1.'))]
        shared.write_json(self.output/'summary.json',dict(schema='b1z1-film-offline-package-v1',
            metric_subset=True,full_summary='summary.csv.gz',rows=compact.to_dict('records')))
        self.files.append('summary.json')
        selected=select_examples(frame,int(self.cfg.get('package_example_windows',24)))
        examples=[]; feature_samples=[]
        for path,run in self.runs.items():
            wanted=[s for s in selected if s['run']==path]
            if not wanted: continue
            specs={s['start_s']:dict(start=s['start_s'],end=s['end_s']) for s in wanted}
            cfg={**self.cfg,'_fields':FIELDS,'_window_specs':list(specs.values()),'trial_ids':sorted({s['trial_id'] for s in wanted})}
            for tid,spec,raw in film.stream_windows(run,cfg):
                found=[s for s in wanted if s['trial_id']==tid and s['start_s']==spec['start']]
                if not found: continue
                for segment in film.contiguous_segments(raw,run['meta']['dt'],REQUIRED):
                    if len(segment)!=round(self.cfg['comparison_window_seconds']/run['meta']['dt']): continue
                    d={k:np.stack([r[k] for r in segment]) for k in segment[0]}; meta=run['meta']; s=found[0]
                    _,arrays,_=analyze_window(d,meta,self.cfg,self.heads.get(meta['checkpoint_sha256'],(None,None))[0],self.cfg.get('feature_bases',{}).get(meta['checkpoint_sha256']))
                    # Save only bounded full features; no histories/observations/checkpoints.
                    for k in ('obs_t','history_t','privileged_t'): arrays.pop(k,None)
                    for k,x in list(arrays.items()):
                        if np.asarray(x).ndim>=1 and len(x)==len(d['time_seconds']) and np.issubdtype(np.asarray(x).dtype,np.number) and np.isfinite(x).all() and np.asarray(x).ndim<=2:
                            if k.startswith(('film_gamma','film_beta','feature_','action_','torque_','recorded_','base_')) and 'quaternion' not in k:
                                fit=detrended(x,d['time_seconds']); arrays[k+'__trend']=fit['trend']; arrays[k+'__residual']=fit['residual']
                    # Full feature pairs are stored only at the small stratified
                    # sample below; traces keep fixed projections, not dense tensors.
                    for k in ('film_features_pre_t','film_features_post_t'): arrays.pop(k,None)
                    name=f'example-{int(s["window_id"]):06d}.npz'; np.savez_compressed(self.output/name,**arrays)
                    self.files.append(name); examples.append(dict(s,file=name))
                    if 'film_features_pre_t' in d:
                        ids=np.linspace(0,len(segment)-1,min(self.cfg.get('feature_pairs_per_window',16),len(segment)),dtype=int)
                        feature_samples.append((s,d,ids))
                    for k,x in arrays.items(): self.dictionary.setdefault(k,dict(shape_tail=list(np.asarray(x).shape[1:]),dtype=str(np.asarray(x).dtype),units=array_units(k)))
        samples={}
        for s,d,ids in feature_samples:
            prefix=f'w{int(s["window_id"]):06d}__'
            for k in ('film_features_pre_t','film_features_post_t','time_seconds','step'):
                samples[prefix+k]=d[k][ids]
        if samples: np.savez_compressed(self.output/'paired_features.npz',**samples); self.files.append('paired_features.npz')
        priority={s['window_id']:i for i,s in enumerate(selected)}
        examples.sort(key=lambda s:priority[s['window_id']])
        # Copy the tiny plotting-only source, not the analysis framework/checkpoints.
        shutil.copyfile(Path(__file__).with_name('plot_exp2_film_package.py'),self.output/'plot_package.py'); self.files.append('plot_package.py')
        for name in ('stage_trials','stage_seeds','stage_summary'):
            source=self.output.parent/(name+'.csv')
            if source.exists(): shutil.copyfile(source,self.output/source.name); self.files.append(source.name)
        manifest=dict(schema='b1z1-film-offline-package-v1',complete=False,configuration=film.clean(self.cfg),
                      examples=examples,feature_bases=self.cfg.get('feature_bases',{}),spectra_parts=self.parts,
                      records=[dict(run=str(r['path']),checkpoint=r['meta']['checkpoint_sha256'],training_seed=r['meta'].get('checkpoint_training_seed'),
                                    evaluation_seed=r['meta']['evaluation_seed'],joint_names=r['meta']['dof_names'],
                                    environment=r['meta']['resolved_environment'],evaluation=r['meta']['evaluation'],
                                    records_sha256=film.file_hash(r['path']/'records.json'),metadata_sha256=film.file_hash(r['path']/'metadata.json')) for r in self.runs.values()],
                      dictionary=self.dictionary,files=self.files,
                      metric_units={k:metric_units(k) for k in frame.columns if k not in IDENTITY},
                      conventions=CONVENTIONS,display_names=shared.DISPLAY_NAMES if hasattr(shared,'DISPLAY_NAMES') else {})
        manifest['analysis_source_sha256']={name:film.file_hash(Path(__file__).with_name(name)) for name in
            ('exp2_film_package.py','exp2_task_diagnostics.py','exp3_film_oscillation.py','plot_exp2_film_package.py')}
        from b1z1_paper_names import DISPLAY_NAMES
        manifest['display_names']=DISPLAY_NAMES
        manifest['coverage']=[]
        for path,run in self.runs.items():
            requested=set(run['trial_ids'])
            if self.cfg.get('trial_ids') is not None: requested &= set(self.cfg['trial_ids'])
            observed=set(frame.loc[frame.run==path,'trial_id'])
            manifest['coverage'].append(dict(run=path,requested_trials=len(requested),trials_with_complete_windows=len(observed),
                                            trials_without_complete_windows=sorted(requested-observed)))
        shared.write_json(self.output/'manifest.json',manifest)
        (self.output/'README.md').write_text(README+f'\n## Coverage of this export\n\n{len(frame)} complete windows. '
            f'Evaluation seeds: {self.cfg["evaluation_seeds"]}; explicit trial filter: {self.cfg.get("trial_ids")}. '
            'See manifest coverage for requested trials without complete windows; their full-stage adherence remains in the stage CSVs.\n')
        from plot_exp2_film_package import rebuild
        rebuilt=rebuild(self.output,self.output/'rebuilt',self.cfg.get('package_plot_config',{}))
        if not np.allclose(summary['mean'],rebuilt['mean'],equal_nan=True): raise AssertionError('Package summary regeneration differs')
        manifest['complete']=True; shared.write_json(self.output/'manifest.json',manifest)
        (self.output/'COMPLETE').write_text('complete\n')
        archive=self.output.parent/'offline_package.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for name in [*self.files,'manifest.json','README.md','COMPLETE']:
                tar.add(self.output/name,arcname='offline_package/'+name)
            outputs=json.loads((self.output/'rebuilt'/'regeneration.json').read_text())['outputs']
            for name in [*outputs,'regeneration.json']:
                if name not in ('trials.csv','seeds.csv','summary.csv'):
                    tar.add(self.output/'rebuilt'/name,arcname='offline_package/rebuilt/'+name)
        return dict(directory=str(self.output),archive=str(archive),windows=len(frame),examples=len(examples))


CONVENTIONS = {
    'time':'Actor-time t in seconds; successor t+dt is never paired with a reset. Fixed complete post-transient windows only.',
    'errors':'EE Cartesian measured-reference in current yaw-aligned frame [m]. Base body-frame measured-command [m/s,m/s,rad/s]. FiLM uses command-estimated state with recorded observation scales; measured comparison uses identical coordinates.',
    'detrending':'Componentwise OLS linear trend, mean(t) origin. bias=mean(x); residual=x-trend. No event/gap/reset crossing. Stage adherence in stage_trials uses prescribed duration, failed tails zero; window adherence covers available complete windows only.',
    'action':'Original/fixed/bypass head evaluations on identical recorded pre-features; fixed uses interval-mean gamma/beta. Position is default-relative rad, FF Nm after configured scales. Never new executed trajectories.',
    'torque':'PD+FF after gain/branch/motor scaling equals preclip request [Nm]. Actual recorded clipping bounds used. Current-state requests separate from source-ID-delayed start-state requests and recorded interval/substep execution. Delay sources outside window unavailable.',
    'coupling':'Detrended variance(total)=variance(PD)+variance(FF)+2*covariance. Positive covariance reinforces; negative cancels. Not a unique physical compensation decomposition.',
    'motion':'Exact finite endpoint identity in WORLD m/s: base translation + base rotation + motion relative to torso - reference velocity. Torso-relative term is not isolated actuator causation.',
    'sensitivity':'Original saved MLP only. Pre-features/state fixed; central differences in original normalized inputs. Coordinate derivatives per normalized unit; magnitude per fractional change; direction per rad; freeze is finite difference to interval mean. No new encoder or modified policy.',
    'spectra':'Linear detrend per component then Hann Welch, no additional segment detrend. PSD signal-unit^2/Hz; CSD conjugate(X)*Y, real+imag in NPZ; phase radians only above coherence threshold and usable band. Positive signal lag is not causality.',
    'aggregation':'Windows within trial, trials within evaluation seed, equal evaluation-seed mean/sample SD (ddof=1). Checkpoints never pooled. Evaluation seeds are not independently trained policies. Per-metric counts retained; missing never zero.',
    'examples':'Median/10th/90th percentile trial vertical RMS per checkpoint/task/load/motion stage, nearest actual window. Deterministic coverage balancing over task/profile, load/motion, amplitude level and checkpoint. All population window metrics/spectra retained; only traces/full-feature pairs bounded.'}
README = '''# Offline FiLM/Exp2 diagnostic package

Run `python plot_package.py . --output rebuilt` after installing numpy, pandas,
scipy, matplotlib. No repository, checkpoint, PyTorch, CUDA, or simulator is needed.
Use `--config plot_settings.json` to override fonts, colors, labels, figure sizes,
example window IDs and max_trace_figures. Times New Roman is preferred; DejaVu Serif
is the recorded fallback. Figures are PDF/PNG; tables CSV/LaTeX.

`manifest.json` defines timing, frames, units, gains, ordering, selection and provenance.
`windows/trials/seeds/summary.csv.gz` retain unrounded statistics and counts; stage
CSVs retain full prescribed-duration adherence. `associations.csv.gz` contains
within-stage, within-seed trial correlations, including weak/undefined results.
`spectra-*.npz` contains each window's frequency, PSD, complex cross spectra,
coherence and masked phase. `example-*.npz` contains bounded aligned recorded and
offline signals, trends and residuals. `paired_features.npz` retains traceable pairs.
Missing diagnostics remain NaN in CSV/NPZ and null in JSON with recorded reasons.

All action probes are offline evaluations of saved heads/conditioning on recorded
features, not executed policies. Combined requests and delayed start-state estimates
are not substep execution. No new encoding is sent to old weights. The package does
not identify a unique causal feedback or compensation mechanism.
'''
