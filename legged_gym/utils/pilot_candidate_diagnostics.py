"""Streaming distributions, not averages of batch quantiles or signed means."""
import csv
import json
from pathlib import Path
import numpy as np


class Distributions:
    # Histograms retain every eligible sample. Quantiles are explicitly bounded
    # bin estimates (width 0.001 m for offsets), never averaged batch quantiles.
    def __init__(self,output,dt):
        self.output=Path(output);self.dt=dt;self.groups={};self.pending=[]

    def add(self,data,valid,profile,group_prefix=''):
        keys=['raw_offset','bounded_offset','filtered_offset','accepted_offset',
              'workspace_reduction','geometry_reduction','filter_difference','final_reduction',
              'applied_force_task','recruited_force_task','recruited_wrench',
              'direct_torso_wrench','combined_wrench','base_reference','base_measured',
              'requested_base_correction','requested_base_total','base_clipped',
              'gates','profile_gate','workspace_gate','joint_gate','collision_gate','directional_progress','gait_phase']
        force=data['applied_force_task'];coeff=data['reference_stage_selected_coefficients']
        active=force.abs()>1e-6
        masks={'all':valid,'yielding':valid&(active&(coeff>0)).any(-1),
               'protected':valid&(active&(coeff==0)).any(-1),
               'recruitment':valid&(data['recruited_force_task'].abs()>1e-6).any(-1),
               'restricted':valid&(data['workspace_gate'].amax(-1)>.5),
               'ample':valid&(data['workspace_gate'].amax(-1)<=.5)}
        for axis,name in enumerate(('x','y','z')):
            for sign,label in ((1,'positive'),(-1,'negative')):
                present=valid&(force[:,axis]*sign>1e-6)
                masks[f'{name}_{label}_yielding']=present&(coeff[:,axis]>0)
                masks[f'{name}_{label}_protected']=present&(coeff[:,axis]==0)
        masks['strong_profile']=valid&(coeff.amax(-1)>=.8)
        if group_prefix:masks={group_prefix+'/'+k:v for k,v in masks.items()}
        # Bounded by one rollout, transferred once per field at the PPO boundary.
        self.pending.append(({k:data[k] for k in keys if k in data},masks))

    def flush(self,iteration,write=True):
        import torch
        if not self.pending and not write:return
        arrays={k:torch.cat([d[k] for d,m in self.pending]).float().cpu().numpy()
                for k in self.pending[0][0]} if self.pending else {}
        masks={k:torch.cat([m[k] for d,m in self.pending]).cpu().numpy()
               for k in self.pending[0][1]} if self.pending else {}
        self.pending.clear()
        for key,a in arrays.items():
            signals={f'{key}/{i}':np.abs(a[:,i]) for i in range(a.shape[-1])}
            if 'wrench' in key or key in ('base_reference','base_measured','requested_base_correction','requested_base_total'):
                # Do not combine N with Nm, or m/s with rad/s, in one norm.
                signals[key+'/planar_norm']=np.linalg.norm(a[:,:2],axis=-1)
            else:signals[key+'/norm']=np.linalg.norm(a,axis=-1)
            high=1.024 if any(x in key for x in ('offset','reduction','difference')) else 256.
            for signal,values in signals.items():
                edges=np.linspace(0,high,1025)
                for group,mask in masks.items():
                    v=values[mask & np.isfinite(values)]
                    if not len(v):continue
                    identity=(group,signal)
                    if identity not in self.groups:
                        self.groups[identity]=dict(edges=edges,hist=np.zeros(1024,dtype=np.int64),count=0,total=0.,maximum=None,overflow=0,above=np.zeros(3,dtype=np.int64))
                    s=self.groups[identity]
                    s['hist']+=np.histogram(v,edges)[0];s['count']+=len(v)
                    s['total']+=v.sum(dtype=np.float64);s['overflow']+=int((v>high).sum())
                    s['above']+=np.array([(v>=x-1e-6).sum() for x in (.15,.20,.25)])
                    if len(v):s['maximum']=max(s['maximum'] or 0.,float(v.max()))
        if not write:return
        rows=[];histograms={}
        for index,((group,signal),s) in enumerate(self.groups.items()):
            n=s['count'];cum=np.cumsum(s['hist'])
            row=dict(group=group,signal=signal,count=n,observed_seconds=n*self.dt,
                     mean=s['total']/n if n else None,maximum=s['maximum'],overflow=s['overflow'])
            for q in (.5,.9,.95,.99):
                b=np.searchsorted(cum,q*n)
                row[f'p{int(q*100)}_lower']=float(s['edges'][b]) if n and b<1024 else None
                row[f'p{int(q*100)}_upper']=float(s['edges'][b+1]) if n and b<1024 else None
            for threshold,count in zip((15,20,25),s['above']):row[f'fraction_ge_{threshold}cm']=float(count/n) if n and 'offset' in signal else None
            row['histogram_id']=index
            row['edges_id']='offset_edges' if s['edges'][-1]==1.024 else 'general_edges'
            rows.append(row)
            # Shared bins are stored once, not duplicated for every signal/group.
            histograms[row['edges_id']]=s['edges'];histograms[f'counts_{index}']=s['hist']
        if not rows:return
        with (self.output/'distributions.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        np.savez_compressed(self.output/'distribution_histograms.npz',**histograms)
        (self.output/'distribution_metadata.json').write_text(json.dumps(dict(
            through_iteration=iteration,quantiles='lower/upper histogram bounds, not exact quantiles',
            duration='sum of valid robot-control intervals, not wall time',
            signs='absolute component and Euclidean norm distributions',
            threshold='>= with 1e-6 m numerical tolerance; offsets only',
            masks='finite pre-failure; yielding/protected use actual force and sign-selected coefficient; restricted uses workspace gate >0.5',
            units='offset/reduction/difference m; force N; wrench [N,N,Nm]; base [m/s,m/s,rad/s]; gates/phase dimensionless'),indent=2))
