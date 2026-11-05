"""URDF FK and EE-only sphere guidance against torso/link primitives.

Not full-arm collision protection: only pairs containing the EE are evaluated.
"""
import math
import xml.etree.ElementTree as ET

import torch
from .ee_collision import point_segment, segment_segment, point_box, segment_box


def skew(v):
    x,y,z=v.unbind(-1); o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)


def rotation(axis,angle):
    k=skew(axis)
    return torch.eye(3,device=axis.device,dtype=axis.dtype)+angle.sin()[...,None,None]*k+(1-angle.cos())[...,None,None]*(k@k)


class RobotGeometry:
    def __init__(self,path,joint_names,arm_indices,base_name,ee_name,feet,cfg,device):
        self.cfg=cfg
        self.names=list(joint_names)
        self.arm_ids=torch.as_tensor(arm_indices,device=device,dtype=torch.long)
        self.arm_names=[self.names[i] for i in self.arm_ids.tolist()]
        self.base,self.ee,self.feet=base_name,ee_name,list(feet)
        root=ET.parse(path).getroot()
        self.joints=[]
        children={j.find('child').get('link'):j for j in root.findall('joint')}
        needed=set()
        roots={j.find('parent').get('link') for j in root.findall('joint')}-set(children)
        if len(roots)!=1:raise ValueError('Compliance requires one URDF root')
        self.root=roots.pop()
        for link in [base_name,ee_name]+list(feet):
            while link != self.root:
                j=children[link]; needed.add(j.get('name')); link=j.find('parent').get('link')
        known={self.root}
        remaining=[j for j in root.findall('joint') if j.get('name') in needed]
        self.parents={}
        self.arm_links=[]
        tensor=lambda a:torch.tensor(a,device=device,dtype=torch.float32)
        while remaining:
            ready=[j for j in remaining if j.find('parent').get('link') in known]
            if not ready: raise ValueError('Disconnected compliance URDF chain')
            for j in ready:
                origin=j.find('origin')
                xyz=tensor([float(x) for x in origin.get('xyz','0 0 0').split()] if origin is not None else [0,0,0])
                rpy=[float(x) for x in origin.get('rpy','0 0 0').split()] if origin is not None else [0,0,0]
                r=torch.eye(3,device=device)
                for a,angle in reversed(list(zip(torch.eye(3,device=device),rpy))):
                    r=r@rotation(a,tensor(angle))
                axis=j.find('axis')
                a=tensor([float(x) for x in axis.get('xyz').split()] if axis is not None else [1,0,0])
                a=a/a.norm().clamp_min(1e-8)
                parent,child=j.find('parent').get('link'),j.find('child').get('link')
                name=j.get('name'); kind=j.get('type')
                if kind != 'fixed' and name not in self.names:
                    raise ValueError('Missing compliance joint state: '+name)
                index=self.names.index(name) if name in self.names else None
                self.joints.append((name,parent,child,xyz,r,a,index,kind))
                self.parents[child]=parent
                if name in self.arm_names:self.arm_links.append(child)
                known.add(child); remaining.remove(j)
        # Use named foot chains, excluding torso and proximal hip mounting links.
        self.leg_segments=[]
        for foot in feet:
            chain=[foot]
            while chain[-1] != base_name:chain.append(self.parents[chain[-1]])
            chain.reverse()
            self.leg_segments.extend(zip(chain[1:-1],chain[2:]))
        trunk=root.find(f"link[@name='{base_name}']/collision")
        box=trunk.find('geometry/box')
        if box is None:raise ValueError('Compliance requires an explicit torso collision box')
        self.half_size=tensor([float(x)/2 for x in box.get('size').split()])
        o=trunk.find('origin')
        self.box_center=tensor([float(x) for x in o.get('xyz','0 0 0').split()] if o is not None else [0,0,0])
        if o is not None and any(float(x)!=0 for x in o.get('rpy','0 0 0').split()):
            raise ValueError('Rotated torso collision boxes need an explicit transform')
        # Compile fixed mounts once. Probes change only arm joints, so they
        # need neither leg FK nor Jacobians. Keep forward() as a parity oracle.
        by_child={j[2]:j for j in self.joints}
        def chain(link):
            result=[]
            while link!=self.root:
                result.append(by_child[link]);link=by_child[link][1]
            return list(reversed(result))
        p=tensor([0,0,0]);r=torch.eye(3,device=device)
        for name,_,_,offset,fixed,_,_,kind in chain(base_name):
            if kind!='fixed':raise ValueError('Compliance torso mount must be fixed')
            p=p+r@offset;r=r@fixed
        p=-(r.T@p);r=r.T
        self.arm_steps=[];order=[]
        for name,_,_,offset,fixed,a,_,kind in chain(ee_name):
            p=p+r@offset;r=r@fixed
            if kind!='fixed':
                if name not in self.arm_names or kind not in ('revolute','continuous'):
                    raise ValueError('Compliance EE chain must contain only configured revolute arm joints')
                k=skew(a)
                self.arm_steps.append((p,r,k,k@k,self.arm_names.index(name)))
                order.append(name);p=tensor([0,0,0]);r=torch.eye(3,device=device)
        if order!=self.arm_names:raise ValueError('Arm indices must follow URDF chain order')
        self.arm_tail=p
        excluded=set(cfg.ee_excluded_arm_segments)
        if any(i<0 or i>=len(self.arm_names) for i in excluded):
            raise ValueError('Invalid EE attachment-neighbour segment exclusion')
        # Segments 4/5 are the wrist attachment and own EE segment on Z1.
        # Exclusions are explicit, not inferred from distances or current pose.
        self.ee_obstacle_segments=[i for i in range(len(self.arm_names)) if i not in excluded]
        if cfg.compile_geometry and torch.device(device).type=='cuda':
            # No reference state, simulator calls or RNG enters these kernels.
            # Dynamic batch dimensions share code across bounded probe batches.
            self.forward=torch.compile(self.forward,dynamic=True,fullgraph=True)
            self._probe_clearance=torch.compile(self._probe_clearance,dynamic=True,fullgraph=True)
            self.path_clearance=torch.compile(self.path_clearance,dynamic=True,fullgraph=True)
            self.point_clearance=torch.compile(self.point_clearance,dynamic=True,fullgraph=True)

    def forward(self,q):
        n=len(q); eye=torch.eye(3,device=q.device,dtype=q.dtype).expand(n,3,3)
        pos={self.root:q.new_zeros(n,3)}; rot={self.root:eye}; origins={}; axes={}
        for name,parent,child,offset,r,a,index,kind in self.joints:
            p=pos[parent]+torch.einsum('bij,j->bi',rot[parent],offset)
            rr=rot[parent]@r
            origins[name]=p; axes[name]=torch.einsum('bij,j->bi',rr,a)
            if kind in ('revolute','continuous'):rr=rr@rotation(a,q[:,index])
            elif kind=='prismatic':p=p+axes[name]*q[:,index,None]
            pos[child],rot[child]=p,rr
        arm=torch.stack([origins[name] for name in self.arm_names]+[pos[self.ee]],1)
        jac=torch.stack([torch.cross(axes[name],pos[self.ee]-origins[name],dim=-1) for name in self.arm_names],-1)
        legs=torch.stack([torch.stack((pos[a],pos[b]),1) for a,b in self.leg_segments],1)
        # Z1 and trunk are siblings under the fixed URDF root. Express every
        # output in the measured torso frame, including nonidentity fixed mounts.
        inverse=rot[self.base].transpose(1,2)
        ee=torch.einsum('bij,bj->bi',inverse,pos[self.ee]-pos[self.base])
        arm=torch.einsum('bij,bkj->bki',inverse,arm-pos[self.base][:,None])
        legs=torch.einsum('bij,bklj->bkli',inverse,legs-pos[self.base][:,None,None])
        return ee,inverse@jac,arm,legs

    def arm_points(self,q):
        eye=torch.eye(3,device=q.device,dtype=q.dtype)
        p=q.new_zeros(len(q),3);r=eye.expand(len(q),3,3);points=[]
        for offset,fixed,k,k2,index in self.arm_steps:
            p=p+torch.einsum('bij,j->bi',r,offset);points.append(p)
            rr=eye+q[:,index].sin()[:,None,None]*k+(1-q[:,index].cos())[:,None,None]*k2
            r=r@fixed@rr
        points.append(p+torch.einsum('bij,j->bi',r,self.arm_tail))
        return torch.stack(points,1)

    def clearance(self,q,full_q,legs=None):
        return self.clearance_pairs(q,full_q,legs).amin(-1)

    def clearance_pairs(self,q,full_q,legs=None):
        shape=q.shape[:-1]
        if legs is None:legs=self.forward(full_q)[3]
        legs=legs.reshape(len(full_q),*([1]*(q.ndim-2)),*legs.shape[1:]).expand(*shape,*legs.shape[1:])
        # Probe count grows with environments; bound geometry scratch memory while
        # keeping all per-contact arithmetic on device (no per-robot CPU loop).
        flat=q.reshape(-1,q.shape[-1]);legs=legs.reshape(len(flat),*legs.shape[-3:])
        size=self.cfg.collision_batch_size
        return torch.cat([self._probe_clearance(part,leg)
                          for part,leg in zip(flat.split(size),legs.split(size))]).reshape(*shape,-1)

    def _probe_clearance(self,q,legs):
        return self._clearance_points(self.arm_points(q),legs)

    def _clearance_flat(self,full):
        # Unoptimized reference path retained for focused numerical-parity tests.
        _,_,arm,legs=self.forward(full)
        return self._clearance_points(arm,legs).amin(-1)

    def _clearance_points(self,arm,legs):
        return self.point_clearance(arm[...,-1,:],arm,legs)

    def point_clearance(self,ee,arm,legs):
        box=point_box(ee,self.box_center,self.half_size)-self.cfg.ee_radius
        leg=point_segment(ee[...,None,:],legs[...,0,:],legs[...,1,:])-self.cfg.ee_radius-self.cfg.leg_radius
        ids=self.ee_obstacle_segments
        other=point_segment(ee[...,None,:],arm[...,:-1,:][...,ids,:],arm[...,1:,:][...,ids,:])-self.cfg.ee_radius-self.cfg.arm_radius
        return torch.cat((box[...,None],leg,other),-1)

    def path_clearance(self,start,end,arm,legs):
        """Swept EE sphere over the full requested Cartesian segment.

        Obstacle primitives are frozen at the supplied probe/measurement pose;
        configuration probes separately check their motion. No IK certificate.
        """
        box=segment_box(start,end,self.box_center,self.half_size)-self.cfg.ee_radius
        ids=self.ee_obstacle_segments
        leg=segment_segment(start[...,None,:],end[...,None,:],legs[...,0,:],legs[...,1,:])-self.cfg.ee_radius-self.cfg.leg_radius
        other=segment_segment(start[...,None,:],end[...,None,:],arm[...,:-1,:][...,ids,:],arm[...,1:,:][...,ids,:])-self.cfg.ee_radius-self.cfg.arm_radius
        swept=torch.cat((box[...,None],leg,other),-1)
        return torch.minimum(swept,torch.minimum(self.point_clearance(start,arm,legs),self.point_clearance(end,arm,legs)))
