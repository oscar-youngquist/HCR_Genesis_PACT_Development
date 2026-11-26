"""Opt-in nominal, flat-ground sanity scenarios, isolated to the evaluator."""
import json
import numpy as np
import torch


def configure(cfg):
    from legged_gym.utils.terrain import Terrain
    original=Terrain.make_terrain
    # Retain the supported mesh/grid pipeline, but build every cell as slope=0.
    Terrain.make_terrain=lambda self,choice,difficulty:original(self,0.,0.)
    for key in vars(cfg.domain_rand):
        if key.startswith('randomize_') and key!='randomize_friction':setattr(cfg.domain_rand,key,False)
    for key in ('randomize_base_mass','randomize_com_displacement','randomize_joint_armature',
                'randomize_joint_friction','randomize_joint_damping','randomize_joint_stiffness',
                'randomize_pd_gain','randomize_motor_strength','randomize_ctrl_delay'):
        setattr(cfg.domain_rand,key,False)
    cfg.noise.add_noise=False
    cfg.commands.heading_command=False  # A zero yaw command must mean no turn.


def nominal_properties(env):
    s=env.simulator;r=s._robot;ids=torch.arange(env.num_envs,device=env.device)
    zeros=torch.zeros(env.num_envs,12,device=env.device)
    r.write_joint_armature_to_sim(zeros,s._dof_indices,ids)
    r.write_joint_friction_coefficient_to_sim(zeros,zeros,zeros,s._dof_indices,ids)
    r.write_joint_damping_to_sim(zeros,s._dof_indices,ids)
    for key in ('_joint_armature','_joint_friction','_joint_damping','_joint_stiffness','_added_base_mass','_base_com_bias'):
        getattr(s,key).zero_()
    s.push_robots=lambda:None


def audit_application(s):
    """Startup buffer checks, no physics advance; restore flags/offsets and clear."""
    from scripts.hard_pact_shadow_estimation import rotate
    c=s._robot.permanent_wrench_composer;cfg=s._cfg.domain_rand
    enabled=cfg.persistent_wrench_at_com;bias=s._base_com_bias.clone()
    q=s._robot.data.body_link_quat_w[:,s._base_link_index][:,(1,2,3,0)]
    n=len(q);probe=q.new_zeros(n,6);errors=[]
    read=lambda:torch.cat((rotate(q,c.composed_force_as_torch[:,s._base_link_index]),
                          rotate(q,c.composed_torque_as_torch[:,s._base_link_index])),-1)
    try:
        # Demonstrate the installed kernel bug directly, separately from adapter.
        probe[:,3:]=.5;c.reset()
        c.set_forces_and_torques(forces=probe[:,:3,None].transpose(1,2),torques=probe[:,3:,None].transpose(1,2),
            positions=s._robot.data.body_link_pos_w[:,s._base_link_index,None,:],body_ids=[s._base_link_index],is_global=True)
        kernel_error=(read()-probe).abs().amax()
        for mode in (False,True):
            cfg.persistent_wrench_at_com=mode
            for offset in (0.,1.):
                s._base_com_bias.copy_(bias+offset*bias.new_tensor([.03,-.02,.04]))
                arm=rotate(q,s._default_com_pos+s._base_com_bias)
                cases=[q.new_tensor([3.,-4.,5.,-.5,.5,-.5])]
                cases += [sign*torch.eye(6,device=q.device)[j]*(5 if j<3 else .5) for j in range(6) for sign in (-1,1)]
                for case_index,value in enumerate(cases):
                    probe.copy_(value);expected=probe.clone()
                    if mode:expected[:,3:]+=torch.cross(arm,probe[:,:3],dim=-1)
                    for _ in range(2):
                        s.hard_pact_apply_base_wrench_world(probe)
                        errors.append((read()-expected).abs().amax())
                    s.hard_pact_apply_base_wrench_world(torch.zeros_like(probe));errors.append(read().abs().amax())
                    if case_index==0:
                        # Installed deprecated wrapper delegates to the composer;
                        # test its no-position/world-frame contract separately.
                        s._robot.set_external_force_and_torque(expected[:,:3,None].transpose(1,2),
                            expected[:,3:,None].transpose(1,2),body_ids=[s._base_link_index],is_global=True)
                        errors.append((read()-expected).abs().amax())
    finally:
        cfg.persistent_wrench_at_com=enabled;s._base_com_bias.copy_(bias)
        s.hard_pact_apply_base_wrench_world(torch.zeros_like(probe))
    maximum=float(torch.stack(errors).max());assert maximum<2e-5,maximum
    return dict(semantics=s.hard_pact_wrench_application_semantics,max_abs_error=maximum,tolerance=2e-5,
        installed_position_kernel_error=float(kernel_error),checks=len(errors),physics_advanced=False,
        cases='Both modes, nominal/random offsets, live orientation, signed axes, combined loads, repeated replacement, zero clearing; installed deprecated wrapper checked separately')


def install(o,mode):
    t=o.tapes
    if getattr(o.s,'hard_pact_wrench_application_semantics',None):
        o.manifest['production_wrench_audit']=audit_application(o.s)
    if o.args.momentum_sanity_correct_wrench:
        # Probe the installed API without stepping physics, then restore its buffers.
        from scripts.hard_pact_shadow_estimation import rotate
        probe=torch.zeros(o.e.num_envs,6,device=o.e.device)
        probe[:6]=torch.diag(probe.new_tensor([5,5,5,.5,.5,.5]))
        s=o.s;c=s._robot.permanent_wrench_composer;s.hard_pact_apply_base_wrench_world(probe)
        q=s._robot.data.body_link_quat_w[:,s._base_link_index][:,(1,2,3,0)]
        delivered=torch.cat((rotate(q,c.composed_force_as_torch[:,s._base_link_index]),
                             rotate(q,c.composed_torque_as_torch[:,s._base_link_index])),-1)
        o.manifest['composer_startup_audit']=dict(requested=probe.cpu().tolist(),
            expected_link_wrench=s.hard_pact_wrench_at_base_origin_world(probe).cpu().tolist(),
            actual_composer_link_wrench=delivered.cpu().tolist(),physics_advanced=False)
        s.hard_pact_apply_base_wrench_world(torch.zeros_like(probe))
        def corrected_load(wrench):
            label=s.hard_pact_wrench_at_base_origin_world(wrench)
            c.reset();c.set_forces_and_torques(forces=label[:,:3,None].transpose(1,2).contiguous(),
                torques=label[:,3:,None].transpose(1,2).contiguous(),body_ids=[s._base_link_index],is_global=True)
        s.hard_pact_apply_base_wrench_world=corrected_load
    t.commands.zero_()
    if mode=='walking':t.commands[:,:,0]=.5
    # Fixed world-axis pulses: zero-load env0 and independently excited axes.
    t.light.zero_();t.heavy.zero_();t.active.zero_()
    for env in range(1,o.e.num_envs):
        axis=(env-1)%6;amplitude=5. if axis<3 else .5
        if env>6:amplitude=-amplitude
        t.light[100:200,env,axis]=amplitude;t.active[100:200,env]=True
    t.wrench=t.light
    t.before(0);t.wrench_update(None)
    for row in o.manifest['target_plan']:row.update(family='flat_sanity',level=0)
    o.manifest['momentum_sanity']=dict(mode=mode,command_x=.5 if mode=='walking' else 0.,
        pulse_steps=[100,200],force_n=5.,moment_nm=.5,zero_load_environment=0,
        terrain='All generated cells slope=0, no roughness; nominal labels retained only as grid indices.',
        noise=False,instantaneous_pushes=False,passive_joint_parameters=0.,armature=0.,
        corrected_load_delivery=bool(o.args.momentum_sanity_correct_wrench))
    r=o.s._robot;view=r.root_physx_view
    readback=dict(body_names=list(r.body_names),joint_names=list(r.joint_names))
    for key,method in [('masses','get_masses'),('coms','get_coms'),('inertias','get_inertias'),
                       ('armatures','get_dof_armatures'),('friction','get_dof_friction_coefficients'),('damping','get_dof_dampings')]:
        fn=getattr(view,method,None)
        readback[key]=fn().cpu().tolist() if fn is not None else None
    o.manifest['nominal_property_readback']=readback
    from scripts.eval_hard_pact_frozen import tensor_hash
    t.mechanics=tensor_hash(o.e._canonical_randomized_parameters())
    digest=lambda x:tensor_hash({'value':x})
    o.manifest['tapes'].update(commands=digest(t.commands),light_wrench=digest(t.light),
        heavy_wrench=digest(t.heavy),applied=digest(t.wrench),mechanics=t.mechanics,
        semantics='Sanity-only zero/known world-axis pulses; absolute-time commands; no random pushes.')
    np.savez_compressed(o.out/'tapes.npz',commands=t.commands.cpu().numpy(),light=t.light.cpu().numpy(),heavy=t.heavy.cpu().numpy())
