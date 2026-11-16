"""CPU-only, reproducible Oct 7 export and historical-source parity checks."""
import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from standalone import Deployment

COMMIT = '41d5b7b7b5ebbcef3d760bf975d1275ebb7410e4'
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-repo', type=Path, default=ROOT.parent/'HCR_HardPact')
    parser.add_argument('--run', type=Path, default=ROOT/'logs/hardpact_iclr/go2_pact_rough/Oct07_10-22-08_hard_pact_full_isaaclab')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    torch.set_num_threads(2)
    torch.manual_seed(20000)
    archive = out/'archive'
    files = ['rsl_rl/algorithms/'+n for n in ('hard_pact_qp.py','hard_pact_qp_backends.py',
        'hard_pact_qp_curriculum.py','hard_pact_qp_diagnostics.py','hard_pact_qp_capture.py','ppo_hard_pact.py')]
    files += ['rsl_rl/modules/'+n for n in ('actor_critic_hard_pact.py','hard_pact_physics.py','hard_pact_control.py')]
    files += ['legged_gym/envs/go2/go2_hard_pact/'+n for n in ('go2_hard_pact.py','go2_hard_pact_config.py','deployment.py')]
    files += ['legged_gym/dynamics/bard_go2_dynamics.py','legged_gym/dynamics/pinocchio_go2_dynamics.py',
        'legged_gym/envs/go2/go2_pact/go2_pact.py','legged_gym/simulator/isaaclab_simulator_pact.py',
        'legged_gym/simulator/isaaclab_simulator.py','resources/robots/go2/go2_lab_cfg.py',
        'rsl_rl/runners/pact_runner.py','resources/robots/go2/urdf/go2.urdf']
    for rel in files:
        dest = archive/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.check_output(['git','-C',str(args.source_repo),'show',COMMIT+':'+rel]))
    for name in ('hard_pact_resolved_config.json','hard_pact_deployment_contract.json','go2_hard_pact.py','go2_hard_pact_config.py'):
        dest = archive/'run'/name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes((args.run/name).read_bytes())
    for name in ('go2_hard_pact.py','go2_hard_pact_config.py'):
        assert (archive/'run'/name).read_bytes() == (archive/'legged_gym/envs/go2/go2_hard_pact'/name).read_bytes()
    resolved = json.loads((args.run/'hard_pact_resolved_config.json').read_text())
    env = resolved['environment']['resolved']; train = resolved['training']['resolved']
    contract = json.loads((args.run/'hard_pact_deployment_contract.json').read_text())
    cfg = train['algorithm']['hard_pact_qp']; policy = train['policy']
    ckpt = args.run/'model_20000.pt'
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert state['iter']==20000 and state['hard_pact_qp_curriculum']['progress']==1
    sys.path.insert(0,str(args.source_repo))
    module('rsl_rl.modules.hard_pact_physics',archive/'rsl_rl/modules/hard_pact_physics.py')
    ac = module('rsl_rl.modules._oct07_actor',archive/'rsl_rl/modules/actor_critic_hard_pact.py')
    weights = state['model_state_dict']
    with contextlib.redirect_stdout(io.StringIO()):
        actor = ac.ActorCritic_HardPACT(num_actor_obs=57,
            num_critic_obs=env['env']['num_privileged_obs']*env['env']['num_priv_stack'],num_actions=12,
            actor_layers=policy['actor_layers'],critic_layers=policy['critic_layers'],cenet_in_dim=570,
            cenet_latent_dim=16,cenet_velo_dim=12,cenet_enc_layers=policy['cenet_enc_layers'],activation='elu',
            cenet_explicit_layers=policy['cenet_explicit_layers'],grf_decoder_layers=policy['grf_decoder_layers'],
            wrench_decoder_layers=policy['wrench_decoder_layers'],contact_epsilon=policy['contact_epsilon'],
            grf_scale_n=weights['physics_estimator.grf_scale_n'],wrench_scale=weights['physics_estimator.wrench_scale'],
            wrench_qp_clip=weights['physics_estimator.wrench_qp_clip'])
    actor.load_state_dict(weights,strict=True)
    actor.eval()
    portable = Deployment(actor,env,cfg).eval()
    for parameter in portable.parameters(): parameter.requires_grad_(False)
    scripted = torch.jit.script(portable)
    scripted.save(str(out/'hardpact_oct07_20000.pt'))
    loaded = torch.jit.load(str(out/'hardpact_oct07_20000.pt')).eval()
    reference = module('rsl_rl.algorithms._oct07_qp',archive/'rsl_rl/algorithms/hard_pact_qp.py')
    control = module('_oct07_control',archive/'rsl_rl/modules/hard_pact_control.py')
    qcfg = reference.HardPACTQPConfig(**contract['qp_update']['solver_and_objective_settings'])
    joints = {j.attrib['name']:j.find('limit').attrib for j in ET.parse(archive/'resources/robots/go2/urdf/go2.urdf').getroot().findall('joint') if j.find('limit') is not None}
    names = env['asset']['dof_names']
    limits = {key:torch.tensor([float(joints[n][attr]) for n in names]) for key,attr in
        [('torque_limits','effort'),('position_lower','lower'),('position_upper','upper')]}
    limits['velocity_limits']=torch.tensor(env['asset']['dof_vel_limits'])
    # Prefer exact backend-realized values saved with the run, including its
    # float32 URDF conversion rounding; cross-check rather than substitute URDF.
    realized_limits = {k:torch.tensor(v) for k,v in contract['qp_update']['physical_joint_limits'].items()}
    for key in limits:
        torch.testing.assert_close(limits[key],realized_limits[key],atol=1e-5,rtol=1e-6)
    limits = realized_limits
    qp = reference.HardPACTDifferentiableQP(qcfg,limits['torque_limits'],limits['position_lower'],limits['position_upper'],limits['velocity_limits'])
    checks={}
    def check(name,a,b,atol=2e-5,rtol=2e-5):
        torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        checks[name]={'max_absolute_error':float((a-b).abs().max()) if a.numel() else 0.,'atol':atol,'rtol':rtol}
    fixtures={}
    with torch.inference_mode():
        for batch in (1,8):
            obs=torch.randn(batch,57)*.1;history=torch.randn(batch,570)*.1
            action,z,explicit=loaded(obs,history)
            check(f'actor_{batch}',action,actor.act_inference(obs,history),0,0)
            check(f'latent_{batch}',z,actor.cenet_z,0,0)
            check(f'explicit_{batch}',explicit,actor.cenet_torso_velo,0,0)
            tau=torch.randn(batch,12)*5
            grf,raw,clipped=loaded.physics(z,explicit,tau)
            heads=actor.physics_estimator
            check(f'grf_{batch}',grf,heads.grf_to_physical(heads.predict_grf(z,explicit,tau)).reshape(batch,4,3),0,0)
            check(f'wrench_{batch}',raw,heads.wrench_to_physical(heads.predict_wrench(z,explicit)),0,0)
            R=torch.randn(batch,18,18);M=R@R.transpose(1,2)+torch.eye(18)*3
            quat=torch.randn(batch,4);quat=quat/quat.norm(dim=-1,keepdim=True)
            d={'tau_nom':tau,'mass_matrix':M,'bias':torch.randn(batch,18),'foot_jacobians':torch.randn(batch,4,3,18)*.1,
                'base_jacobian':torch.randn(batch,6,18)*.1,'force_pred_world':torch.randn(batch,4,3)*20,
                'wrench_pred_world':torch.randn(batch,6),'contact_probability':torch.rand(batch,4),
                'foot_acceleration_bias':torch.randn(batch,4,3),'base_quaternion':quat,
                'base_angular_velocity_world':torch.randn(batch,3),'base_linear_velocity_world':torch.randn(batch,3),
                'velocity_command':torch.randn(batch,3),'estimated_height':torch.full((batch,1),.38),
                'estimated_base_linear_velocity_world':torch.randn(batch,3),'dt':torch.full((batch,1),.005),
                'joint_position':(limits['position_lower']+limits['position_upper']).expand(batch,-1)/2,
                'joint_velocity':torch.randn(batch,12),'previous_torque':torch.randn(batch,12),
                'effective_kp':torch.rand(batch,12)*30,'effective_kd':torch.rand(batch,12),**limits}
            m=loaded.primary(d);ref=qp._build(d)
            for stage,actual,expected in [('primary',m,ref),('recovery',loaded.recovery(m),qp._soft_joint_problem(ref))]:
                for key in ('Q','p','G','h','physical_G','physical_h','variable_scale','acceleration_map','acceleration_offset'):
                    check(f'{stage}_{key}_{batch}',actual[key],getattr(expected,key))
                for key in ('native_lower','native_upper'):
                    torch.testing.assert_close(actual[key],getattr(expected,key),atol=0,rtol=0)
                assert torch.linalg.eigvalsh(actual['Q']).min()>0
                assert actual['Q'].shape[-1] == (24 if stage=='primary' else 36)
            candidate=torch.randn(batch,24)
            physical,feasible,residual=loaded.candidate(candidate,m,d['contact_probability'])
            rc,_,ri=qp._certificate(ref,physical/m['variable_scale'],.001)
            assert torch.equal(feasible,rc&m['valid'])
            check(f'candidate_residual_{batch}',residual,ri,0,0)
            assert (physical[:,:12].abs()<=limits['torque_limits']).all()
            assert (physical[:,12:].reshape(batch,4,3)[d['contact_probability']<.5]==0).all()
            # Nonunit gain reconstruction, corrected FF replacement, no second motor multiplication.
            q=d['joint_position'];v=d['joint_velocity'];kp=torch.full_like(q,30);kd=torch.full_like(q,.75)
            motor=torch.full_like(q,.8);pw=torch.full_like(q,1.2);fw=torch.full_like(q,1.6)
            target,ff,pd,requested,bounded=loaded.nominal(action,q,v,kp,kd,motor,pw,fw,limits['torque_limits'])
            parameters={'control_kp':kp,'control_kd':kd,'control_motor_strength':motor,
                'control_feedback_weight':pw,'control_feedforward_weight':fw,'control_torque_limits':limits['torque_limits']}
            ref_total,ref_pd,_,_=control.requested_torque_components(target,ff,q,v,parameters)
            check(f'nominal_requested_{batch}',requested,ref_total)
            check(f'nominal_pd_{batch}',pd,ref_pd)
            check(f'nominal_bounded_{batch}',bounded,control.bounded_nominal_torque(target,ff,q,v,parameters))
            check(f'wrench_clamp_{batch}',clipped,heads.wrench_to_qp_physical(heads.predict_wrench(z,explicit)),0,0)
            for alpha in (0.,.5,1.):
                selected=bounded+alpha*(physical[:,:12]-bounded)
                corrected,available=loaded.feedforward_command(selected,pd,motor,fw)
                check(f'ff_reconstruction_{batch}_{alpha}',pd+motor*fw*corrected,selected)
                assert available.all()
            fixtures={'obs':obs,'history':history,'action':action,'latent':z,'explicit':explicit,'physics_tau':tau,
                'grf_yaw':grf,'wrench_raw':raw,'wrench_clipped':clipped,**{'input_'+k:v for k,v in d.items()},
                **{'primary_'+k:v for k,v in m.items()},**{'recovery_'+k:v for k,v in loaded.recovery(m).items()}}
    torch.save(fixtures,out/'golden_vectors.pt')
    np.savez_compressed(out/'golden_vectors.npz',**{k:v.numpy() for k,v in fixtures.items()})
    (out/'validation.json').write_text(json.dumps({'device':'cpu','torch':torch.__version__,'checks':checks,
        'scope':'TorchScript serialization, exact neural parity, historical primary/recovery assembly, SPD, candidate checks, torque reconstruction. No numerical QP solve/hardware/simulator/C++ execution.'},indent=2))
    versions={}
    for name in ('torch','numpy','cupiqp','pin','warp-lang'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='unavailable'
    (out/'resolved_deployment.json').write_text(json.dumps({'schema':'oct07_hardpact_deployment_v1',
        'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'checkpoint_iteration':20000,
        'historical_source_commit':COMMIT,'provenance':'Archived environment and config byte-match this commit; full original dirty-tree identity unavailable.',
        'training_mode':cfg['qp_update_mode'],'deployment_mode':'every_substep','execution_alpha':1.,
        'latent_mode':'deterministic_mean','qp_config':dataclasses.asdict(qcfg),'saved_qp_config':cfg,
        'rollout_numerical_profile':qp._profile(False),'ppo_numerical_profile':qp._profile(True),
        'export_environment_versions_not_original_training_versions':versions,
        'qp_curriculum_at_checkpoint':state['hard_pact_qp_curriculum'],
        'joint_names':names,'foot_names':env['asset'].get('foot_names'),
        'limits':{k:v.tolist() for k,v in limits.items()},'limits_source':'Saved backend-realized physical_joint_limits; checked against named URDF/asset configuration.',
        'environment':env,'policy':policy,
        'external_requirements':['rigid-body mechanics and state estimator','numerical QP solver','control scheduler, delay queues, sensor validity and safety supervisor'],
        'weights_excluded':'Training critics, latent variance, reconstruction decoder and exploration std are not used by deterministic deployment.'},indent=2))
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix!='.pyc' and p.name!='SHA256.json'}
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'checks':len(checks),'artifact':str(out/'hardpact_oct07_20000.pt'),'bytes':(out/'hardpact_oct07_20000.pt').stat().st_size,'status':'PASS'},indent=2))


if __name__=='__main__': main()
