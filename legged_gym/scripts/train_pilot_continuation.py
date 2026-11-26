"""Explicit candidate continuation, preserving optimizer/curriculum state."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def differences(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((differences(a.get(k),b.get(k),prefix+'.'+k) for k in sorted(a.keys()|b.keys())),[])
    return [] if a==b else [dict(path=prefix,old=a,new=b)]


def resize_progress(state,old_count,new_count):
    """Deterministic stratified downselection; episodes restart, optimizers do not."""
    import torch
    if not 0<new_count<=old_count:raise ValueError('Continuation supports equal or reduced environment counts')
    result=copy.deepcopy(state)
    indices=torch.div(torch.arange(new_count)*old_count,new_count,rounding_mode='floor')
    for owner,fields in result['progress'].items():
        for name,value in fields.items():
            if torch.is_tensor(value) and value.ndim and value.shape[0]==old_count:
                fields[name]=value.index_select(0,indices.to(value.device)).clone()
    # Global iteration/interaction counters and scalar curriculum state stay intact.
    return result,indices


def allowed_change(change,old_count,new_count,production=False):
    if production:
        if change['path']=='.environment.compliance.max_offset_norm':
            return change['old'] in (None,.3) and change['new']==.3
        if change['path'].startswith('.environment.pilot_interactions'):
            return True  # Explicit opt-in, versioned settings printed in migration report.
    if any(change['path'].startswith(p) for p in (
            '.environment.pilot_candidate','.environment.compliance.base_gain','.environment.env.num_envs')):
        return True
    # Buffer totals scale with robots, not contact-force labeling semantics.
    if change['path'] in ('.grf_semantics.contact_capacity_total','.grf_semantics.sensor_count'):
        return change['old']*new_count==change['new']*old_count
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'logs/b1z1/b1z1_pact_lab/Nov19_04-02-42_b1z1_pact_improved')
    p.add_argument('--checkpoint');p.add_argument('--iterations',type=int)
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--gpu',default='cuda:1');p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--candidate-config',type=Path);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--production-interactions',action='store_true',help='Use normal PACT path, not the old experimental wrapper')
    p.add_argument('--smoke-full-mixture',action='store_true',help='Test-only immediate final mixture')
    args=p.parse_args()
    if args.iterations is None:args.iterations=50 if args.production_interactions else 1000
    if args.smoke_full_mixture and not args.production_interactions:p.error('Full-mixture smoke requires --production-interactions')
    if args.iterations<1: p.error('iterations must be positive')
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    os.environ['SIMULATOR']='isaaclab_b1z1_pact'
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
    import torch
    from legged_gym.scripts.eval_b1z1_pilot_compliance import util,restore_sections
    from legged_gym.scripts.eval_b1z1_exp1 import util as shared
    from legged_gym.envs.b1z1.pilot_continuation import Candidate,configure,environment_class
    from legged_gym.utils.helpers import class_to_dict,set_seed
    path,number=util.resolve_checkpoint(args.run,args.checkpoint)
    state=torch.load(path,map_location='cpu',weights_only=False)
    saved=state['training_state']['contract']
    spec=importlib.util.spec_from_file_location('candidate_archive',path.parent/'b1z1_pact_config.py')
    archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)
    cfg=shared.materialize(archive.B1Z1PACTCfg());train=shared.materialize(archive.B1Z1PACTCfgPPO())
    from legged_gym.envs.b1z1.pilot_interactions import prepare_restore
    prepare_restore(cfg,saved['environment'])
    restore_sections(cfg,saved['environment']);restore_sections(train,saved['training'])
    cfg._compliance_layout_added=True
    old_count=cfg.env.num_envs
    if not 0<args.num_envs<=old_count:p.error('num-envs must be positive and <= saved count')
    cfg.env.num_envs=args.num_envs
    if args.production_interactions:
        from legged_gym.envs.b1z1.pilot_interactions import InteractionCfg,configure as production_configure
        if not saved['environment'].get('pilot_interactions',{}).get('enabled',False):
            cfg.pilot_interactions=InteractionCfg()
            cfg.pilot_interactions.mixture_start_iteration=state['iteration']
        cfg.pilot_interactions.smoke_full_mixture=args.smoke_full_mixture
        cfg.pilot_interactions.diagnostics=True
        # This explicit migration records the currently executed source, not
        # stale hashes restored from its input checkpoint. Changes stay audited.
        cfg.pilot_interactions.source_hashes=None
        production_configure(cfg)
    else:
        candidate=Candidate(**(json.loads(args.candidate_config.read_text()) if args.candidate_config else {}))
        configure(cfg,candidate)
        if hasattr(cfg,'pilot_interactions'):cfg.pilot_interactions.enabled=False
    train.runner.max_iterations=args.iterations;train.runner.save_interval=100
    train.runner.resume=False;train.runner.run_name='pilot_candidate_continuation'
    train.runner.experiment_name='pilot_candidate_continuation'
    config=class_to_dict(train)
    report=dict(command=[sys.executable,*sys.argv],checkpoint=str(path),filename_iteration=number,
        next_iteration=state['iteration'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        requested_updates=args.iterations,training_seed=train.seed,num_envs=cfg.env.num_envs,
        saved_num_envs=old_count,interaction_budget=args.iterations*cfg.env.num_envs*train.runner.num_steps_per_env,
        rollout_steps=train.runner.num_steps_per_env,force_curriculum=state.get('force_curriculum_state'),
        environment_changes=differences(saved['environment'],class_to_dict(cfg)),
        status='preflight',fresh_physical_episodes=True,optimizer_restored=False)
    dump(args.output_dir/'resume_report.json',report)
    dump(args.output_dir/'resolved_environment.json',class_to_dict(cfg))
    dump(args.output_dir/'resolved_training.json',config)
    (args.output_dir/'source.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    # Evaluation uses archived class definitions plus the saved resolved contract.
    import shutil
    shutil.copy2(path.parent/'b1z1_pact_config.py',args.output_dir/'b1z1_pact_config.py')
    if args.preflight_only: return
    env=None
    try:
        from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
        from rsl_rl.runners.b1z1_pact_runner import B1Z1PACTRunner
        from legged_gym.utils.pilot_candidate_diagnostics import Distributions
        set_seed(train.seed)
        cls=B1Z1PACT if args.production_interactions else environment_class(B1Z1PACT,training=True)
        env=cls(cfg,class_to_dict(cfg.sim),args.gpu,True)
        recorder=Distributions(args.output_dir,env.dt)
        class Runner(B1Z1PACTRunner):
            def _log(self,iteration,*rest,**kwargs):
                super()._log(iteration,*rest,**kwargs)
                recorder.flush(iteration)
                # Scalar events remain authoritative; also export compact metrics.
                metrics=rest[1]
                row={'iteration':iteration,**{k:float(v) for k,v in metrics.items()
                    if isinstance(v,(float,int)) or torch.is_tensor(v) and v.numel()==1}}
                with (args.output_dir/'training_scalars.jsonl').open('a') as f:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
        runner=Runner(env,config,str(args.output_dir),args.gpu)
        actual=copy.deepcopy(runner._training_contract)
        changes=differences(saved,actual)
        unexplained=[d for d in changes if not allowed_change(d,old_count,args.num_envs,args.production_interactions)]
        report['contract_changes']=changes
        if unexplained:
            raise ValueError('Unapproved continuation contract differences: '+json.dumps(unexplained))
        # Narrow, explicit experiment migration: the normal validator still checks
        # ownership, moment shapes, RNG, progress shapes and checkpoint boundaries.
        # No checkpoint is rewritten; subsequent saves use the candidate contract.
        migrated=dict(state)
        migrated['training_state'],indices=resize_progress(state['training_state'],old_count,args.num_envs)
        migrated['training_state']['contract']=actual
        report['progress_migration']=dict(rule='floor(i*old_count/new_count); same selected rows for every per-environment progress tensor',
            old_count=old_count,new_count=args.num_envs,indices=indices.tolist(),
            global_counters_preserved=True,optimizer_tensors_unchanged=True)
        # A temporary copy bridges only audited config/progress differences. The
        # original checkpoint is immutable; the native loader/validator is reused.
        with tempfile.TemporaryDirectory(prefix='pilot-candidate-resume-') as temporary:
            bridge=Path(temporary)/'resume.pt';torch.save(migrated,bridge)
            runner.load(str(bridge),load_optimizer=True)
        runner._resume_fresh_episodes()
        if args.production_interactions:
            # Initialization/reset warmups are not training interactions.
            env.interactions.stats.zero_();env.interactions.counts.zero_()
            env.interactions.counts.scatter_add_(0,env.interactions.family,
                torch.ones(env.num_envs,device=env.device))
            env.interactions.max_offset.zero_()
        gradient_bad=torch.zeros((),device=args.gpu)
        if args.production_interactions:
            def check_gradient(g):
                gradient_bad.add_((~torch.isfinite(g)).sum())
                return g
            for parameter in runner.alg.actor_critic.parameters():
                if parameter.requires_grad:parameter.register_hook(check_gradient)
        # Exclude constructor/reset warmup intervals from training distributions.
        env.candidate_recorder=recorder
        report.update(status='training',optimizer_restored=True,
            resumed_iteration=runner.current_learning_iteration,
            curriculum_beta=float(env._staged_force_curriculum.beta))
        dump(args.output_dir/'resume_report.json',report)
        runner.learn(args.iterations,init_at_random_ep_len=False)
        scalar_rows=[json.loads(line) for line in (args.output_dir/'training_scalars.jsonl').read_text().splitlines()]
        with (args.output_dir/'training_scalars.csv').open('w') as f:
            fields=sorted(set().union(*(row.keys() for row in scalar_rows)))
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(scalar_rows)
        report.update(status='complete',completed_updates=args.iterations,
            final_iteration=runner.current_learning_iteration)
        if args.production_interactions:
            report['interaction_smoke']=env.interactions.summary()
            report['nonfinite_gradient_elements']=gradient_bad.item()
        dump(args.output_dir/'resume_report.json',report)
        (args.output_dir/'COMPLETE').write_text('Requested continuation completed.\n')
    except BaseException:
        report.update(status='blocked',error=traceback.format_exc())
        dump(args.output_dir/'resume_report.json',report)
        (args.output_dir/'ERROR.txt').write_text(report['error'])
        traceback.print_exc();sys.stderr.flush()
        raise
    finally:
        if env is not None: env.simulator._app_launcher.app.close()


if __name__=='__main__': main()
