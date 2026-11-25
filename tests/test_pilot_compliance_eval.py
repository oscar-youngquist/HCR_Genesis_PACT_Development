"""CPU-only protocol tests; no simulator or training imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'legged_gym/utils/pilot_compliance_eval.py'
spec=importlib.util.spec_from_file_location('rss_test_util',path)
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)


class ProtocolTests(unittest.TestCase):
    def test_numeric_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for name in ('model_9.pt','model_100.pt','model_latest.pt'): (p/name).touch()
            self.assertEqual(u.resolve_checkpoint(p)[1],100)
            self.assertEqual(u.resolve_checkpoint(p,'9')[1],9)
            with self.assertRaises(FileNotFoundError): u.resolve_checkpoint(p,'10')

    def test_profiles(self):
        c=u.EVAL_CONFIG
        rows=[u.condition(n,c) for n in u.CONDITIONS]
        self.assertEqual(rows[0]['profile'],[0,0,0,0,0,.5,0,0,0,0,0,0])
        self.assertEqual(rows[2]['profile'],[0,0,0,0,.5,0,0,0,0,0,0,0])
        for i in (4,6):
            self.assertEqual(rows[i]['profile'][7],.5)
            self.assertEqual(rows[i+1]['profile'][6:],[0]*6)
            self.assertEqual(rows[i]['profile'][:6],rows[i+1]['profile'][:6])

    def test_manifest_pairing_and_partial_batches(self):
        a=u.manifests(2,3,u.CONDITIONS,u.EVAL_CONFIG)
        b=u.manifests(2,3,list(reversed(u.CONDITIONS)),u.EVAL_CONFIG)
        key=lambda r:(r['trial_id'],r['condition'])
        self.assertEqual(sorted(a,key=key),sorted(b,key=key))
        self.assertEqual(len({r['pair_hash'] for r in a}),3)
        self.assertEqual([len(x) for x in u.batches(a,10)],[10,10,4])
        self.assertNotEqual(a[0]['pair_hash'],u.manifests(1,1,u.CONDITIONS,u.EVAL_CONFIG)[0]['pair_hash'])

    def test_ramps_boundaries_and_rotated_force(self):
        c=u.EVAL_CONFIG
        t=np.array([4.99,5,5.01,6,9.99,10,10.01,11,11.01])
        np.testing.assert_allclose(u.force_scale(t,c),[0,0,.01,1,1,1,.99,0,0])
        # Heading rotation leaves vertical unchanged, rotates forward into world y.
        f=np.array([20.,0.,0.]);yaw=np.pi/2
        r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        np.testing.assert_allclose(r@f,[0,20,0],atol=1e-12)

    def test_seed_weighting_and_paired_uncertainty(self):
        rows=[]
        for seed,count,val in ((0,1,1.),(1,3,3.)):
            for trial in range(count):
                for name,d in (('restricted_base_on',val),('restricted_base_off',0.)):
                    rows.append(dict(seed=seed,trial_id=trial,condition=name,pair_hash=str(trial),
                        failed=False,setup_tracking_failed=False,**{k:d for k in u.METRICS}))
        result=u.aggregate(rows)
        on=next(r for r in result['per_condition'] if r['condition']=='restricted_base_on')
        self.assertEqual(on['metrics']['base_guidance_velocity_m_s']['mean'],2.)
        pair=u.paired_contrasts(rows,20)[2]
        self.assertEqual(pair['mean_difference'],2.)
        self.assertEqual(pair['n_pairs'],4)
        self.assertEqual(pair['ci95'],[2.,2.])
        rows[-1]['pair_hash']='incorrect'
        with self.assertRaises(ValueError): u.paired_contrasts(rows,20)

    def test_empty_statistics(self):
        self.assertEqual(u.statistics([None,float('nan')]),dict(mean=None,sd=None,n=0))
        self.assertIsNone(u.statistics([2.])['sd'])
        c=copy.deepcopy(u.EVAL_CONFIG);c['force_off_s']=4
        with self.assertRaises(ValueError): u.validate_config(c)

    def test_terminal_sample_excluded_not_replaced(self):
        c=u.EVAL_CONFIG;trial=u.manifests(0,1,[u.CONDITIONS[0]],c)[0]
        n=3;z=np.zeros((n,3));d=dict(time=np.arange(1,n+1)*.02,
            alive_for_metrics=np.array([True,True,False]),failure=np.array([False,False,True]),
            failure_reasons=np.zeros((n,8),bool),ee_task=z.copy(),reference_task=z.copy(),
            nominal_task=z.copy(),base_world=z.copy(),base_measured=z.copy(),base_reference=z.copy(),
            workspace_margin=z.copy(),joint_margin=np.ones(n),gates=z.copy(),joint_gate=z.copy(),
            workspace_gate=z.copy(),collision_gate=z.copy(),clipped=np.zeros(n,bool),applied_force_world=z.copy())
        d['ee_task'][:,2]=[.1,.2,100.];d['failure_reasons'][-1,0]=True
        result=u.summarize(d,trial,c,.02)
        self.assertAlmostEqual(result['ee_tracking_error_m'],.15)
        self.assertTrue(result['failed']);self.assertEqual(result['failure_reasons'],['contact'])
        self.assertEqual(result['valid_samples'],2);self.assertAlmostEqual(result['observed_seconds'],.06)
        self.assertIsNone(result['signed_displacement_m'])
        # No available baseline/loaded window is not a zero response.
        self.assertTrue(result['setup_tracking_failed'])

    def test_native_film_history_partial_reset(self):
        import torch
        from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
        torch.set_num_threads(1)
        actor=ActorCriticB1Z1PACT(93,93,17,186,latent_dim=4,actor_layers=(16,),critic_layers=(16,),
            context_layers=(16,),explicit_decoder_layers=(16,),force_decoder_layers=(16,),
            grf_decoder_layers=(16,),film_hidden_dim=8,film_v02_enabled=True).eval()
        actor.requires_grad_(False);obs=torch.randn(2,93);hist=torch.randn(2,186)
        weights={k:v.clone() for k,v in actor.state_dict().items()}
        with torch.inference_mode():
            first=actor.act_inference(obs,hist).clone()
            actor.act_inference(obs,hist);old=actor._film_history.clone()
            actor.reset(torch.tensor([True,False]))
            self.assertTrue((actor._film_history[0]==0).all())
            torch.testing.assert_close(actor._film_history[1],old[1])
            after=actor.act_inference(obs,hist)
            torch.testing.assert_close(after[0],first[0])
            torch.testing.assert_close(actor._film_history[1,:-1],old[1,1:])
        for k,v in actor.state_dict().items(): torch.testing.assert_close(v,weights[k])

    def test_capture_precedes_reset_and_force_is_single_publish(self):
        import os
        from unittest.mock import patch
        from types import SimpleNamespace as NS
        import torch
        with patch.dict(os.environ,{'SIMULATOR':'isaaclab_b1z1_pact'}):
            script=path.parents[1]/'scripts/eval_b1z1_pilot_compliance.py'
            module_spec=importlib.util.spec_from_file_location('rss_entry_test',script)
            entry=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(entry)
            class Base:
                def check_termination(self):
                    self.fail_buf=torch.tensor([True,False]);self.reset_buf=self.fail_buf.clone()
                    self.contact_fail_buf=self.fail_buf.clone()
                    for k in ('roll_fail_buf','pitch_fail_buf','height_min_fail_buf','height_max_fail_buf','time_out_buf'):
                        setattr(self,k,torch.zeros(2,dtype=torch.bool))
            cls=entry.make_environment_class(Base,u.EVAL_CONFIG);env=cls()
        z=torch.zeros(2,3);false=torch.zeros(2,dtype=torch.bool)
        s=NS(base_pos=z.clone(),base_quat=torch.tensor([[0.,0.,2**-.5,2**-.5]]).repeat(2,1),
            ee_pos=torch.tensor([[1.,0.,.7]]).repeat(2,1),dof_pos=torch.zeros(2,19),
            dof_vel=torch.zeros(2,19),dof_pos_limits=torch.tensor([-3.,3.]).repeat(19,1),
            torques=torch.zeros(2,19),_gripper_index=1,_base_link_index=0,
            _external_force_world=torch.zeros(2,2,3),_external_torque_world=torch.zeros(2,2,3))
        publications=[]
        def publish(force):
            publications.append(force.clone());s._external_force_world[:,1]=force
        s.apply_ee_force=publish;env.simulator=s
        target=NS(position=s.ee_pos.clone(),velocity=z.clone(),base=z.clone(),gates=z.clone(),
            workspace=z.clone(),joint=z.clone(),collision=z.clone(),clipped=false.clone())
        env.compliance=NS(target=target,reference=NS(recovering=false.clone(),collect_diagnostics=False),recovery_failed=false.clone(),
            profile=torch.zeros(2,12),bounds=torch.tensor([[.1,2.],[-2.,2.],[-2.,2.]]),
            nominal=lambda advance:(z.clone(),z.clone()),center=lambda:z.clone(),
            measured_velocity=lambda:z.clone(),measured_base=lambda:z.clone(),wrench_count=torch.zeros(2))
        env.ready=True;env.capture=True;env.eval_time=7.;env.eval_active=torch.ones(2,dtype=torch.bool)
        env.eval_force_direction=torch.tensor([[1.,0.,0.]]).repeat(2,1)
        env.eval_amplitude=torch.ones(2)*20;env.ee_force_ext_world=z.clone();env.commands=torch.zeros(2,6)
        env._push_gripper(torch.arange(2))
        self.assertEqual(len(publications),1)
        torch.testing.assert_close(publications[0],torch.tensor([[0.,20.,0.]]).repeat(2,1),atol=1e-5,rtol=0)
        env.compliance.wrench_count+=1;env.check_termination();captured=env.transition
        s.ee_pos.zero_();env.reset_buf.zero_();env.compliance.target.position.zero_()
        self.assertTrue(captured['failure'][0]);self.assertFalse(captured['alive_for_metrics'][0])
        self.assertTrue(captured['alive_for_metrics'][1]);self.assertEqual(float(captured['ee_world'][0,2]),float(torch.tensor(.7)))
        self.assertTrue((captured['reference_advances']==1).all())


if __name__=='__main__': unittest.main()
